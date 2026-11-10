#!/usr/bin/env python3
"""VSI-Bench eval for GEO-Memory (Stage B): memory over ALL sampled frames, Qwen visual tokens only for the
frames chosen by select_qwen_frame_indices(); Memory and Qwen frame budgets are configurable.

Pipeline per question (same protocol as training / GeoMemoryTrainingModel.generate_answer):
  frames -> frozen Qwen3.5 ViT (post-merger) -> GeoMemory recurrent state M_k (all N frames)
  -> visual fusion F'_k = Fusion(F_k, M_k) for all N frames -> frozen Qwen3.5 greedy decode.
"""
import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/scripts/evaluation")
from eval_vsi_mra import (VSIBenchDataset, score_record, aggregate_results, ss_sample_frame_indices, patch_qwen3_5_flash_attention)  # noqa: E402

patch_qwen3_5_flash_attention()
from geomemory.config import GeoMemoryConfig  # noqa: E402
from geomemory.training_model import GeoMemoryTrainingModel  # noqa: E402
from geomemory.train import _load_geomemory_checkpoint  # noqa: E402
from geomemory.position import grid_to_patch_xy  # noqa: E402
from geomemory.qwen_protocol import QwenPromptBuilder  # noqa: E402
from geomemory.state import GeoMemoryState  # noqa: E402
from geomemory.fusion import select_qwen_frame_indices  # noqa: E402

DEFAULT_QWEN = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B"
VSI = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/benchmark/VSI-Bench"


MAX_PIXELS = 451584   # per-frame inference budget (set from --max_pixels)


def fit_pixels(img, max_pixels=None, align=32):
    """Full-FOV resize so w*h <= max_pixels, both sides multiples of `align`."""
    max_pixels = MAX_PIXELS if max_pixels is None else max_pixels
    if not max_pixels:
        return img
    w, h = img.size
    s = min(1.0, (max_pixels / (w * h)) ** 0.5)
    nw, nh = max(align, int(w * s) // align * align), max(align, int(h * s) // align * align)
    return img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.BICUBIC)


@torch.no_grad()
def vit_features(lm, processor, frames, device):
    frames = [fit_pixels(f) for f in frames]
    p = processor.image_processor(images=frames, return_tensors="pt", do_resize=False)
    pv, grid = p["pixel_values"].to(device, torch.bfloat16), p["image_grid_thw"].to(device)
    out = lm.model.get_image_features(pv, grid, return_dict=True)
    pooled = out.pooler_output
    if not isinstance(pooled, (list, tuple)):
        counts = (grid.prod(-1) // 4).tolist()
        pooled = torch.split(pooled, counts)
    return list(pooled), grid.cpu()


@torch.no_grad()
def prepare(model, builder, processor, frames, question, device, strict=False, qwen_idx=None):
    """Frames -> fused visual embeds injected at <|image_pad|> + native 3D M-RoPE positions."""
    feats, grid = vit_features(model.language_model, processor, frames, device)
    return prepare_from_features(model, builder, feats, grid, question, device, strict, qwen_idx)


@torch.no_grad()
def prepare_from_features(model, builder, feats, grid, question, device, strict=False, qwen_idx=None):
    """feats: list of T [N,D] post-merger ViT features; grid: [T,3].  Shared by eval and parity tests."""
    n_tok = [f.shape[0] for f in feats]
    T = len(feats)
    pf = torch.stack(feats)[None].to(torch.bfloat16) if len(set(n_tok)) == 1 else None
    if pf is None:
        raise ValueError("frames must share one grid")
    N = n_tok[0]
    xy = torch.stack([grid_to_patch_xy(grid[t], N, 2) for t in range(T)])[None].to(device, torch.bfloat16)
    core = model.geomemory(pf, patch_xy=xy)
    mem = core.memory_history
    assert mem.shape[1] == T, f"memory updated {mem.shape[1]} times for {T} frames"
    cfg = model.cfg
    # Same selector as the collator: Qwen sees K frames, each fused with the memory AT that frame.
    idx = list(qwen_idx) if qwen_idx is not None else select_qwen_frame_indices(T, cfg.qwen_frame_indices, cfg.qwen_full_length)
    if strict:
        assert T == cfg.qwen_full_length, f"strict mode needs {cfg.qwen_full_length} frames, got {T}"
        assert idx == list(cfg.qwen_frame_indices), (idx, cfg.qwen_frame_indices)
    idx_t = torch.tensor(idx, device=device)
    sel_pf, sel_mem = pf[:, idx_t], mem[:, idx_t]
    sel_geometry = None if core.patch_geometry is None else {
        key: value[:, idx_t] for key, value in core.patch_geometry.items()
    }
    fused = model.geomemory.visual_fusion(
        sel_pf, GeoMemoryState(memory=sel_mem, pose=None, step=torch.tensor([T - 1], device=device)), None,
        geometry=sel_geometry)
    K = len(idx)
    frames_llm = model.visual_to_llm(fused)[0].reshape(K * N, -1)
    sel_tokens = [n_tok[i] for i in idx]
    ids, _ = builder.build(question, None, sel_tokens)
    ids_t = torch.tensor([ids], device=device)
    mm = (ids_t == builder.image_pad_id)
    lm = model.language_model
    emb = lm.get_input_embeddings()(ids_t)
    if int(mm.sum()) != frames_llm.shape[0]:
        raise ValueError("placeholder/feature count mismatch")
    emb = emb.masked_scatter(mm.unsqueeze(-1), frames_llm.to(emb.dtype))
    if getattr(model, "numeric", None) is not None:
        ev_slots = (ids_t == model.evidence_id)
        qvalid = (~mm) & (~ev_slots)
        state = GeoMemoryState(memory=mem[:, -1], pose=None if core.pose_history is None else core.pose_history[:, -1],
                               step=torch.tensor([T - 1], device=device))
        full_mask = torch.ones(pf.shape[:-1], device=device, dtype=torch.bool)
        num = model.numeric(
            lm.get_input_embeddings()(ids_t).detach(), qvalid, state,
            patch_features=pf, patch_geometry=core.patch_geometry, patch_mask=full_mask,
            frame_mask=torch.ones(pf.shape[:2], device=device, dtype=torch.bool),
            at_t=torch.tensor([T - 1], device=device),
            geometry_embed=(None if core.patch_geometry is None else model.geomemory.visual_fusion.encode_geometry(
                core.patch_geometry)[0]),
        )
        if (model.cfg.numeric_inject_evidence and model.cfg.direct_evidence_enabled
                and model.cfg.numeric_evidence_slots > 0):
            emb = emb.masked_scatter(ev_slots.unsqueeze(-1), num["evidence"].to(emb.dtype))
    grid = grid[idx_t.cpu()].to(device).reshape(-1, 3)   # grids of the Qwen-visible frames only
    pos, _ = model._qwen_decoder().get_rope_index(
        ids_t, mm_token_type_ids=mm.to(torch.int32), image_grid_thw=grid,
        attention_mask=torch.ones_like(ids_t))
    if strict:
        assert int(mm.sum()) == K * N and K == len(cfg.qwen_frame_indices)
    return dict(ids=ids_t, mm=mm, emb=emb, pos=pos, grid=grid, idx=idx,
                numeric=(num if getattr(model, 'numeric', None) is not None else None))


@torch.no_grad()
def greedy_decode(model, builder, P, max_new_tokens, return_logits=False, forced=None):
    lm = model.language_model
    dec = model._qwen_decoder()
    embed = lm.get_input_embeddings()
    out = dec(inputs_embeds=P["emb"], position_ids=P["pos"], use_cache=True, return_dict=True)
    cache, next_pos = out.past_key_values, P["pos"].max() + 1
    logits = lm.lm_head(out.last_hidden_state[:, -1])
    toks, all_logits = [], []
    for step in range(max_new_tokens):
        all_logits.append(logits.float().cpu())
        nxt = int(logits.argmax(-1).item()) if forced is None else forced[step]
        if forced is None and (nxt == builder.im_end_id or nxt == builder.tokenizer.eos_token_id):
            break
        toks.append(nxt)
        out = dec(inputs_embeds=embed(torch.tensor([[nxt]], device=P["emb"].device)),
                  position_ids=next_pos.view(1, 1, 1).expand(3, 1, 1),
                  past_key_values=cache, use_cache=True, return_dict=True)
        cache, next_pos = out.past_key_values, next_pos + 1
        logits = lm.lm_head(out.last_hidden_state[:, -1])
    return (toks, all_logits) if return_logits else toks


@torch.no_grad()
def answer(model, builder, processor, frames, question, device, max_new_tokens, strict=False, qwen_idx=None):
    P = prepare(model, builder, processor, frames, question, device, strict, qwen_idx)
    if not getattr(model, "_logged_eval_frame_counts", False):
        print(f"Prepared VSI input: Memory={len(frames)}, Qwen={len(P['idx'])}, "
              f"visual_tokens={int(P['mm'].sum())}, indices={P['idx']}", flush=True)
        model._logged_eval_frame_counts = True
    toks = greedy_decode(model, builder, P, max_new_tokens)
    return builder.tokenizer.decode(toks, skip_special_tokens=True).strip()


def load_all(ckpt, qwen_path, device, memory_frames=None, qwen_frames=None):
    from transformers import AutoProcessor, AutoTokenizer
    from geomemory.train import _load_language_model
    from geomemory.qwen_fa2 import install_qwen35_fa2_mrope_patch
    from types import SimpleNamespace
    cfg = GeoMemoryConfig.from_json(os.path.join(ckpt, "geomemory_config.json"))
    if memory_frames is not None:
        if memory_frames < 1 or memory_frames > cfg.max_time_steps:
            raise ValueError("Memory frame budget is outside the supported time range")
        cfg.qwen_full_length = memory_frames
    if qwen_frames is not None:
        if qwen_frames < 1 or qwen_frames > cfg.qwen_full_length:
            raise ValueError("Qwen frame count must be within the Memory frame budget")
        cfg.qwen_frame_indices = (
            [cfg.qwen_full_length - 1] if qwen_frames == 1 else
            [round(i * (cfg.qwen_full_length - 1) / (qwen_frames - 1)) for i in range(qwen_frames)]
        )
    lm = _load_language_model(qwen_path, SimpleNamespace(bf16=True), True, "flash_attention_2").to(device).eval()
    install_qwen35_fa2_mrope_patch(lm.model.language_model)
    lm.requires_grad_(False)
    model = GeoMemoryTrainingModel(cfg, language_model=lm, freeze_language_model=True)
    _load_geomemory_checkpoint(model, ckpt)
    model.geomemory.to(device, torch.bfloat16)
    model.visual_to_llm.to(device, torch.bfloat16)
    model.eval()
    tok = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)
    ev = cfg.numeric_evidence_slots if (cfg.use_numeric_readout and cfg.numeric_inject_evidence) else 0
    model.evidence_id = tok.convert_tokens_to_ids('<|vision_pad|>')
    if getattr(model, 'numeric', None) is not None:
        model.numeric.to(device, torch.bfloat16)
    return model, processor, QwenPromptBuilder(tok, ev)


def plan_frames(total, memory_frames, qwen_frames=32):
    """Memory frame indices and positions of the Qwen frames inside them.

    The Qwen frames are ALWAYS the uniform `qwen_frames` set of the video, so a 32- and a 64-frame
    Memory run see identical visual frames; extra Memory frames are interpolated between them.
    """
    q = ss_sample_frame_indices(total, qwen_frames)
    if memory_frames <= len(q) or total < memory_frames:
        return ss_sample_frame_indices(total, memory_frames), None
    chosen = set(q.tolist())
    cands = sorted(set(((q[:-1] + q[1:]) // 2).tolist() + ((3 * q[:-1] + q[1:]) // 4).tolist()
                       + ((q[:-1] + 3 * q[1:]) // 4).tolist()) - chosen)
    need = memory_frames - len(chosen)
    if len(cands) < need:
        return ss_sample_frame_indices(total, memory_frames), None
    if len(cands) > need:
        cands = [cands[i] for i in np.linspace(0, len(cands) - 1, need).round().astype(int)]
    mem = np.array(sorted(chosen | set(cands)), dtype=np.int64)
    if len(mem) != memory_frames:
        return ss_sample_frame_indices(total, memory_frames), None
    pos = {int(v): i for i, v in enumerate(mem)}
    return mem, [pos[int(v)] for v in q]


class PlannedVSIDataset(VSIBenchDataset):
    def __init__(self, *a, qwen_frames=32, **k):
        super().__init__(*a, **k)
        self.qwen_frames = qwen_frames

    def __getitem__(self, idx):
        import cv2
        from eval_vsi_mra import build_vsi_prompt
        row = self.rows[idx]
        cap = cv2.VideoCapture(os.path.join(self.vsi_root, row["dataset"], f"{row['scene_name']}.mp4"))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mem, qpos = plan_frames(total, self.max_num_frames, self.qwen_frames)
        frames = []
        for fi in mem:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        if len(frames) != len(mem):
            qpos = None
        return {"id": row["id"], "scene_name": row["scene_name"],
                "question": build_vsi_prompt(row["question"], row["question_type"], row.get("options")),
                "answer": str(row["ground_truth"]), "question_type": row["question_type"],
                "frames": frames, "qwen_idx": qpos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="stage_b dir with model.safetensors + geomemory_config.json")
    ap.add_argument("--qwen_path", default=DEFAULT_QWEN)
    ap.add_argument("--vsi_jsonl", default=f"{VSI}/raw/test.jsonl")
    ap.add_argument("--vsi_root", default=f"{VSI}/videos")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_num_frames", type=int, default=32, help="Maximum Memory frames")
    ap.add_argument("--qwen_frames", type=int, default=32, help="Number of selected Qwen visual frames")
    ap.add_argument("--strict", action="store_true", help="assert memory length and selected frames match the checkpoint configuration")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--max_pixels", type=int, default=451584, help="per-frame pixel budget (0 = unlimited)")
    ap.add_argument("--fixed_qwen_set", type=int, default=1, help="Qwen frames = same uniform set for any Memory length")
    args = ap.parse_args()
    global MAX_PIXELS
    MAX_PIXELS = args.max_pixels
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")

    model, processor, builder = load_all(
        args.ckpt, args.qwen_path, device, args.max_num_frames, args.qwen_frames,
    )
    if args.rank == 0:
        model.cfg.save_json(os.path.join(args.out_dir, "eval_geomemory_config.json"))
        with open(os.path.join(args.out_dir, "eval_protocol.json"), "w") as handle:
            json.dump({"memory_frames": args.max_num_frames, "qwen_frames": args.qwen_frames,
                       "qwen_frame_indices": model.cfg.qwen_frame_indices,
                       "max_pixels": args.max_pixels, "fixed_qwen_set": bool(args.fixed_qwen_set)}, handle, indent=2)

    DS = PlannedVSIDataset if args.fixed_qwen_set else VSIBenchDataset
    kw = {"qwen_frames": args.qwen_frames} if args.fixed_qwen_set else {}
    ds = DS(args.vsi_jsonl, args.vsi_root, args.max_num_frames,
            args.max_samples, args.rank, args.world, **kw)
    print(f"[rank={args.rank}] samples={len(ds)} Memory={args.max_num_frames} Qwen={args.qwen_frames}", flush=True)
    out_path = os.path.join(args.out_dir, f"predictions_rank{args.rank}.json")
    records = []
    if os.path.exists(out_path):
        records = json.load(open(out_path))
    done = {r["id"] for r in records}
    print(f"[rank={args.rank}] resume: {len(done)} done", flush=True)
    import time
    for i in tqdm(range(len(ds)), desc=f"rank{args.rank}", dynamic_ncols=True):
        if ds.rows[i]["id"] in done:
            continue
        item = None
        for attempt in range(8):  # transient thread-exhaustion in cv2 -> 0 frames
            try:
                item = ds[i]
                if len(item["frames"]) > 0:
                    break
            except ValueError:
                pass
            time.sleep(3 * (attempt + 1))
        if item is None or len(item["frames"]) == 0:
            print(f"[rank={args.rank}] SKIP id={ds.rows[i]['id']} (frames unreadable)", flush=True)
            continue
        try:
            pred = answer(model, builder, processor, item["frames"], item["question"],
                          device, args.max_new_tokens, args.strict, item.get("qwen_idx"))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            pred = ""
        records.append({"id": item["id"], "scene_name": item["scene_name"],
                        "question": item["question"], "answer": item["answer"],
                        "prediction": pred, "question_type": item["question_type"],
                        "score": score_record(item["question_type"], pred, item["answer"])})
        if len(records) % args.save_every == 0 or i + 1 == len(ds):
            json.dump(records, open(out_path, "w"), ensure_ascii=False, indent=2)
    if records:
        m = aggregate_results(records); m["num_records"] = len(records)
        json.dump(m, open(os.path.join(args.out_dir, f"metrics_rank{args.rank}.json"), "w"), indent=2)
        print(json.dumps(m))


if __name__ == "__main__":
    main()
