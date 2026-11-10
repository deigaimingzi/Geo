#!/usr/bin/env python3
"""Multi-donor validation of Delta_State on v3 Stage B (original fusion).

For each sample, test N different donor memories:
  - K random donors from OTHER scenes (same observed prefix length)
  - 1 cross-dataset donor (e.g. arkitscenes -> scannet)
  - 1 random Gaussian noise memory (same shape)
  - 1 zero memory

Reports the distribution of Delta_State = NLL_donor - NLL_real across all
samples and donors, to verify whether the near-zero Delta_State is robust.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_vsi_multidonor.py \
        --ckpt output/v3/stage_b --out_dir output/v3/stage_b/multidonor \
        --rank 0 --world 8 --max_samples 200 --num_donors 5
"""
import argparse, json, os, sys, random, time, math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/scripts/evaluation")
from eval_vsi_mra import VSIBenchDataset, score_record

from geomemory.config import GeoMemoryConfig
from geomemory.training_model import GeoMemoryTrainingModel
from geomemory.train import _load_geomemory_checkpoint
from geomemory.position import grid_to_patch_xy
from geomemory.qwen_protocol import QwenPromptBuilder
from geomemory.state import GeoMemoryState
from geomemory.fusion import select_qwen_frame_indices
from safetensors.torch import load_file

DEFAULT_QWEN = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B"
DEFAULT_VSI_JSONL = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/benchmark/VSI-Bench/raw/test.jsonl"
DEFAULT_VSI_ROOT = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/benchmark/VSI-Bench/videos"


class OriginalFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.visual_norm = nn.LayerNorm(cfg.vision_dim)
        self.memory_norm = nn.LayerNorm(cfg.memory_dim)
        self.query_projection = nn.Linear(cfg.vision_dim, cfg.memory_dim)
        self.cross_attention = nn.MultiheadAttention(
            cfg.memory_dim, cfg.visual_fusion_heads, batch_first=True)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(cfg.memory_dim),
            nn.Linear(cfg.memory_dim, cfg.vision_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(cfg.visual_fusion_residual_init)))

    def forward(self, visual_tokens, state, visual_mask=None, geometry=None, geometry_embedding=None):
        query = self.query_projection(self.visual_norm(visual_tokens))
        memory = self.memory_norm(state.memory)
        if query.ndim == 4:
            B, T, N, _ = query.shape
            if memory.ndim == 3:
                memory = memory.unsqueeze(1).expand(B, T, -1, -1)
            query_flat = query.reshape(B * T, N, -1)
            memory_flat = memory.reshape(B * T, memory.shape[-2], memory.shape[-1])
            correction, _ = self.cross_attention(query_flat, memory_flat, memory_flat, need_weights=False)
            correction = correction.reshape(B, T, N, -1)
        else:
            correction, _ = self.cross_attention(query, memory, memory, need_weights=False)
        correction = self.output_projection(correction)
        if visual_mask is not None:
            correction = correction * visual_mask.to(correction.dtype).unsqueeze(-1)
        return visual_tokens + self.residual_scale.tanh() * correction


def load_original_fusion(cfg, ckpt_path, device):
    sd = load_file(os.path.join(ckpt_path, 'model.safetensors'))
    fusion = OriginalFusion(cfg).to(device, torch.bfloat16)
    p = 'geomemory.visual_fusion.'
    fusion.visual_norm.weight.data = sd[p + 'visual_norm.weight'].to(device, torch.bfloat16)
    fusion.visual_norm.bias.data = sd[p + 'visual_norm.bias'].to(device, torch.bfloat16)
    fusion.memory_norm.weight.data = sd[p + 'memory_norm.weight'].to(device, torch.bfloat16)
    fusion.memory_norm.bias.data = sd[p + 'memory_norm.bias'].to(device, torch.bfloat16)
    fusion.query_projection.weight.data = sd[p + 'query_projection.weight'].to(device, torch.bfloat16)
    fusion.query_projection.bias.data = sd[p + 'query_projection.bias'].to(device, torch.bfloat16)
    fusion.cross_attention.in_proj_weight.data = sd[p + 'cross_attention.in_proj_weight'].to(device, torch.bfloat16)
    fusion.cross_attention.in_proj_bias.data = sd[p + 'cross_attention.in_proj_bias'].to(device, torch.bfloat16)
    fusion.cross_attention.out_proj.weight.data = sd[p + 'cross_attention.out_proj.weight'].to(device, torch.bfloat16)
    fusion.cross_attention.out_proj.bias.data = sd[p + 'cross_attention.out_proj.bias'].to(device, torch.bfloat16)
    fusion.output_projection[0].weight.data = sd[p + 'output_projection.0.weight'].to(device, torch.bfloat16)
    fusion.output_projection[0].bias.data = sd[p + 'output_projection.0.bias'].to(device, torch.bfloat16)
    fusion.output_projection[1].weight.data = sd[p + 'output_projection.1.weight'].to(device, torch.bfloat16)
    fusion.output_projection[1].bias.data = sd[p + 'output_projection.1.bias'].to(device, torch.bfloat16)
    fusion.residual_scale.data = sd[p + 'residual_scale'].to(device, torch.bfloat16)
    fusion.eval()
    return fusion


def load_all(ckpt, qwen_path, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    cfg = GeoMemoryConfig.from_json(os.path.join(ckpt, "geomemory_config.json"))
    lm = AutoModelForImageTextToText.from_pretrained(
        qwen_path, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    lm.requires_grad_(False)
    model = GeoMemoryTrainingModel(cfg, language_model=lm, freeze_language_model=True)
    _load_geomemory_checkpoint(model, ckpt)
    model.geomemory.to(device, torch.bfloat16)
    model.visual_to_llm.to(device, torch.bfloat16)
    model.eval()
    original_fusion = load_original_fusion(cfg, ckpt, device)
    model.geomemory.visual_fusion = original_fusion
    tok = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)
    ev = cfg.numeric_evidence_slots if (cfg.use_numeric_readout and cfg.numeric_inject_evidence) else 0
    model.im_end_id = tok.convert_tokens_to_ids('<|im_end|>')
    builder = QwenPromptBuilder(tok, ev)
    return model, processor, builder, cfg


@torch.no_grad()
def vit_features(lm, processor, frames, device):
    p = processor.image_processor(images=frames, return_tensors="pt")
    pv, grid = p["pixel_values"].to(device, torch.bfloat16), p["image_grid_thw"].to(device)
    out = lm.model.get_image_features(pv, grid, return_dict=True)
    pooled = out.pooler_output
    if not isinstance(pooled, (list, tuple)):
        counts = (grid.prod(-1) // 4).tolist()
        pooled = torch.split(pooled, counts)
    return list(pooled), grid.cpu()


@torch.no_grad()
def compute_memory_history(model, feats, grid, device):
    T = len(feats)
    pf = torch.stack(feats)[None].to(torch.bfloat16)
    N = feats[0].shape[0]
    xy = torch.stack([grid_to_patch_xy(grid[t], N, 2) for t in range(T)])[None].to(device, torch.bfloat16)
    core = model.geomemory(pf, patch_xy=xy)
    return core.memory_history, core.patch_geometry, pf, xy


@torch.no_grad()
def build_prepared(model, builder, feats, grid, question, mem_history, pf, xy, device):
    cfg = model.cfg
    T = pf.shape[1]
    N = pf.shape[2]
    idx = select_qwen_frame_indices(T, cfg.qwen_frame_indices, cfg.qwen_full_length)
    idx_t = torch.tensor(idx, device=device)
    sel_pf = pf[:, idx_t]
    sel_mem = mem_history[:, idx_t]
    fused = model.geomemory.visual_fusion(
        sel_pf,
        GeoMemoryState(memory=sel_mem, pose=None, step=torch.tensor([T - 1], device=device)),
        None, geometry=None)
    K = len(idx)
    frames_llm = model.visual_to_llm(fused)[0].reshape(K * N, -1)
    sel_tokens = [N] * K
    ids, _ = builder.build(question, None, sel_tokens)
    ids_t = torch.tensor([ids], device=device)
    mm = (ids_t == builder.image_pad_id)
    lm = model.language_model
    emb = lm.get_input_embeddings()(ids_t)
    if int(mm.sum()) != frames_llm.shape[0]:
        raise ValueError("placeholder/feature count mismatch")
    emb = emb.masked_scatter(mm.unsqueeze(-1), frames_llm.to(emb.dtype))
    grid_sel = grid[idx_t.cpu()].to(device).reshape(-1, 3)
    pos, _ = model._qwen_decoder().get_rope_index(
        ids_t, mm_token_type_ids=mm.to(torch.int32), image_grid_thw=grid_sel,
        attention_mask=torch.ones_like(ids_t))
    return dict(ids=ids_t, mm=mm, emb=emb, pos=pos)


@torch.no_grad()
def compute_nll(model, builder, P, answer):
    lm = model.language_model
    dec = model._qwen_decoder()
    ans_ids = builder.answer_ids(answer)
    ans_t = torch.tensor([ans_ids], device=P["emb"].device)
    full_emb = torch.cat([P["emb"], lm.get_input_embeddings()(ans_t)], dim=1)
    full_pos = torch.cat([P["pos"], P["pos"][..., -1:].expand(3, 1, len(ans_ids)) +
                          torch.arange(1, len(ans_ids) + 1, device=P["pos"].device).view(1, 1, -1)], dim=-1)
    full_att = torch.ones(full_emb.shape[:2], device=P["emb"].device, dtype=torch.long)
    out = dec(inputs_embeds=full_emb, attention_mask=full_att, position_ids=full_pos,
              use_cache=False, return_dict=True)
    hidden = out.last_hidden_state
    plen = P["emb"].shape[1]
    shift_hidden = hidden[:, plen - 1:-1]
    logits = lm.lm_head(shift_hidden[0]).float()
    tok_loss = nn.functional.cross_entropy(logits, ans_t[0], reduction="none")
    return float(tok_loss.mean().item()), len(ans_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--qwen_path", default=DEFAULT_QWEN)
    ap.add_argument("--vsi_jsonl", default=DEFAULT_VSI_JSONL)
    ap.add_argument("--vsi_root", default=DEFAULT_VSI_ROOT)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_num_frames", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--num_donors", type=int, default=5)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, processor, builder, cfg = load_all(args.ckpt, args.qwen_path, device)
    initial_mem = model.geomemory.memory.initial_memory.detach().clone()

    ds = VSIBenchDataset(args.vsi_jsonl, args.vsi_root, args.max_num_frames,
                         args.max_samples, args.rank, args.world)
    # Build id -> dataset mapping from the JSONL
    id_to_dataset = {}
    for line in open(args.vsi_jsonl):
        r = json.loads(line)
        id_to_dataset[r["id"]] = r.get("dataset", "")
    print(f"[rank={args.rank}] samples={len(ds)} num_donors={args.num_donors}", flush=True)

    # Phase 1: compute memory for all samples
    print("Phase 1: computing memory for all samples ...", flush=True)
    mem_cache = {}  # id -> {mem_history, dataset, scene_name, question, answer, question_type, feats, grid, pf, xy}
    for i in tqdm(range(len(ds)), desc="mem"):
        item = None
        for attempt in range(8):
            try:
                item = ds[i]
                if len(item["frames"]) > 0:
                    break
            except ValueError:
                pass
            time.sleep(3 * (attempt + 1))
        if item is None or len(item["frames"]) == 0:
            continue
        try:
            feats, grid = vit_features(model.language_model, processor, item["frames"], device)
            mem_history, _, pf, xy = compute_memory_history(model, feats, grid, device)
            mem_cache[item["id"]] = {
                "mem_history": mem_history.cpu(),
                "dataset": id_to_dataset.get(item["id"], ""),
                "scene_name": item.get("scene_name", ""),
                "question": item["question"],
                "answer": item["answer"],
                "question_type": item["question_type"],
                "feats": feats,
                "grid": grid,
                "pf": pf,
                "xy": xy,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    print(f"  cached {len(mem_cache)} samples", flush=True)

    # Group by dataset for cross-dataset donors
    by_dataset = {}
    for sid, mc in mem_cache.items():
        by_dataset.setdefault(mc["dataset"], []).append(sid)
    print(f"  datasets: {[(d, len(v)) for d, v in by_dataset.items()]}", flush=True)

    # Phase 2: for each sample, compute NLL_real + NLL for multiple donors
    print("Phase 2: computing NLL for real + multiple donor conditions ...", flush=True)
    all_ids = list(mem_cache.keys())
    results = []

    for si, sid in enumerate(tqdm(all_ids, desc="nll")):
        mc = mem_cache[sid]
        T = mc["mem_history"].shape[1]
        real_mem = mc["mem_history"].to(device)

        # Compute NLL_real
        try:
            P = build_prepared(model, builder, mc["feats"], mc["grid"], mc["question"],
                               real_mem, mc["pf"], mc["xy"], device)
            nll_real, _ = compute_nll(model, builder, P, mc["answer"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        # Select donors
        # 1. K random same-dataset donors (different scene)
        same_ds = [oid for oid in by_dataset.get(mc["dataset"], []) if oid != sid
                   and mem_cache[oid]["mem_history"].shape[1] == T]
        random.shuffle(same_ds)
        same_ds_donors = same_ds[:args.num_donors]

        # 2. K random cross-dataset donors
        other_ds = []
        for d, ids in by_dataset.items():
            if d != mc["dataset"]:
                other_ds.extend([oid for oid in ids if mem_cache[oid]["mem_history"].shape[1] == T])
        random.shuffle(other_ds)
        cross_ds_donors = other_ds[:args.num_donors]

        # 3. Gaussian noise memory
        noise_mem = torch.randn_like(real_mem) * real_mem.std()

        # 4. Zero memory
        zero_mem = torch.zeros_like(real_mem)

        # 5. Initial memory
        init_mem = initial_mem.unsqueeze(0).expand(1, T, -1, -1).contiguous()

        donor_list = []
        for did in same_ds_donors:
            donor_list.append(("same_ds", mem_cache[did]["scene_name"], mem_cache[did]["mem_history"].to(device)))
        for did in cross_ds_donors:
            donor_list.append(("cross_ds", mem_cache[did]["scene_name"], mem_cache[did]["mem_history"].to(device)))
        donor_list.append(("noise", "gaussian", noise_mem))
        donor_list.append(("zero", "zeros", zero_mem))
        donor_list.append(("initial", "init", init_mem))

        donor_nlls = []
        for dtype, dscene, dmem in donor_list:
            try:
                P = build_prepared(model, builder, mc["feats"], mc["grid"], mc["question"],
                                   dmem, mc["pf"], mc["xy"], device)
                nll, _ = compute_nll(model, builder, P, mc["answer"])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                nll = float("nan")
            donor_nlls.append({
                "donor_type": dtype,
                "donor_scene": dscene,
                "nll": nll,
                "delta_state": nll - nll_real if not math.isnan(nll) else None,
            })

        results.append({
            "id": sid,
            "dataset": mc["dataset"],
            "scene_name": mc["scene_name"],
            "question_type": mc["question_type"],
            "nll_real": nll_real,
            "donors": donor_nlls,
        })

        if (si + 1) % 50 == 0 or si + 1 == len(all_ids):
            json.dump(results, open(os.path.join(args.out_dir, f"multidonor_rank{args.rank}.json"), "w"),
                      ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Delta_State distribution across donor types")
    print("=" * 80)
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in results:
        for d in r["donors"]:
            if d["delta_state"] is not None and not math.isnan(d["delta_state"]):
                by_type[d["donor_type"]].append(d["delta_state"])

    print(f"\n{'donor_type':<15} {'mean':>10} {'std':>10} {'median':>10} {'min':>10} {'max':>10} {'n':>6}")
    print("-" * 75)
    for dtype in ["same_ds", "cross_ds", "noise", "zero", "initial"]:
        vals = by_type[dtype]
        if vals:
            arr = np.array(vals)
            print(f"{dtype:<15} {arr.mean():>10.6f} {arr.std():>10.6f} {np.median(arr):>10.6f} "
                  f"{arr.min():>10.6f} {arr.max():>10.6f} {len(arr):>6}")

    # Per question type for same_ds
    print(f"\nPer question type (same_ds donors):")
    print(f"{'question_type':<35} {'mean':>10} {'std':>10} {'n':>6}")
    print("-" * 65)
    by_qt = defaultdict(list)
    for r in results:
        for d in r["donors"]:
            if d["donor_type"] == "same_ds" and d["delta_state"] is not None:
                by_qt[r["question_type"]].append(d["delta_state"])
    for qt in sorted(by_qt):
        arr = np.array(by_qt[qt])
        print(f"{qt:<35} {arr.mean():>10.6f} {arr.std():>10.6f} {len(arr):>6}")

    summary = {
        "n_samples": len(results),
        "num_donors": args.num_donors,
        "by_donor_type": {dt: {"mean": float(np.mean(v)) if v else None,
                                "std": float(np.std(v)) if v else None,
                                "median": float(np.median(v)) if v else None,
                                "min": float(np.min(v)) if v else None,
                                "max": float(np.max(v)) if v else None,
                                "n": len(v)}
                          for dt, v in by_type.items()},
        "per_question_type_same_ds": {qt: {"mean": float(np.mean(v)) if v else None,
                                            "std": float(np.std(v)) if v else None,
                                            "n": len(v)}
                                        for qt, v in by_qt.items()},
    }
    json.dump(summary, open(os.path.join(args.out_dir, f"multidonor_summary_rank{args.rank}.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.out_dir}/multidonor_summary_rank{args.rank}.json")


if __name__ == "__main__":
    main()
