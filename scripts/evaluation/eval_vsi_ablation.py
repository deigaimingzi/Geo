#!/usr/bin/env python3
"""Stage B memory ablation eval on VSI-Bench.

Reuses the exact same model loading + visual-fusion pipeline as
eval_vsi_geomemory.py, but evaluates five memory conditions on the same
5,130 VSI QAs and reports *both* greedy accuracy/MRA *and* answer NLL
(negative log-likelihood, token-mean, answer-only) so that
    Delta_State  = NLL_Wrong  - NLL_Real
    Delta_History= NLL_Short  - NLL_Real
can be computed per question type.

Conditions
----------
real        : use the scene-correct M_t prefix (normal behaviour)
wrong       : replace M_t with another scene's M_t at the SAME observed
              prefix length (visual tokens unchanged)
short       : use only the first frame to build memory (M_1 broadcast)
initial     : use the model's learned initial_memory (no update at all)
fusion_off  : bypass the visual-fusion residual entirely (F'_k = F_k)

For the `wrong` condition a donor scene is selected whose number of
observed frames matches the target's, so the swapped memory lives in the
same temporal regime.  Only the memory tensor is swapped; the visual tokens
fed to Qwen stay identical to the `real` condition, isolating the effect
of memory content.

Usage (one rank per GPU):
    CUDA_VISIBLE_DEVICES=0 python eval_vsi_ablation.py \
        --ckpt output/v3/stage_b --out_dir output/v3/stage_b/ablation \
        --conditions real wrong short initial fusion_off \
        --rank 0 --world 8
"""
import argparse, json, os, sys, random, time
import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/scripts/evaluation")
from eval_vsi_mra import (VSIBenchDataset, score_record, aggregate_results,  # noqa: E402
                          NA_QUESTION_TYPES, MCA_QUESTION_TYPES)

from geomemory.config import GeoMemoryConfig  # noqa: E402
from geomemory.training_model import GeoMemoryTrainingModel  # noqa: E402
from geomemory.train import _load_geomemory_checkpoint  # noqa: E402
from geomemory.position import grid_to_patch_xy  # noqa: E402
from geomemory.qwen_protocol import QwenPromptBuilder  # noqa: E402
from geomemory.state import GeoMemoryState  # noqa: E402
from geomemory.fusion import select_qwen_frame_indices  # noqa: E402

DEFAULT_QWEN = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B"
DEFAULT_VSI_JSONL = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/benchmark/VSI-Bench/raw/test.jsonl"
DEFAULT_VSI_ROOT = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/benchmark/VSI-Bench/videos"

ALL_CONDITIONS = ["real", "wrong", "short", "initial", "fusion_off"]


# ---------------------------------------------------------------------------
# Model loading (identical to eval_vsi_geomemory.load_all)
# ---------------------------------------------------------------------------
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
    tok = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)
    ev = cfg.numeric_evidence_slots if (cfg.use_numeric_readout and cfg.numeric_inject_evidence) else 0
    model.evidence_id = tok.convert_tokens_to_ids('<|vision_pad|>')
    if getattr(model, "numeric", None) is not None:
        model.numeric.to(device, torch.bfloat16)
    return model, processor, QwenPromptBuilder(tok, ev)


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
    """Run the recurrent memory over all T frames; return (mem_history, patch_geometry, pf, xy).

    mem_history: [1, T, K, Dm]  memory state after each frame prefix.
    """
    n_tok = [f.shape[0] for f in feats]
    T = len(feats)
    pf = torch.stack(feats)[None].to(torch.bfloat16)
    N = n_tok[0]
    xy = torch.stack([grid_to_patch_xy(grid[t], N, 2) for t in range(T)])[None].to(device, torch.bfloat16)
    core = model.geomemory(pf, patch_xy=xy)
    mem = core.memory_history
    assert mem.shape[1] == T, f"memory updated {mem.shape[1]} times for {T} frames"
    return mem, core.patch_geometry, pf, xy


@torch.no_grad()
def build_prepared(model, builder, feats, grid, question, mem_history, patch_geometry,
                   pf, xy, device, condition, donor_mem_history=None, fusion_off=False):
    """Build the prepared inputs (ids, mm, emb, pos, grid) for one condition.

    mem_history : [1, T, K, Dm]  the memory to use (already substituted for
                  wrong/short/initial conditions).
    fusion_off  : if True, bypass the visual-fusion residual (F'_k = F_k).
    """
    cfg = model.cfg
    T = pf.shape[1]
    N = pf.shape[2]
    idx = select_qwen_frame_indices(T, cfg.qwen_frame_indices, cfg.qwen_full_length)
    idx_t = torch.tensor(idx, device=device)
    sel_pf = pf[:, idx_t]
    sel_mem = mem_history[:, idx_t]
    sel_geometry = None if patch_geometry is None else {
        key: value[:, idx_t] for key, value in patch_geometry.items()
    }

    if fusion_off:
        # Bypass the fusion residual: use raw visual tokens.
        fused = sel_pf
    else:
        fused = model.geomemory.visual_fusion(
            sel_pf,
            GeoMemoryState(memory=sel_mem, pose=None, step=torch.tensor([T - 1], device=device)), None,
            geometry=sel_geometry)

    K = len(idx)
    frames_llm = model.visual_to_llm(fused)[0].reshape(K * N, -1)
    sel_tokens = [pf.shape[2]] * K
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
    return dict(ids=ids_t, mm=mm, emb=emb, pos=pos, grid=grid_sel, idx=idx)


@torch.no_grad()
def greedy_decode(model, builder, P, max_new_tokens):
    lm = model.language_model
    dec = model._qwen_decoder()
    embed = lm.get_input_embeddings()
    out = dec(inputs_embeds=P["emb"], position_ids=P["pos"], use_cache=True, return_dict=True)
    cache, next_pos = out.past_key_values, P["pos"].max() + 1
    logits = lm.lm_head(out.last_hidden_state[:, -1])
    toks = []
    for _ in range(max_new_tokens):
        nxt = int(logits.argmax(-1).item())
        if nxt == builder.im_end_id or nxt == builder.tokenizer.eos_token_id:
            break
        toks.append(nxt)
        out = dec(inputs_embeds=embed(torch.tensor([[nxt]], device=P["emb"].device)),
                  position_ids=next_pos.view(1, 1, 1).expand(3, 1, 1),
                  past_key_values=cache, use_cache=True, return_dict=True)
        cache, next_pos = out.past_key_values, next_pos + 1
        logits = lm.lm_head(out.last_hidden_state[:, -1])
    return toks


@torch.no_grad()
def compute_nll(model, builder, P, answer):
    """Answer-only token-mean NLL (same protocol as training _answer_loss).

    Returns (nll, n_answer_tokens).  nll is the mean cross-entropy over the
    answer tokens (including the trailing <|im_end|>).
    """
    lm = model.language_model
    dec = model._qwen_decoder()
    ans_ids = builder.answer_ids(answer)
    ans_t = torch.tensor([ans_ids], device=P["emb"].device)
    full_emb = torch.cat([P["emb"], lm.get_input_embeddings()(ans_t)], dim=1)
    full_pos = torch.cat([P["pos"], P["pos"][..., -1:].expand(3, 1, len(ans_ids)) + torch.arange(1, len(ans_ids) + 1, device=P["pos"].device).view(1, 1, -1)], dim=-1)
    full_att = torch.ones(full_emb.shape[:2], device=P["emb"].device, dtype=torch.long)
    out = dec(inputs_embeds=full_emb, attention_mask=full_att, position_ids=full_pos,
              use_cache=False, return_dict=True)
    hidden = out.last_hidden_state
    plen = P["emb"].shape[1]
    shift_hidden = hidden[:, plen - 1:-1]
    shift_labels = ans_t
    logits = lm.lm_head(shift_hidden[0]).float()
    tok_loss = nn.functional.cross_entropy(logits, shift_labels[0], reduction="none")
    nll = float(tok_loss.mean().item())
    return nll, len(ans_ids)


def get_condition_memory(condition, real_mem, donor_mem, initial_mem, T):
    """Return the memory_history tensor [1,T,K,Dm] for the given condition."""
    if condition == "real":
        return real_mem
    if condition == "wrong":
        if donor_mem is None:
            return real_mem
        return donor_mem
    if condition == "short":
        short_mem = real_mem[:, :1].expand(1, T, -1, -1).contiguous()
        return short_mem
    if condition == "initial":
        init_mem = initial_mem.unsqueeze(0).expand(1, T, -1, -1).contiguous()
        return init_mem
    if condition == "fusion_off":
        return real_mem
    raise ValueError(f"unknown condition {condition}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--qwen_path", default=DEFAULT_QWEN)
    ap.add_argument("--vsi_jsonl", default=DEFAULT_VSI_JSONL)
    ap.add_argument("--vsi_root", default=DEFAULT_VSI_ROOT)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_num_frames", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                    choices=ALL_CONDITIONS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")
    random.seed(args.seed)

    model, processor, builder = load_all(args.ckpt, args.qwen_path, device)

    # Pre-compute the learned initial_memory for the `initial` condition.
    initial_mem = model.geomemory.memory.initial_memory.detach().clone()

    ds = VSIBenchDataset(args.vsi_jsonl, args.vsi_root, args.max_num_frames,
                         args.max_samples, args.rank, args.world)
    print(f"[rank={args.rank}] samples={len(ds)} frames={args.max_num_frames} "
          f"conditions={args.conditions}", flush=True)

    # ---- Pre-scan: compute (feats, grid, mem_history) for every sample once ----
    # We cache per-sample memory so the `wrong` condition can pick a donor with
    # the same observed-frame count without recomputing.
    sample_cache = {}  # id -> dict(feats, grid, mem_history, patch_geometry, pf, xy)
    cache_path = os.path.join(args.out_dir, f"memcache_rank{args.rank}.pt")
    if os.path.exists(cache_path):
        try:
            sample_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"[rank={args.rank}] loaded memcache with {len(sample_cache)} entries", flush=True)
        except Exception:
            sample_cache = {}

    # ---- Main eval loop ----
    out_path = os.path.join(args.out_dir, f"ablation_rank{args.rank}.json")
    records = []
    if os.path.exists(out_path):
        records = json.load(open(out_path))
    done = {r["id"] for r in records}
    print(f"[rank={args.rank}] resume: {len(done)} done", flush=True)

    all_ids = [ds.rows[i]["id"] for i in range(len(ds))]
    for i in tqdm(range(len(ds)), desc=f"rank{args.rank}", dynamic_ncols=True):
        if ds.rows[i]["id"] in done:
            continue
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
            print(f"[rank={args.rank}] SKIP id={ds.rows[i]['id']} (frames unreadable)", flush=True)
            continue

        # Compute ViT features + real memory history once.
        try:
            feats, grid = vit_features(model.language_model, processor, item["frames"], device)
            mem_history, patch_geometry, pf, xy = compute_memory_history(model, feats, grid, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[rank={args.rank}] OOM computing memory for id={item['id']}, skip", flush=True)
            continue

        sid = item["id"]
        T = mem_history.shape[1]
        sample_cache[sid] = {
            "mem_history": mem_history.cpu(),
            "patch_geometry": None if patch_geometry is None else
                {k: v.cpu() for k, v in patch_geometry.items()},
        }

        # Pick a donor for the `wrong` condition: same T, different scene.
        donor_mem = None
        if "wrong" in args.conditions:
            candidates = [oid for oid in sample_cache if oid != sid
                          and sample_cache[oid]["mem_history"].shape[1] == T]
            if candidates:
                donor_id = random.choice(candidates)
                donor_mem = sample_cache[donor_id]["mem_history"].to(device)

        per_condition = {}
        for cond in args.conditions:
            cond_mem = get_condition_memory(cond, mem_history, donor_mem, initial_mem, T)
            cond_mem = cond_mem.to(device)
            fusion_off = (cond == "fusion_off")
            try:
                P = build_prepared(model, builder, feats, grid, item["question"],
                                   cond_mem, patch_geometry, pf, xy, device, cond,
                                   fusion_off=fusion_off)
                toks = greedy_decode(model, builder, P, args.max_new_tokens)
                pred = builder.tokenizer.decode(toks, skip_special_tokens=True).strip()
                nll, n_tok = compute_nll(model, builder, P, item["answer"])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                pred, nll, n_tok = "", float("nan"), 0
            per_condition[cond] = {
                "prediction": pred,
                "nll": nll,
                "n_answer_tokens": n_tok,
                "score": score_record(item["question_type"], pred, item["answer"]),
            }

        records.append({
            "id": sid,
            "scene_name": item["scene_name"],
            "question": item["question"],
            "answer": item["answer"],
            "question_type": item["question_type"],
            "conditions": per_condition,
        })
        if len(records) % args.save_every == 0 or i + 1 == len(ds):
            json.dump(records, open(out_path, "w"), ensure_ascii=False, indent=2)
            torch.save(sample_cache, cache_path)

    # Save final cache.
    torch.save(sample_cache, cache_path)
    print(f"[rank={args.rank}] done, {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
