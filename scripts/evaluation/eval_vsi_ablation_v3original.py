#!/usr/bin/env python3
"""V3 Stage B memory ablation using the RECONSTRUCTED original fusion module.

The current code's 4-layer output_projection doesn't match the checkpoint's 2-layer
structure, causing all fusion weights to be dropped on load and making fusion a no-op.

This script rebuilds the original 2-layer fusion, loads the checkpoint weights
manually, then runs the 5-condition ablation (real/wrong/short/initial/fusion_off)
on VSI-Bench with NLL computation.

Usage (one rank per GPU):
    CUDA_VISIBLE_DEVICES=0 python eval_vsi_ablation_v3original.py \
        --ckpt output/v3/stage_b --out_dir output/v3/stage_b/ablation-original \
        --rank 0 --world 8
"""
import argparse, json, os, sys, random, time, math
import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/scripts/evaluation")
from eval_vsi_mra import (VSIBenchDataset, score_record, aggregate_results,
                          NA_QUESTION_TYPES, MCA_QUESTION_TYPES)

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

ALL_CONDITIONS = ["real", "wrong", "short", "initial", "fusion_off"]


class OriginalFusion(nn.Module):
    """Reconstructed original v3 fusion: 2-layer output_projection, no geometry gate."""

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
    """Build original fusion and load weights from checkpoint."""
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
    # Replace the broken fusion with the reconstructed original
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
    n_tok = [f.shape[0] for f in feats]
    T = len(feats)
    pf = torch.stack(feats)[None].to(torch.bfloat16)
    N = n_tok[0]
    xy = torch.stack([grid_to_patch_xy(grid[t], N, 2) for t in range(T)])[None].to(device, torch.bfloat16)
    core = model.geomemory(pf, patch_xy=xy)
    mem = core.memory_history
    return mem, core.patch_geometry, pf, xy


@torch.no_grad()
def build_prepared(model, builder, feats, grid, question, mem_history, patch_geometry,
                   pf, xy, device, condition, donor_mem_history=None, fusion_off=False):
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
        fused = sel_pf
    else:
        fused = model.geomemory.visual_fusion(
            sel_pf,
            GeoMemoryState(memory=sel_mem, pose=None, step=torch.tensor([T - 1], device=device)),
            None, geometry=sel_geometry)

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
    nll = float(tok_loss.mean().item())
    return nll, len(ans_ids)


def get_condition_memory(condition, real_mem, donor_mem, initial_mem, T):
    if condition == "real":
        return real_mem
    if condition == "wrong":
        return donor_mem if donor_mem is not None else real_mem
    if condition == "short":
        return real_mem[:, :1].expand(1, T, -1, -1).contiguous()
    if condition == "initial":
        return initial_mem.unsqueeze(0).expand(1, T, -1, -1).contiguous()
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
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS, choices=ALL_CONDITIONS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")
    random.seed(args.seed)

    model, processor, builder, cfg = load_all(args.ckpt, args.qwen_path, device)
    initial_mem = model.geomemory.memory.initial_memory.detach().clone()

    ds = VSIBenchDataset(args.vsi_jsonl, args.vsi_root, args.max_num_frames,
                         args.max_samples, args.rank, args.world)
    print(f"[rank={args.rank}] samples={len(ds)} frames={args.max_num_frames} "
          f"conditions={args.conditions}", flush=True)

    sample_cache = {}
    cache_path = os.path.join(args.out_dir, f"memcache_rank{args.rank}.pt")
    if os.path.exists(cache_path):
        try:
            sample_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"[rank={args.rank}] loaded memcache with {len(sample_cache)} entries", flush=True)
        except Exception:
            sample_cache = {}

    out_path = os.path.join(args.out_dir, f"ablation_rank{args.rank}.json")
    records = []
    if os.path.exists(out_path):
        records = json.load(open(out_path))
    done = {r["id"] for r in records}
    print(f"[rank={args.rank}] resume: {len(done)} done", flush=True)

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

    torch.save(sample_cache, cache_path)
    print(f"[rank={args.rank}] done, {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
