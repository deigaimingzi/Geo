#!/usr/bin/env python3
"""Causal OST evaluation: v8 preprocessing, native training prompts, incremental memory.

Scoring is the local exact-match/MRA implementation, not an official downloaded scorer.
Task types, rather than the spelling of a number, distinguish estimation from counting.
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "evaluation"))
from pathlib import Path
from geomemory.ost import (PROTOCOL, DIALOGUE_PROTOCOL, atomic_json, frame_metadata, image_labels, load_image,
                          qa_record, read_sessions, safe_image_path, supervised)  # noqa: E402
from geomemory.ost_metrics import hierarchical_scores, official_score

QWEN = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B"
BENCH = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/benchmark/OST-Bench"
THRESH = np.arange(0.5, 0.96, 0.05)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frames(paths, root):
    return [load_image(safe_image_path(root, p)) for p in paths]


def rec_lookup_pred(recs, sid, turn_id):
    """Return a previously saved prediction for (sid, turn_id), or empty string."""
    for r in recs:
        if r["scan_id"] == sid and int(r["turn_id"]) == int(turn_id):
            return r.get("prediction", "")
    return ""


def _norm(s):
    return re.sub(r"[\s\"'.,;:!]+", " ", str(s).lower()).strip()


def _strip_wrap(x):
    x = x.strip().strip("[]").strip()
    x = re.sub(r"['\"]\s*,\s*['\"]", ", ", x)  # list form ['a', 'b'] -> a, b
    return x.strip().strip("'\"").strip()


def extract_answer(text):
    """Return the answer span of a reply.  Handles JSON / pseudo-JSON ({answer: ..}, "answer": ".."),
    list form ['a', 'b'] / ['a'], and bare text.  Multi-object options contain commas, so values are
    taken up to the closing quote / bracket rather than the first comma."""
    t = text.strip()
    try:
        value = json.loads(t)
        if isinstance(value, dict) and "answer" in value:
            answer = value["answer"]
            return ", ".join(map(str, answer)) if isinstance(answer, list) else str(answer)
    except (ValueError, TypeError):
        pass
    m = re.search(r"[\"']?answer[\"']?\s*[:=_]*\s*(?=[\"'\[\w\-])", t, re.I)
    if m:
        rest = t[m.end():]
        if rest[:1] in "\"'":
            q = rest[0]
            end = rest.find(q + ",", 1)
            end2 = rest.find(q + "\n", 1)
            end3 = rest.find(q + "}", 1)
            end4 = rest.find(q + " ", 1) if False else -1
            cands = [e for e in (end, end2, end3) if e > 0]
            end = min(cands) if cands else rest.rfind(q)
            # a trailing  ", 'reason'  must not swallow the option list: cut before ,"reason"
            return _strip_wrap(rest[1:end] if end > 0 else rest[1:])
        if rest[:1] == "[":
            end = rest.find("]")
            return _strip_wrap(rest[1:end] if end > 0 else rest[1:])
        return re.split(r"[,}\n]", rest, 1)[0].strip()
    m = re.match(r"\s*\[(.*?)\]", t, re.S)
    if m:
        return _strip_wrap(m.group(1))
    return t


def mra(pred, gt):
    if gt == 0:
        return float(pred == 0)
    rel = abs(pred - gt) / abs(gt)
    return float(np.mean([rel < 1 - t for t in THRESH]))


def _is_num(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def score(rec):
    """None -> excluded (conversation-only turn, no gold answer). Options -> exact match.
    No options + numeric gold: Estimation -> MRA, Counting/Temporal-loc -> exact match."""
    gt, opts = rec["answer"], rec["option"]
    if rec["type"] == "None" or (gt == "" and not opts):
        return None
    pred = extract_answer(rec["prediction"])
    if opts:
        return float(_norm(pred) == _norm(gt))
    if not _is_num(gt):
        return float(_norm(pred) == _norm(gt))
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", pred)
    if not nums:
        return 0.0
    v, g = float(nums[0]), float(gt)
    if not np.isfinite(v) or not np.isfinite(g):
        return 0.0
    return mra(v, g) if "(Estimation)" in rec["type"] else float(v == g)


def aggregate(recs):
    out, by = {}, {}
    total = len(recs)
    recs = [r for r in recs if r["score"] is not None]
    for r in recs:
        by.setdefault(r["type"].split("(")[0], []).append(r["score"])
    for k, v in sorted(by.items()):
        out[k] = float(np.mean(v)) * 100
    out["overall"] = float(np.mean([r["score"] for r in recs])) * 100 if recs else 0.0
    out["num_records"] = len(recs)
    out["num_total_turns"] = total
    out["num_excluded"] = total - len(recs)
    out["macro_category"] = float(np.mean([np.mean(v) for v in by.values()])) * 100 if by else 0.0
    out["category_counts"] = {k: len(v) for k, v in sorted(by.items())}
    types, turns = {}, {}
    for rec in recs:
        types.setdefault(rec["type"], []).append(rec["score"])
        turns.setdefault(str(rec["turn_id"]), []).append(rec["score"])
    out["by_type"] = {k: {"score": float(np.mean(v)) * 100, "n": len(v)} for k, v in sorted(types.items())}
    out["by_turn"] = {k: {"score": float(np.mean(v)) * 100, "n": len(v)} for k, v in sorted(turns.items())}
    return out


def aggregate_with_scorer(recs, scorer):
    metrics = aggregate(recs)
    metrics["overall_micro"] = metrics["overall"]
    if scorer == "official":
        try:
            metrics["official_dimensions"] = hierarchical_scores(metrics["by_type"])
            metrics["overall"] = metrics["official_dimensions"]["Overall"]
            metrics["official_macro_complete"] = True
        except KeyError as missing:
            metrics["overall"] = None
            metrics["official_macro_complete"] = False
            metrics["missing_macro_type"] = str(missing)
    return metrics


@torch.no_grad()
def vit(lm, processor, frames, device):
    p = processor.image_processor(images=frames, return_tensors="pt", do_resize=False)
    pv, grid = p["pixel_values"].to(device, torch.bfloat16), p["image_grid_thw"].to(device)
    base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
    out = base.model.get_image_features(pv, grid, return_dict=True)
    pooled = out.pooler_output
    if not isinstance(pooled, (list, tuple)):
        pooled = torch.split(pooled, (grid.prod(-1) // 4).tolist())
    return list(pooled), grid.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--qwen_path", default=QWEN)
    ap.add_argument("--bench", default=f"{BENCH}/OST_bench_eval_nested.json")
    ap.add_argument("--image_root", default=f"{BENCH}/image_upload")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--feature_root", help="Optional prepare_ost_stage_c.py eval_features directory")
    ap.add_argument("--limit", type=int, default=0, help="max sessions (smoke)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--multi_round", action="store_true",
                    help="Official Multi-round: include previous Q&A as dialogue history")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--scorer", choices=("local", "official"), default="local")
    a = ap.parse_args()
    if not 0 <= a.rank < a.world:
        ap.error("rank must be within world size")
    os.makedirs(a.out_dir, exist_ok=True)
    data = read_sessions(a.bench)
    if a.limit:
        data = data[:a.limit]
    expected = {(s["scan_id"], int(t["turn_id"])) for s in data for t in s["user_message"]}
    scorer = official_score if a.scorer == "official" else score
    protocol = {"protocol": DIALOGUE_PROTOCOL, "checkpoint": str(Path(a.ckpt).resolve()),
                "checkpoint_sha256": file_sha256(Path(a.ckpt) / "model.safetensors"),
                "benchmark_sha256": file_sha256(a.bench),
                "bench": str(Path(a.bench).resolve()), "limit": a.limit, "world": a.world,
                "qwen_path": str(Path(a.qwen_path).resolve()), "long_edge": 512,
                 "max_new_tokens": a.max_new_tokens,
                  "dialogue_history": "all turns: images+questions+model responses",
                  "scoring": ("official option-LCS/enumeration/MRA + hierarchical macro" if a.scorer == "official"
                              else "local type-aware exact-match/MRA(0.50:0.05:0.95)")}
    adapter_path = Path(a.ckpt) / "adapter_model.safetensors"
    if adapter_path.is_file():
        protocol["lora_sha256"] = file_sha256(adapter_path)
    protocol_path = Path(a.out_dir) / "eval_protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if previous != protocol:
            raise ValueError("Output directory belongs to a different evaluation protocol")
    elif a.rank == 0:
        atomic_json(protocol_path, protocol)
    if a.merge:
        recs = []
        for rank in range(a.world):
            with open(os.path.join(a.out_dir, f"predictions_rank{rank}.json")) as handle:
                recs += json.load(handle)
        keys = [(r["scan_id"], int(r["turn_id"])) for r in recs]
        if len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError(f"Incomplete/duplicate evaluation: records={len(keys)}, expected={len(expected)}")
        for rec in recs:
            rec["score"] = scorer(rec)
        m = aggregate_with_scorer(recs, a.scorer)
        if a.scorer == "official" and not a.limit and not m["official_macro_complete"]:
            raise ValueError("Full official evaluation is missing one or more task types")
        m["coverage_complete"] = True
        atomic_json(Path(a.out_dir) / "metrics.json", m)
        print(json.dumps(m, indent=2))
        return
    from geomemory.ost_inference import OSTStream, greedy_decode, load_model
    device = torch.device("cuda:0")
    model, processor, builder = load_model(a.ckpt, a.qwen_path, device)
    data = data[a.rank :: a.world]
    out_path = os.path.join(a.out_dir, f"predictions_rank{a.rank}.json")
    recs = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r["scan_id"], int(r["turn_id"])) for r in recs}
    for si, s in enumerate(tqdm(data, desc=f"rank{a.rank}", dynamic_ncols=True)):
        sid = s["scan_id"]
        if all((sid, t["turn_id"]) in done for t in s["user_message"]):
            continue
        paths, turn_ids, in_turn = frame_metadata(s)
        cached_features, cached_grid = None, None
        if a.feature_root:
            cache_path = Path(a.feature_root) / f"{sid}.npz"
            with np.load(cache_path, allow_pickle=False) as cache:
                if str(cache["protocol"]) != PROTOCOL or cache["frames"].tolist() != paths:
                    raise ValueError(f"Incompatible OST cache: {cache_path}")
                cached_features = torch.from_numpy(cache["patch_features"])
                cached_grid = torch.from_numpy(cache["grid_thw"])
        stream = OSTStream(model)
        seen = 0
        for t in s["user_message"]:
            end = seen + len(t["image_paths"])
            if cached_features is None:
                frames = load_frames(t["image_paths"], a.image_root)
                features, grid = vit(model.language_model, processor, frames, device)
                features = torch.stack(features)
            else:
                features, grid = cached_features[seen:end], cached_grid[seen:end]
            cur_turn_ids = turn_ids[seen:end]
            stream.observe(features, grid, turn_ids=cur_turn_ids)
            seen = end
            qa = qa_record(t, seen)
            qa["official_system_prompt"] = s["system_prompt"]
            cur_frame_range = list(range(seen - len(t["image_paths"]), seen))
            cur_labels = image_labels(turn_ids[:seen], in_turn[:seen], cur_frame_range)
            if (sid, t["turn_id"]) in done:
                stream.record_turn(qa, rec_lookup_pred(recs, sid, t["turn_id"]), cur_labels, cur_frame_range)
                continue
            prepared = stream.prepare(builder, qa, turn_ids[:seen], in_turn[:seen])
            pred = greedy_decode(model, builder, prepared, a.max_new_tokens)
            stream.record_turn(qa, pred, cur_labels, cur_frame_range)
            r = {"session": sid, "scan_id": s["scan_id"], "turn_id": t["turn_id"], "type": t["type"],
                 "option": t["option"], "answer": t["answer"], "prediction": pred}
            r["score"] = scorer(r)
            recs.append(r)
            done.add((sid, int(t["turn_id"])))
        atomic_json(Path(out_path), recs)
    m = aggregate_with_scorer(recs, a.scorer)
    atomic_json(Path(out_path), recs)
    atomic_json(Path(a.out_dir) / f"metrics_rank{a.rank}.json", m)
    print(json.dumps(m))


if __name__ == "__main__":
    main()
