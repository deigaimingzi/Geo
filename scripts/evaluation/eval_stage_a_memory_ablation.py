"""Stage A Memory ablation (inference-time intervention).

Loads the trained v8 Stage A checkpoint (epoch 20) and, on the SAME validation
samples / SAME rays / SAME teacher targets, runs three readout conditions:

  full    : read_geometry(state, ray)                      -- baseline
  zero    : read_geometry(zero_memory_state, ray)          -- Memory removed
  shuffle : read_geometry(rolled_memory_state, ray)        -- Memory from another scene

The recurrent update (model.update / memory) is ALWAYS run normally; only the
state handed to read_geometry is swapped.  This isolates whether the Readout
depends on scene-specific Memory, without re-training.

Reports per-prefix XYZ MAE (CUT3R units) for current / history / heldout, plus
the Memory Dependence deltas  Δ_zero = E_zero - E_full  and
Δ_shuffle = E_shuffle - E_full.

This script does NOT modify any training code or the checkpoint; it only reads
output/v8/stage_a.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from geomemory.config import GeoMemoryConfig  # noqa: E402
from geomemory.data.collator import GeoMemoryCollator  # noqa: E402
from geomemory.data.dataset import GeoMemoryDataset  # noqa: E402
from geomemory.state import GeoMemoryState  # noqa: E402
from geomemory.train import _load_geomemory_checkpoint  # noqa: E402
from geomemory.training_model import GeoMemoryTrainingModel  # noqa: E402


def _zero_state_like(state: GeoMemoryState) -> GeoMemoryState:
    """Same shape / device / dtype as ``state`` but all-zero memory."""
    return GeoMemoryState(
        memory=torch.zeros_like(state.memory),
        pose=None if state.pose is None else torch.zeros_like(state.pose),
        step=state.step,
    )


def _shuffled_state(state: GeoMemoryState) -> GeoMemoryState:
    """Roll the batch dim by 1 so each scene reads another scene's memory.

    Requires batch >= 2.  The resulting memory is a real model output (no
    distribution shift to an unseen zero tensor), but it no longer matches the
    ray's scene / coordinate frame.
    """
    if state.memory.shape[0] < 2:
        raise ValueError("Shuffle Memory requires batch >= 2")
    return GeoMemoryState(
        memory=torch.roll(state.memory, shifts=1, dims=0),
        pose=None if state.pose is None else torch.roll(state.pose, shifts=1, dims=0),
        step=state.step,
    )


@torch.no_grad()
def _mae(pred_xyz, target_xyz, mask):
    """Masked mean-absolute-error over the last (xyz) dim, then over valid rays."""
    if not bool(mask.any()):
        return None
    err = (pred_xyz.float() - target_xyz.float()).abs().mean(dim=-1)  # [B, R]
    m = mask.to(err.dtype)
    return float((err * m).sum().item() / m.sum().clamp_min(1.0).item())


def _collect(prefix: str, acc: dict, mae_current, mae_history, mae_heldout):
    """Accumulate per-prefix MAEs into the running lists keyed by prefix."""
    if mae_current is not None:
        acc.setdefault(f"{prefix}_current", []).append(mae_current)
    if mae_history is not None:
        acc.setdefault(f"{prefix}_history", []).append(mae_history)
    if mae_heldout is not None:
        acc.setdefault(f"{prefix}_heldout", []).append(mae_heldout)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default=os.path.join(ROOT, "output", "v8", "stage_a"),
        help="Stage A checkpoint directory (default: output/v8/stage_a)",
    )
    ap.add_argument(
        "--manifest",
        default=os.path.join(ROOT, "output", "v8", "manifests", "stageA_train.jsonl"),
        help="Manifest to evaluate on (default: v8 stageA_train.jsonl)",
    )
    ap.add_argument("--config", default="", help="Override config path")
    ap.add_argument("--max_samples", type=int, default=64, help="# scenes to evaluate")
    ap.add_argument("--query_count", type=int, default=128, help="rays per frame")
    ap.add_argument("--heldout_count", type=int, default=64, help="held-out rays")
    ap.add_argument("--heldout_p", type=float, default=1.0, help="force a held-out frame")
    ap.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Scenes per batch; must be >= 2 for Shuffle Memory",
    )
    ap.add_argument(
        "--prefixes",
        default="",
        help="Comma list of prefix lengths to report (e.g. 8,16,24,32); empty = all",
    )
    ap.add_argument("--out", default="", help="Write JSON results here")
    ap.add_argument("--seed", type=int, default=1234, help="Dataset ray sampling seed")
    a = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = GeoMemoryConfig.from_json(
        a.config or os.path.join(a.ckpt, "geomemory_config.json")
    )
    model = GeoMemoryTrainingModel(cfg, language_model=None, freeze_language_model=True)
    _load_geomemory_checkpoint(model, a.ckpt)
    model.to(dev, torch.bfloat16).eval()

    # Match v8 Stage A data settings (v7_stratified + teacher_cache32).
    ds = GeoMemoryDataset(
        a.manifest,
        stage="geometry",
        v7_stratified=True,
        heldout_count=a.heldout_count,
        heldout_p=a.heldout_p,
        query_count=a.query_count,
        max_samples=a.max_samples,
        seed=a.seed,
        deterministic_rays=True,  # reproducible rays across runs
    )
    col = GeoMemoryCollator(tokenizer=None, model_max_length=8192)

    want_prefixes = (
        set(int(x) for x in a.prefixes.split(",") if x.strip()) if a.prefixes else None
    )

    # accumulators: {condition_prefix: [mae per batch]}
    acc: dict = {}

    n = len(ds)
    idx = 0
    while idx < n:
        # Build a batch of batch_size scenes (pad the last chunk with repeats
        # so batch >= 2 for the shuffle condition; the repeated scene's MAE is
        # still valid, it just gets counted once via masking).
        chunk = []
        for _ in range(min(a.batch_size, n - idx)):
            chunk.append(ds[idx])
            idx += 1
        if len(chunk) < 2:
            # Top up to 2 by reusing the first scene (shuffle still works).
            chunk.append(chunk[0])
        b = col(chunk)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}

        # Cast inputs to bf16 like training.
        feats = b["patch_features"].to(torch.bfloat16)
        pxy = b.get("patch_xy")
        if pxy is not None:
            pxy = pxy.to(torch.bfloat16)
        T = feats.shape[1]

        # --- Run the recurrent update ONCE, capturing state_history[t] ---
        core = model.geomemory(
            feats,
            patch_xy=pxy,
            patch_mask=b.get("patch_mask"),
            frame_mask=b.get("frame_mask"),
            update_mask=b.get("update_mask"),
        )
        state_history = core.state_history  # list of GeoMemoryState, len T
        if not state_history:
            continue

        # For each prefix t, evaluate the three conditions on the SAME rays.
        for t in range(T):
            if want_prefixes is not None and (t + 1) not in want_prefixes:
                continue
            st_full = state_history[t]
            st_zero = _zero_state_like(st_full)
            try:
                st_shuf = _shuffled_state(st_full)
            except ValueError:
                st_shuf = None  # batch == 1 (shouldn't happen with the top-up)

            # ---- current rays (t_src == t) ----
            crq = b.get("current_ray_queries")
            cxt = b.get("current_xyz_targets")
            cm = b.get("current_query_mask")
            cc = b.get("current_query_confidence")
            cur_full = cur_zero = cur_shuf = None
            if crq is not None and t < crq.shape[1]:
                rq = crq[:, t].to(torch.bfloat16)
                mask = cm[:, t] if cm is not None else None
                if cc is not None:
                    mask = mask & (cc[:, t] >= cfg.loss.min_confidence)
                if mask is not None and bool(mask.any()):
                    p_f = model.geomemory.read_geometry(st_full, rq).xyz_world
                    p_z = model.geomemory.read_geometry(st_zero, rq).xyz_world
                    cur_full = _mae(p_f, cxt[:, t], mask)
                    cur_zero = _mae(p_z, cxt[:, t], mask)
                    if st_shuf is not None:
                        p_s = model.geomemory.read_geometry(st_shuf, rq).xyz_world
                        cur_shuf = _mae(p_s, cxt[:, t], mask)

            # ---- history rays (t_src < t) ----
            hrq = b.get("history_ray_queries")
            hxt = b.get("history_xyz_targets")
            hm = b.get("history_query_mask")
            hc = b.get("history_query_confidence")
            hist_full = hist_zero = hist_shuf = None
            if hrq is not None and t > 0 and t < hrq.shape[1]:
                rq = hrq[:, t].to(torch.bfloat16)
                mask = hm[:, t] if hm is not None else None
                if hc is not None:
                    mask = mask & (hc[:, t] >= cfg.loss.min_confidence)
                if mask is not None and bool(mask.any()):
                    p_f = model.geomemory.read_geometry(st_full, rq).xyz_world
                    p_z = model.geomemory.read_geometry(st_zero, rq).xyz_world
                    hist_full = _mae(p_f, hxt[:, t], mask)
                    hist_zero = _mae(p_z, hxt[:, t], mask)
                    if st_shuf is not None:
                        p_s = model.geomemory.read_geometry(st_shuf, rq).xyz_world
                        hist_shuf = _mae(p_s, hxt[:, t], mask)

            # ---- held-out rays (frame never written to memory) ----
            orq = b.get("heldout_ray_queries")
            oxt = b.get("heldout_xyz_targets")
            om = b.get("heldout_query_mask")
            oc = b.get("heldout_query_confidence")
            ho_full = ho_zero = ho_shuf = None
            if orq is not None and t < orq.shape[1]:
                rq = orq[:, t].to(torch.bfloat16)
                mask = om[:, t] if om is not None else None
                if oc is not None:
                    mask = mask & (oc[:, t] >= cfg.loss.min_confidence)
                if mask is not None and bool(mask.any()):
                    p_f = model.geomemory.read_geometry(st_full, rq).xyz_world
                    p_z = model.geomemory.read_geometry(st_zero, rq).xyz_world
                    ho_full = _mae(p_f, oxt[:, t], mask)
                    ho_zero = _mae(p_z, oxt[:, t], mask)
                    if st_shuf is not None:
                        p_s = model.geomemory.read_geometry(st_shuf, rq).xyz_world
                        ho_shuf = _mae(p_s, oxt[:, t], mask)

            _collect("full", acc, cur_full, hist_full, ho_full)
            _collect("zero", acc, cur_zero, hist_zero, ho_zero)
            _collect("shuffle", acc, cur_shuf, hist_shuf, ho_shuf)

    # --- Aggregate ---
    def mean(key):
        vals = acc.get(key)
        return float(np.mean(vals)) if vals else None

    def delta(a_key, b_key):
        ma, mb = mean(a_key), mean(b_key)
        if ma is None or mb is None:
            return None
        return mb - ma  # b - a  (e.g. zero - full)

    prefixes_used = sorted(
        {t + 1 for i in range(n) for t in range(len(ds[i]["patch_features"]))}
    ) if want_prefixes is None else sorted(want_prefixes)

    results = {}
    for cond in ("full", "zero", "shuffle"):
        for kind in ("current", "history", "heldout"):
            key = f"{cond}_{kind}"
            results[key] = mean(key)

    # Memory dependence deltas (error increase when Memory is removed/replaced).
    for kind in ("current", "history", "heldout"):
        results[f"delta_zero_{kind}"] = delta(f"full_{kind}", f"zero_{kind}")
        results[f"delta_shuffle_{kind}"] = delta(f"full_{kind}", f"shuffle_{kind}")

    # Overall (weighted equally across the three kinds that exist).
    all_full = [v for k, v in results.items() if k.startswith("full_") and v is not None]
    all_zero = [v for k, v in results.items() if k.startswith("zero_") and v is not None]
    all_shuf = [v for k, v in results.items() if k.startswith("shuffle_") and v is not None]
    results["full_overall"] = float(np.mean(all_full)) if all_full else None
    results["zero_overall"] = float(np.mean(all_zero)) if all_zero else None
    results["shuffle_overall"] = float(np.mean(all_shuf)) if all_shuf else None
    if results["full_overall"] is not None and results["zero_overall"] is not None:
        results["delta_zero_overall"] = results["zero_overall"] - results["full_overall"]
    if results["full_overall"] is not None and results["shuffle_overall"] is not None:
        results["delta_shuffle_overall"] = results["shuffle_overall"] - results["full_overall"]

    results["meta"] = {
        "ckpt": a.ckpt,
        "manifest": a.manifest,
        "max_samples": a.max_samples,
        "query_count": a.query_count,
        "heldout_count": a.heldout_count,
        "batch_size": a.batch_size,
        "n_scenes_evaluated": n,
        "prefixes_reported": prefixes_used,
        "seed": a.seed,
    }

    print(json.dumps(results, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults written to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
