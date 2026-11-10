"""Stage A Memory ablation: scene-level paired bootstrap + Initial-Memory baseline.

Loads the trained v8 Stage A checkpoint (epoch 20) and, on the SAME validation
samples / SAME rays / SAME teacher targets, runs FOUR readout conditions:

  full    : read_geometry(state_history[t], ray)           -- trained recurrent memory
  zero    : read_geometry(zero_memory_state, ray)          -- Memory removed (zeros)
  init    : read_geometry(initial_state, ray)              -- trained initial_memory param
  shuffle : read_geometry(rolled_memory_state, ray)        -- Memory from another scene

The recurrent update (model.update / memory) is ALWAYS run normally; only the
state handed to read_geometry is swapped.  This isolates whether the Readout
depends on scene-specific Memory, without re-training.

Per-scene MAE is recorded for each condition, then paired bootstrap (resampling
scenes, not rays) gives 95% CI for Δ_zero, Δ_init, Δ_shuffle.  Also reports the
fraction of scenes with Δ > 0.

Outputs JSON to stdout and optionally --out.
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
    return GeoMemoryState(
        memory=torch.zeros_like(state.memory),
        pose=None if state.pose is None else torch.zeros_like(state.pose),
        step=state.step,
    )


def _shuffled_state(state: GeoMemoryState) -> GeoMemoryState:
    if state.memory.shape[0] < 2:
        raise ValueError("Shuffle Memory requires batch >= 2")
    return GeoMemoryState(
        memory=torch.roll(state.memory, shifts=1, dims=0),
        pose=None if state.pose is None else torch.roll(state.pose, shifts=1, dims=0),
        step=state.step,
    )


@torch.no_grad()
def _mae_per_scene(pred_xyz, target_xyz, mask, batch_size):
    """Return a list of per-scene MAE values (None if scene has no valid rays).

    pred_xyz / target_xyz: [B, R, 3]
    mask: [B, R] bool
    """
    err = (pred_xyz.float() - target_xyz.float()).abs().mean(dim=-1)  # [B, R]
    out = []
    for b in range(batch_size):
        m = mask[b]
        if not bool(m.any()):
            out.append(None)
        else:
            mf = m.to(err.dtype)
            out.append(float((err[b] * mf).sum().item() / mf.sum().clamp_min(1.0).item()))
    return out


@torch.no_grad()
def _read_mae(model, state, ray_queries, targets, mask, batch_size):
    """Run read_geometry and return per-scene MAE list."""
    if ray_queries is None:
        return [None] * batch_size
    pred = model.geomemory.read_geometry(state, ray_queries).xyz_world
    return _mae_per_scene(pred, targets, mask, batch_size)


def _filter_mask(mask, conf, t, min_conf):
    """Combine query_mask with confidence threshold at time t."""
    if mask is None:
        return None
    m = mask[:, t]
    if conf is not None:
        m = m & (conf[:, t] >= min_conf)
    return m


def paired_bootstrap(deltas, n_boot=10000, seed=0, ci=0.95):
    """Paired bootstrap CI for the mean of a list of scalar deltas.

    deltas: list of floats (NaN/None dropped).  Resampling unit = one delta
    (which is already per-scene).  Returns (mean, lo, hi).
    """
    arr = np.array([d for d in deltas if d is not None and not np.isnan(d)], dtype=np.float64)
    n = len(arr)
    if n == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return float(arr.mean()), lo, hi


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default=os.path.join(ROOT, "output", "v8", "stage_a"),
    )
    ap.add_argument(
        "--manifest",
        default=os.path.join(ROOT, "output", "v8", "manifests", "stageA_train.jsonl"),
    )
    ap.add_argument("--config", default="")
    ap.add_argument("--max_samples", type=int, default=128)
    ap.add_argument("--query_count", type=int, default=128)
    ap.add_argument("--heldout_count", type=int, default=64)
    ap.add_argument("--heldout_p", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--n_boot", type=int, default=10000, help="bootstrap iterations")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = GeoMemoryConfig.from_json(
        a.config or os.path.join(a.ckpt, "geomemory_config.json")
    )
    model = GeoMemoryTrainingModel(cfg, language_model=None, freeze_language_model=True)
    _load_geomemory_checkpoint(model, a.ckpt)
    model.to(dev, torch.bfloat16).eval()

    ds = GeoMemoryDataset(
        a.manifest,
        stage="geometry",
        v7_stratified=True,
        heldout_count=a.heldout_count,
        heldout_p=a.heldout_p,
        query_count=a.query_count,
        max_samples=a.max_samples,
        seed=a.seed,
        deterministic_rays=True,
    )
    col = GeoMemoryCollator(tokenizer=None, model_max_length=8192)

    min_conf = cfg.loss.min_confidence

    # Per-scene accumulators.  Each entry = one scene's MAE for that condition+kind.
    # Indexed by scene_id (sequential 0..N-1).
    scene_mae = {
        f"{cond}_{kind}": []
        for cond in ("full", "zero", "init", "shuffle")
        for kind in ("current", "history", "heldout")
    }
    scene_ids = []

    n = len(ds)
    idx = 0
    scene_seq = 0
    while idx < n:
        chunk = []
        for _ in range(min(a.batch_size, n - idx)):
            chunk.append(ds[idx])
            idx += 1
        if len(chunk) < 2:
            chunk.append(chunk[0])
        b = col(chunk)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}

        feats = b["patch_features"].to(torch.bfloat16)
        pxy = b.get("patch_xy")
        if pxy is not None:
            pxy = pxy.to(torch.bfloat16)
        T = feats.shape[1]
        B = feats.shape[0]

        # --- Run the recurrent update ONCE, capturing state_history ---
        core = model.geomemory(
            feats,
            patch_xy=pxy,
            patch_mask=b.get("patch_mask"),
            frame_mask=b.get("frame_mask"),
            update_mask=b.get("update_mask"),
        )
        state_history = core.state_history
        if not state_history:
            continue

        # --- Initial-Memory state: trained initial_memory, step=0 ---
        # This is the state BEFORE any frame is written.  Same for all t.
        st_init = model.geomemory.new_scene_state(B, device=dev, dtype=torch.bfloat16)

        # For each prefix t, evaluate all four conditions on the SAME rays.
        # We accumulate per-scene MAE averaged over t for each kind.
        # To keep per-scene granularity, we collect per-scene-per-t MAEs then
        # average over t per scene at the end.
        per_scene_t = {
            f"{cond}_{kind}": {b_i: [] for b_i in range(B)}
            for cond in ("full", "zero", "init", "shuffle")
            for kind in ("current", "history", "heldout")
        }

        for t in range(T):
            st_full = state_history[t]
            st_zero = _zero_state_like(st_full)
            try:
                st_shuf = _shuffled_state(st_full)
            except ValueError:
                st_shuf = None

            # current
            crq = b.get("current_ray_queries")
            cxt = b.get("current_xyz_targets")
            cm = _filter_mask(b.get("current_query_mask"), b.get("current_query_confidence"), t, min_conf)
            if crq is not None and t < crq.shape[1] and cm is not None and bool(cm.any()):
                rq = crq[:, t].to(torch.bfloat16)
                vals = _read_mae(model, st_full, rq, cxt[:, t], cm, B)
                for bi, v in enumerate(vals):
                    if v is not None:
                        per_scene_t[f"full_current"][bi].append(v)
                vals = _read_mae(model, st_zero, rq, cxt[:, t], cm, B)
                for bi, v in enumerate(vals):
                    if v is not None:
                        per_scene_t[f"zero_current"][bi].append(v)
                vals = _read_mae(model, st_init, rq, cxt[:, t], cm, B)
                for bi, v in enumerate(vals):
                    if v is not None:
                        per_scene_t[f"init_current"][bi].append(v)
                if st_shuf is not None:
                    vals = _read_mae(model, st_shuf, rq, cxt[:, t], cm, B)
                    for bi, v in enumerate(vals):
                        if v is not None:
                            per_scene_t[f"shuffle_current"][bi].append(v)

            # history
            hrq = b.get("history_ray_queries")
            hxt = b.get("history_xyz_targets")
            hm = _filter_mask(b.get("history_query_mask"), b.get("history_query_confidence"), t, min_conf)
            if hrq is not None and t > 0 and t < hrq.shape[1] and hm is not None and bool(hm.any()):
                rq = hrq[:, t].to(torch.bfloat16)
                for cond_name, st in [("full", st_full), ("zero", st_zero), ("init", st_init)]:
                    vals = _read_mae(model, st, rq, hxt[:, t], hm, B)
                    for bi, v in enumerate(vals):
                        if v is not None:
                            per_scene_t[f"{cond_name}_history"][bi].append(v)
                if st_shuf is not None:
                    vals = _read_mae(model, st_shuf, rq, hxt[:, t], hm, B)
                    for bi, v in enumerate(vals):
                        if v is not None:
                            per_scene_t[f"shuffle_history"][bi].append(v)

            # heldout
            orq = b.get("heldout_ray_queries")
            oxt = b.get("heldout_xyz_targets")
            om = _filter_mask(b.get("heldout_query_mask"), b.get("heldout_query_confidence"), t, min_conf)
            if orq is not None and t < orq.shape[1] and om is not None and bool(om.any()):
                rq = orq[:, t].to(torch.bfloat16)
                for cond_name, st in [("full", st_full), ("zero", st_zero), ("init", st_init)]:
                    vals = _read_mae(model, st, rq, oxt[:, t], om, B)
                    for bi, v in enumerate(vals):
                        if v is not None:
                            per_scene_t[f"{cond_name}_heldout"][bi].append(v)
                if st_shuf is not None:
                    vals = _read_mae(model, st_shuf, rq, oxt[:, t], om, B)
                    for bi, v in enumerate(vals):
                        if v is not None:
                            per_scene_t[f"shuffle_heldout"][bi].append(v)

        # Collapse per-scene-per-t lists into per-scene mean.
        for key, scene_dict in per_scene_t.items():
            for bi in range(B):
                vals = scene_dict[bi]
                scene_mae[key].append(float(np.mean(vals)) if vals else None)

        # Record scene IDs (skip padding duplicates for the last chunk).
        real_in_chunk = min(a.batch_size, n - (idx - len(chunk)))
        for bi in range(real_in_chunk):
            scene_ids.append(f"scene_{scene_seq}")
            scene_seq += 1

    # --- Aggregate per-scene deltas ---
    def safe_list(key):
        return scene_mae.get(key, [])

    full_cur = safe_list("full_current")
    zero_cur = safe_list("zero_current")
    init_cur = safe_list("init_current")
    shuf_cur = safe_list("shuffle_current")
    full_hist = safe_list("full_history")
    zero_hist = safe_list("zero_history")
    init_hist = safe_list("init_history")
    shuf_hist = safe_list("shuffle_history")
    full_ho = safe_list("full_heldout")
    zero_ho = safe_list("zero_heldout")
    init_ho = safe_list("init_heldout")
    shuf_ho = safe_list("shuffle_heldout")

    # Per-scene overall = mean across available kinds for that scene.
    def per_scene_overall(cond):
        cur = safe_list(f"{cond}_current")
        hist = safe_list(f"{cond}_history")
        ho = safe_list(f"{cond}_heldout")
        out = []
        for i in range(len(cur)):
            vals = [v for v in (cur[i], hist[i], ho[i]) if v is not None]
            out.append(float(np.mean(vals)) if vals else None)
        return out

    full_ov = per_scene_overall("full")
    zero_ov = per_scene_overall("zero")
    init_ov = per_scene_overall("init")
    shuf_ov = per_scene_overall("shuffle")
    # Store overall per-scene lists so safe_list() can find them below.
    scene_mae["full_overall"] = full_ov
    scene_mae["zero_overall"] = zero_ov
    scene_mae["init_overall"] = init_ov
    scene_mae["shuffle_overall"] = shuf_ov

    def delta_list(a_list, b_list):
        """b - a per scene, only where both are not None."""
        out = []
        for x, y in zip(a_list, b_list):
            if x is not None and y is not None:
                out.append(y - x)
            else:
                out.append(None)
        return out

    deltas = {
        "zero_current": delta_list(full_cur, zero_cur),
        "zero_history": delta_list(full_hist, zero_hist),
        "zero_heldout": delta_list(full_ho, zero_ho),
        "zero_overall": delta_list(full_ov, zero_ov),
        "init_current": delta_list(full_cur, init_cur),
        "init_history": delta_list(full_hist, init_hist),
        "init_heldout": delta_list(full_ho, init_ho),
        "init_overall": delta_list(full_ov, init_ov),
        "shuffle_current": delta_list(full_cur, shuf_cur),
        "shuffle_history": delta_list(full_hist, shuf_hist),
        "shuffle_heldout": delta_list(full_ho, shuf_ho),
        "shuffle_overall": delta_list(full_ov, shuf_ov),
    }

    # --- Paired bootstrap ---
    results = {}
    # Mean MAE per condition
    for cond in ("full", "zero", "init", "shuffle"):
        for kind in ("current", "history", "heldout", "overall"):
            key = f"{cond}_{kind}"
            vals = [v for v in safe_list(key) if v is not None]
            results[f"mae_{key}"] = float(np.mean(vals)) if vals else None

    # Bootstrap CI + positive fraction for each delta
    for dkey, dvals in deltas.items():
        mean, lo, hi = paired_bootstrap(dvals, n_boot=a.n_boot, seed=a.seed)
        arr = np.array([d for d in dvals if d is not None and not np.isnan(d)])
        pos_frac = float((arr > 0).mean()) if len(arr) > 0 else None
        n_valid = int(len(arr))
        results[f"delta_{dkey}"] = {
            "mean": mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "frac_positive": pos_frac,
            "n_scenes": n_valid,
        }

    results["meta"] = {
        "ckpt": a.ckpt,
        "manifest": a.manifest,
        "max_samples": a.max_samples,
        "query_count": a.query_count,
        "heldout_count": a.heldout_count,
        "batch_size": a.batch_size,
        "n_scenes_evaluated": scene_seq,
        "n_boot": a.n_boot,
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
