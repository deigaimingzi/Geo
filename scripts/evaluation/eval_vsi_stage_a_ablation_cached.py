"""VSI-Bench Stage A Memory ablation — reads from pre-cached features.

Loads cached ViT + CUT3R npz (produced by cache_vsi_features.py), samples ray
queries from teacher maps, runs recurrent memory update, then evaluates 4
readout conditions: full / zero / init / shuffle.

Scene-level paired bootstrap gives 95% CI + frac(Δ>0).

Usage:
  python scripts/evaluation/eval_vsi_stage_a_ablation_cached.py \
    --ckpt output/v8/stage_a \
    --cache_dir /data/juicefs_sharing_data/11188211/VIT-feather/v8_vit512/features/vsi_bench_test \
    --out output/v8/vsi_ablation.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from geomemory.config import GeoMemoryConfig  # noqa: E402
from geomemory.data.v71_online_rays import sample_queries  # noqa: E402
from geomemory.position import grid_to_patch_xy  # noqa: E402
from geomemory.state import GeoMemoryState  # noqa: E402
from geomemory.train import _load_geomemory_checkpoint  # noqa: E402
from geomemory.training_model import GeoMemoryTrainingModel  # noqa: E402


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------
def _zero_state_like(state):
    return GeoMemoryState(
        memory=torch.zeros_like(state.memory),
        pose=None if state.pose is None else torch.zeros_like(state.pose),
        step=state.step,
    )


def _shuffled_state(state):
    if state.memory.shape[0] < 2:
        raise ValueError("Shuffle requires batch >= 2")
    return GeoMemoryState(
        memory=torch.roll(state.memory, shifts=1, dims=0),
        pose=None if state.pose is None else torch.roll(state.pose, shifts=1, dims=0),
        step=state.step,
    )


@torch.no_grad()
def _mae(pred_xyz, target_xyz, mask):
    if not bool(mask.any()):
        return None
    err = (pred_xyz.float() - target_xyz.float()).abs().mean(dim=-1)
    m = mask.to(err.dtype)
    return float((err * m).sum().item() / m.sum().clamp_min(1.0).item())


@torch.no_grad()
def _read_mae(model, state, ray_queries, targets, mask):
    """ray_queries: [1, R, D], targets: [R, 3], mask: [R]."""
    if not bool(mask.any()):
        return None
    pred = model.geomemory.read_geometry(state, ray_queries).xyz_world  # [1, R, 3]
    return _mae(pred[0], targets, mask)


def _filter_mask(mask, conf, t, min_conf):
    if mask is None:
        return None
    m = mask[:, t]
    if conf is not None:
        m = m & (conf[:, t] >= min_conf)
    return m


def paired_bootstrap(deltas, n_boot=10000, seed=0, ci=0.95):
    arr = np.array([d for d in deltas if d is not None and not np.isnan(d)], dtype=np.float64)
    n = len(arr)
    if n == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(arr.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


# ---------------------------------------------------------------------------
# Load one scene from cache
# ---------------------------------------------------------------------------
def load_scene(cache_path):
    z = np.load(cache_path, allow_pickle=True)
    pf = torch.from_numpy(z["patch_features"]).to(torch.bfloat16)  # [T,N,D]
    grid = z["grid_thw"]  # [T,3]
    aff = z["teacher_uv_affine"]  # [T,4]
    pts3d = z["pts3d"].astype(np.float16)
    conf = z["conf"].astype(np.float16)
    pose = z["pose"].astype(np.float32)
    T = pf.shape[0]
    n_tok = pf.shape[1]
    teacher = {"pts3d": pts3d, "conf": conf, "pose": pose}
    # Build patch_xy from grid
    xy = torch.stack([grid_to_patch_xy(grid[t], n_tok, 2) for t in range(T)]).to(torch.bfloat16)
    return pf, xy, teacher, aff, n_tok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--query_count", type=int, default=128)
    ap.add_argument("--heldout_count", type=int, default=64)
    ap.add_argument("--heldout_p", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    cfg = GeoMemoryConfig.from_json(a.config or os.path.join(a.ckpt, "geomemory_config.json"))
    model = GeoMemoryTrainingModel(cfg, language_model=None, freeze_language_model=True)
    _load_geomemory_checkpoint(model, a.ckpt)
    model.to(dev, torch.bfloat16).eval()
    min_conf = cfg.loss.min_confidence

    # --- Load scene index ---
    index_path = os.path.join(a.cache_dir, "scene_index.jsonl")
    if not os.path.exists(index_path):
        # merge rank files
        import glob
        parts = sorted(glob.glob(os.path.join(a.cache_dir, "scene_index.rank*.jsonl")))
        with open(index_path, "w") as fh:
            for p in parts:
                with open(p) as f:
                    fh.write(f.read())
    scenes = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                continue
            if not os.path.exists(d["npz"]):
                continue
            scenes.append(d)
    if a.max_scenes > 0:
        scenes = scenes[:a.max_scenes]
    print(f"{len(scenes)} scenes to evaluate", flush=True)

    # Per-scene accumulators
    scene_mae = {
        f"{cond}_{kind}": []
        for cond in ("full", "zero", "init", "shuffle")
        for kind in ("current", "history", "heldout")
    }
    scene_names = []
    errors = []

    t0 = time.time()
    idx = 0
    while idx < len(scenes):
        # Build a batch of batch_size scenes
        chunk = []
        for _ in range(min(a.batch_size, len(scenes) - idx)):
            chunk.append(scenes[idx])
            idx += 1
        if len(chunk) < 2:
            chunk.append(chunk[0])  # pad for shuffle

        B = len(chunk)
        try:
            # Load all scenes in batch
            pf_list, xy_list, teacher_list, aff_list = [], [], [], []
            max_T = 0
            for s in chunk:
                pf, xy, teacher, aff, n_tok = load_scene(s["npz"])
                pf_list.append(pf)
                xy_list.append(xy)
                teacher_list.append(teacher)
                aff_list.append(aff)
                max_T = max(max_T, pf.shape[0])

            # Pad to same T (all should be 32, but just in case)
            T = max_T
            D = pf_list[0].shape[-1]
            N = pf_list[0].shape[1]
            pf_batch = torch.zeros(B, T, N, D, dtype=torch.bfloat16, device=dev)
            xy_batch = torch.zeros(B, T, N, 2, dtype=torch.bfloat16, device=dev)
            frame_mask = torch.zeros(B, T, device=dev, dtype=torch.bool)
            for bi, (pf, xy) in enumerate(zip(pf_list, xy_list)):
                t = pf.shape[0]
                pf_batch[bi, :t] = pf.to(dev)
                xy_batch[bi, :t] = xy.to(dev)
                frame_mask[bi, :t] = True

            # Sample ray queries per scene (need per-scene because teacher differs)
            # We process per-scene ray sampling but batch the readout.
            # For simplicity, sample per scene and pad to max R.
            all_queries = []
            for si, (teacher, aff) in enumerate(zip(teacher_list, aff_list)):
                rng = np.random.default_rng(a.seed + idx - len(chunk) + si)
                q = sample_queries(
                    teacher, rng,
                    cur_count=a.query_count,
                    hist_count=a.query_count,
                    heldout_count=a.heldout_count,
                    heldout_p=a.heldout_p,
                    uv_affine=aff,
                )
                all_queries.append(q)

            # Determine R dimensions (may vary slightly per scene)
            def max_R(key):
                return max(q[key].shape[1] for q in all_queries)

            cur_R = max_R("current_ray_queries")
            hist_R = max_R("history_ray_queries")
            ho_R = max_R("heldout_ray_queries")

            def pad_rays(key, dtype, R):
                # ray_queries: [T, R, D], query_mask: [T, R]
                sample = all_queries[0][key]
                feat_dim = sample.shape[-1] if sample.ndim == 3 else 1
                if sample.ndim == 3:
                    out = torch.zeros(B, T, R, feat_dim, dtype=dtype, device=dev)
                else:
                    out = torch.zeros(B, T, R, dtype=dtype, device=dev)
                mask = torch.zeros(B, T, R, dtype=torch.bool, device=dev)
                for bi, q in enumerate(all_queries):
                    v = torch.from_numpy(q[key])
                    t_dim = v.shape[0]
                    r_dim = v.shape[1]
                    out[bi, :t_dim, :r_dim] = v.to(dev, dtype)
                    mask[bi, :t_dim, :r_dim] = torch.from_numpy(q[key.replace("ray_queries", "query_mask")]).to(dev)
                return out, mask

            crq, cm_raw = pad_rays("current_ray_queries", torch.bfloat16, cur_R)
            cxt = torch.zeros(B, T, cur_R, 3, dtype=torch.float32, device=dev)
            cc = torch.zeros(B, T, cur_R, dtype=torch.float32, device=dev)
            for bi, q in enumerate(all_queries):
                v = torch.from_numpy(q["current_xyz_targets"])
                cxt[bi, :v.shape[0], :v.shape[1]] = v.to(dev)
                v2 = torch.from_numpy(q["current_query_confidence"])
                cc[bi, :v2.shape[0], :v2.shape[1]] = v2.to(dev)

            hrq, hm_raw = pad_rays("history_ray_queries", torch.bfloat16, hist_R)
            hxt = torch.zeros(B, T, hist_R, 3, dtype=torch.float32, device=dev)
            hc = torch.zeros(B, T, hist_R, dtype=torch.float32, device=dev)
            for bi, q in enumerate(all_queries):
                v = torch.from_numpy(q["history_xyz_targets"])
                hxt[bi, :v.shape[0], :v.shape[1]] = v.to(dev)
                v2 = torch.from_numpy(q["history_query_confidence"])
                hc[bi, :v2.shape[0], :v2.shape[1]] = v2.to(dev)

            orq, om_raw = pad_rays("heldout_ray_queries", torch.bfloat16, ho_R)
            oxt = torch.zeros(B, T, ho_R, 3, dtype=torch.float32, device=dev)
            oc = torch.zeros(B, T, ho_R, dtype=torch.float32, device=dev)
            for bi, q in enumerate(all_queries):
                v = torch.from_numpy(q["heldout_xyz_targets"])
                oxt[bi, :v.shape[0], :v.shape[1]] = v.to(dev)
                v2 = torch.from_numpy(q["heldout_query_confidence"])
                oc[bi, :v2.shape[0], :v2.shape[1]] = v2.to(dev)

            heldout_frames = [int(q["heldout_frame"]) for q in all_queries]
            update_mask = frame_mask.clone()
            for bi, hf in enumerate(heldout_frames):
                if hf >= 0 and hf < T:
                    update_mask[bi, hf] = False

            # --- Run recurrent memory update ---
            core = model.geomemory(
                pf_batch, patch_xy=xy_batch,
                frame_mask=frame_mask,
                update_mask=update_mask,
            )
            state_history = core.state_history
            if not state_history:
                raise ValueError("no state_history")

            st_init = model.geomemory.new_scene_state(B, device=dev, dtype=torch.bfloat16)

            # --- Evaluate per-prefix ---
            per_t = {f"{cond}_{kind}": {bi: [] for bi in range(B)}
                     for cond in ("full", "zero", "init", "shuffle")
                     for kind in ("current", "history", "heldout")}

            for t in range(T):
                st_full = state_history[t]
                st_zero = _zero_state_like(st_full)
                try:
                    st_shuf = _shuffled_state(st_full)
                except ValueError:
                    st_shuf = None

                # current
                m = _filter_mask(cm_raw, cc, t, min_conf)
                if m is not None and bool(m.any()):
                    rq = crq[:, t]
                    for bi in range(B):
                        mi = m[bi]
                        if bool(mi.any()):
                            for cn, st in [("full", st_full), ("zero", st_zero), ("init", st_init)]:
                                v = _read_mae(model, st.index_select(torch.tensor([bi], device=dev)),
                                              rq[bi:bi+1], cxt[bi, t], mi)
                                if v is not None:
                                    per_t[f"{cn}_current"][bi].append(v)
                            if st_shuf is not None:
                                v = _read_mae(model, st_shuf.index_select(torch.tensor([bi], device=dev)),
                                              rq[bi:bi+1], cxt[bi, t], mi)
                                if v is not None:
                                    per_t["shuffle_current"][bi].append(v)

                # history
                m = _filter_mask(hm_raw, hc, t, min_conf)
                if t > 0 and m is not None and bool(m.any()):
                    rq = hrq[:, t]
                    for bi in range(B):
                        mi = m[bi]
                        if bool(mi.any()):
                            for cn, st in [("full", st_full), ("zero", st_zero), ("init", st_init)]:
                                v = _read_mae(model, st.index_select(torch.tensor([bi], device=dev)),
                                              rq[bi:bi+1], hxt[bi, t], mi)
                                if v is not None:
                                    per_t[f"{cn}_history"][bi].append(v)
                            if st_shuf is not None:
                                v = _read_mae(model, st_shuf.index_select(torch.tensor([bi], device=dev)),
                                              rq[bi:bi+1], hxt[bi, t], mi)
                                if v is not None:
                                    per_t["shuffle_history"][bi].append(v)

                # heldout
                m = _filter_mask(om_raw, oc, t, min_conf)
                if m is not None and bool(m.any()):
                    rq = orq[:, t]
                    for bi in range(B):
                        mi = m[bi]
                        if bool(mi.any()):
                            for cn, st in [("full", st_full), ("zero", st_zero), ("init", st_init)]:
                                v = _read_mae(model, st.index_select(torch.tensor([bi], device=dev)),
                                              rq[bi:bi+1], oxt[bi, t], mi)
                                if v is not None:
                                    per_t[f"{cn}_heldout"][bi].append(v)
                            if st_shuf is not None:
                                v = _read_mae(model, st_shuf.index_select(torch.tensor([bi], device=dev)),
                                              rq[bi:bi+1], oxt[bi, t], mi)
                                if v is not None:
                                    per_t["shuffle_heldout"][bi].append(v)

            # Collapse per-t to per-scene mean
            real_count = min(a.batch_size, len(scenes) - (idx - len(chunk)))
            for key, scene_dict in per_t.items():
                for bi in range(B):
                    vals = scene_dict[bi]
                    scene_mae[key].append(float(np.mean(vals)) if vals else None)

            for bi in range(real_count):
                scene_names.append(chunk[bi]["scene_key"])

        except Exception as e:
            import traceback
            for bi in range(len(chunk)):
                for key in scene_mae:
                    scene_mae[key].append(None)
                scene_names.append(chunk[bi].get("scene_key", "unknown"))
            errors.append({"scenes": [s.get("scene_key") for s in chunk], "error": str(e)[:300],
                           "trace": traceback.format_exc()[-500:]})
            print(f"ERROR: {e}", flush=True)

        if idx % 20 == 0 or idx >= len(scenes):
            elapsed = time.time() - t0
            print(f"{idx}/{len(scenes)} ({idx/elapsed:.2f}/s {elapsed/60:.1f}min)", flush=True)

    # --- Aggregate ---
    def safe_list(key):
        return scene_mae.get(key, [])

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
    scene_mae["full_overall"] = full_ov
    scene_mae["zero_overall"] = zero_ov
    scene_mae["init_overall"] = init_ov
    scene_mae["shuffle_overall"] = shuf_ov

    def delta_list(a_list, b_list):
        out = []
        for x, y in zip(a_list, b_list):
            if x is not None and y is not None:
                out.append(y - x)
            else:
                out.append(None)
        return out

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

    results = {}
    for cond in ("full", "zero", "init", "shuffle"):
        for kind in ("current", "history", "heldout", "overall"):
            key = f"{cond}_{kind}"
            vals = [v for v in safe_list(key) if v is not None]
            results[f"mae_{key}"] = float(np.mean(vals)) if vals else None

    for dkey, dvals in deltas.items():
        mean, lo, hi = paired_bootstrap(dvals, n_boot=a.n_boot, seed=a.seed)
        arr = np.array([d for d in dvals if d is not None and not np.isnan(d)])
        pos_frac = float((arr > 0).mean()) if len(arr) > 0 else None
        results[f"delta_{dkey}"] = {
            "mean": mean, "ci95_lo": lo, "ci95_hi": hi,
            "frac_positive": pos_frac, "n_scenes": int(len(arr)),
        }

    results["meta"] = {
        "ckpt": a.ckpt,
        "cache_dir": a.cache_dir,
        "max_scenes": a.max_scenes,
        "query_count": a.query_count,
        "heldout_count": a.heldout_count,
        "batch_size": a.batch_size,
        "n_boot": a.n_boot,
        "seed": a.seed,
        "n_scenes_evaluated": len([v for v in scene_names if v]),
        "n_errors": len(errors),
        "errors": errors[:10],
    }

    print(json.dumps(results, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults written to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
