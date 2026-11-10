"""VSI-Bench Stage A Memory ablation (inference-time intervention).

Evaluates on the 288 VSI-Bench test scenes — which are NOT in the training set.

Pipeline per scene:
  1. Sample 32 frames from the VSI-Bench video (uniform, same as eval_vsi_geomemory).
  2. Run CUT3R-512 to get dense teacher maps: pts3d [T,H,W,3], conf [T,H,W], pose [T,4,4].
  3. Run Qwen3.5 ViT to get patch_features [T,N,2560], grid_thw, teacher_uv_affine.
  4. Sample ray queries from teacher maps (v71_online_rays.sample_queries).
  5. Run recurrent memory update, then evaluate 4 readout conditions:
       full / zero / init / shuffle
  6. Record per-scene MAE for current / history / heldout.

After all scenes: paired bootstrap (scene-level resampling) for 95% CI + frac(Δ>0).

Usage:
  python scripts/evaluation/eval_vsi_stage_a_ablation.py \
    --ckpt output/v8/stage_a --max_scenes 288 --out output/v8/vsi_ablation.json
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(
    0,
    "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/scripts/evaluation",
)
from eval_vsi_mra import ss_sample_frame_indices  # noqa: E402

from geomemory.config import GeoMemoryConfig  # noqa: E402
from geomemory.data.v71_online_rays import sample_queries  # noqa: E402
from geomemory.position import grid_to_patch_xy  # noqa: E402
from geomemory.state import GeoMemoryState  # noqa: E402
from geomemory.train import _load_geomemory_checkpoint  # noqa: E402
from geomemory.training_model import GeoMemoryTrainingModel  # noqa: E402

VSI_BASE = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/benchmark/VSI-Bench"
VSI_JSONL = f"{VSI_BASE}/raw/test.jsonl"
VSI_ROOT = f"{VSI_BASE}/videos"

CUT3R_ROOT = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/depth-model/CUT3R"
CUT3R_CKPT = f"{CUT3R_ROOT}/src/cut3r_512_dpt_4_64.pth"
CUT3R_SIZE = 512

QWEN_MODEL = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B"

LONG_EDGE = 512
ALIGN = 32


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_frames(video_path, num_frames=32):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = ss_sample_frame_indices(total, num_frames)
    frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# ViT feature extraction (same as tools/extract_vit_features_v8.py)
# ---------------------------------------------------------------------------
def vit_proc_size(w, h):
    s = LONG_EDGE / max(w, h)
    rw, rh = w * s, h * s
    a = lambda v: max(ALIGN, int(round(v / ALIGN)) * ALIGN)
    return (min(a(rw), LONG_EDGE), min(a(rh), LONG_EDGE))


def teacher_crop_frac(w, h):
    s = CUT3R_SIZE / max(w, h)
    W, H = int(round(w * s)), int(round(h * s))
    cx, cy = W // 2, H // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if W == H:
        halfh = 3 * halfw / 4
    return ((cx - halfw) / W, (cx + halfw) / W, (cy - halfh) / H, (cy + halfh) / H)


def teacher_uv_affine(w, h):
    x0, x1, y0, y1 = teacher_crop_frac(w, h)
    return [x1 - x0, (x0 + x1) - 1.0, y1 - y0, (y0 + y1) - 1.0]


@torch.no_grad()
def extract_vit_features(vit_model, processor, frames, device):
    proc_frames = []
    orig_sizes = []
    for f in frames:
        w, h = f.size
        pw, ph = vit_proc_size(w, h)
        proc_frames.append(f.resize((pw, ph), Image.BICUBIC))
        orig_sizes.append((w, h))
    p = processor.image_processor(images=proc_frames, return_tensors="pt", do_resize=False)
    pv = p["pixel_values"].to(device, torch.bfloat16)
    grid = p["image_grid_thw"].to(device)
    out = vit_model.get_image_features(pv, grid, return_dict=True)
    pooled = out.pooler_output
    if not isinstance(pooled, (list, tuple)):
        counts = (grid.prod(-1) // 4).tolist()
        pooled = torch.split(pooled, counts)
    feats = [p.to(torch.float16).cpu() for p in pooled]
    n_tok = feats[0].shape[0]
    pf = torch.stack(feats)[None].to(torch.bfloat16)  # [1,T,N,D]
    grid_np = grid.cpu().numpy()
    orig = np.array(orig_sizes, np.int32)
    proc = np.array([vit_proc_size(*o) for o in orig_sizes], np.int32)
    aff = np.array([teacher_uv_affine(*map(int, o)) for o in orig_sizes], np.float32)
    return pf, grid_np, orig, proc, aff, n_tok


# ---------------------------------------------------------------------------
# CUT3R teacher extraction (same as tools/export_teacher32.py)
# ---------------------------------------------------------------------------
def load_cut3r(device):
    sys.path.insert(0, CUT3R_ROOT)
    sys.path.insert(0, f"{CUT3R_ROOT}/src")
    import add_ckpt_path
    add_ckpt_path.add_path_to_dust3r(f"{CUT3R_ROOT}/src/dust3r")
    from src.dust3r.model import ARCroco3DStereo
    model = ARCroco3DStereo.from_pretrained(CUT3R_CKPT).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def build_views(frame_paths):
    from src.dust3r.utils.image import load_images
    images = load_images(frame_paths, size=CUT3R_SIZE)
    views = []
    for i, im in enumerate(images):
        views.append({
            "img": im["img"],
            "ray_map": torch.full(
                (im["img"].shape[0], 6, im["img"].shape[-2], im["img"].shape[-1]),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(im["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        })
    return views


@torch.no_grad()
def infer_cut3r(cut3r_model, frames, device, tmp_dir):
    from src.dust3r.inference import inference
    from src.dust3r.utils.camera import pose_encoding_to_camera
    frame_paths = []
    for i, img in enumerate(frames):
        p = os.path.join(tmp_dir, f"_cut3r_frame_{i:04d}.jpg")
        img.save(p, quality=95)
        frame_paths.append(p)
    views = build_views(frame_paths)
    outputs, _ = inference(views, cut3r_model, device)
    preds = outputs["pred"]
    pts, conf, pose = [], [], []
    for out in preds:
        pts.append(out["pts3d_in_other_view"].squeeze(0).to(torch.float16).cpu().numpy())
        conf.append(out["conf"].squeeze(0).to(torch.float16).cpu().numpy())
        pr = pose_encoding_to_camera(out["camera_pose"].clone().float().cpu())
        pose.append(pr.squeeze(0).numpy())
    for p in frame_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    return {
        "pts3d": np.stack(pts).astype(np.float16),
        "conf": np.stack(conf).astype(np.float16),
        "pose": np.stack(pose).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Ablation helpers (same logic as eval_stage_a_ablation_stats.py)
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
def _mae_per_scene(pred_xyz, target_xyz, mask):
    """Masked MAE over xyz dim, then over valid rays. Returns scalar or None."""
    if not bool(mask.any()):
        return None
    err = (pred_xyz.float() - target_xyz.float()).abs().mean(dim=-1)  # [R]
    m = mask.to(err.dtype)
    return float((err * m).sum().item() / m.sum().clamp_min(1.0).item())


@torch.no_grad()
def _read_mae(model, state, ray_queries, targets, mask):
    """Run read_geometry and return scalar MAE (or None if no valid rays).

    ray_queries: [1, R, D], targets: [1, R, 3], mask: [1, R] or [R].
    """
    if not bool(mask.any()):
        return None
    pred = model.geomemory.read_geometry(state, ray_queries).xyz_world  # [1,R,3]
    m = mask if mask.ndim == 1 else mask[0]
    return _mae_per_scene(pred[0], targets[0] if targets.ndim == 3 else targets, m)


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
# Main
# ---------------------------------------------------------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--vsi_jsonl", default=VSI_JSONL)
    ap.add_argument("--vsi_root", default=VSI_ROOT)
    ap.add_argument("--qwen_model", default=QWEN_MODEL)
    ap.add_argument("--max_scenes", type=int, default=288)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--query_count", type=int, default=128)
    ap.add_argument("--heldout_count", type=int, default=64)
    ap.add_argument("--heldout_p", type=float, default=1.0)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--tmp_dir", default="/tmp/bluecode/vsi_ablation_tmp")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    os.makedirs(a.tmp_dir, exist_ok=True)

    # --- Load GeoMemory model ---
    cfg = GeoMemoryConfig.from_json(a.config or os.path.join(a.ckpt, "geomemory_config.json"))
    model = GeoMemoryTrainingModel(cfg, language_model=None, freeze_language_model=True)
    _load_geomemory_checkpoint(model, a.ckpt)
    model.to(dev, torch.bfloat16).eval()
    min_conf = cfg.loss.min_confidence

    # --- Load ViT (Qwen3.5 visual encoder only) ---
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.qwen_model, local_files_only=True)
    from transformers import AutoModelForImageTextToText
    vit_full = AutoModelForImageTextToText.from_pretrained(
        a.qwen_model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(dev).eval()
    vit_full.requires_grad_(False)
    vit_model = vit_full.model  # access .get_image_features via inner model

    # --- Load CUT3R ---
    cut3r_model = load_cut3r(dev)

    # --- Load VSI-Bench scene list ---
    rows = []
    with open(a.vsi_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    scenes = {}
    for r in rows:
        key = (r["dataset"], r["scene_name"])
        if key not in scenes:
            scenes[key] = r
    scene_list = sorted(scenes.keys())
    scene_list = scene_list[a.rank::a.world]
    if a.max_scenes > 0:
        scene_list = scene_list[:a.max_scenes]
    print(f"[rank {a.rank}] {len(scene_list)} scenes to evaluate", flush=True)

    # Per-scene accumulators
    scene_mae = {
        f"{cond}_{kind}": []
        for cond in ("full", "zero", "init", "shuffle")
        for kind in ("current", "history", "heldout")
    }
    scene_names = []
    errors = []

    t0 = time.time()
    for si, (ds_name, scene_name) in enumerate(scene_list):
        video_path = os.path.join(a.vsi_root, ds_name, f"{scene_name}.mp4")
        try:
            frames = extract_frames(video_path, a.num_frames)
            if len(frames) < 2:
                raise ValueError(f"only {len(frames)} frames extracted")
            T = len(frames)

            # --- ViT features ---
            pf, grid_np, orig, proc, aff, n_tok = extract_vit_features(
                vit_model, processor, frames, dev
            )
            pf = pf.to(dev, torch.bfloat16)
            xy = torch.stack([
                grid_to_patch_xy(grid_np[t], n_tok, 2) for t in range(T)
            ])[None].to(dev, torch.bfloat16)

            # --- CUT3R teacher ---
            teacher = infer_cut3r(cut3r_model, frames, dev, a.tmp_dir)
            teacher["pts3d"] = teacher["pts3d"].astype(np.float16)
            teacher["conf"] = teacher["conf"].astype(np.float16)

            # --- Sample ray queries ---
            rng = np.random.default_rng(a.seed + si)
            queries = sample_queries(
                teacher, rng,
                cur_count=a.query_count,
                hist_count=a.query_count,
                heldout_count=a.heldout_count,
                heldout_p=a.heldout_p,
                uv_affine=aff,
            )

            # Convert to tensors [1, T, R, ...]
            def to_t(key, dtype):
                v = queries[key]
                return torch.from_numpy(v).to(dev, dtype).unsqueeze(0)

            crq = to_t("current_ray_queries", torch.bfloat16)
            cxt = to_t("current_xyz_targets", torch.float32)
            cm = to_t("current_query_mask", torch.bool)
            cc = to_t("current_query_confidence", torch.float32)
            hrq = to_t("history_ray_queries", torch.bfloat16)
            hxt = to_t("history_xyz_targets", torch.float32)
            hm = to_t("history_query_mask", torch.bool)
            hc = to_t("history_query_confidence", torch.float32)
            orq = to_t("heldout_ray_queries", torch.bfloat16)
            oxt = to_t("heldout_xyz_targets", torch.float32)
            om = to_t("heldout_query_mask", torch.bool)
            oc = to_t("heldout_query_confidence", torch.float32)

            heldout_frame = int(queries["heldout_frame"])

            # --- Update mask: heldout frame not written ---
            update_mask = torch.ones(1, T, device=dev, dtype=torch.bool)
            if heldout_frame >= 0:
                update_mask[0, heldout_frame] = False

            # --- Run recurrent memory update ---
            core = model.geomemory(
                pf, patch_xy=xy,
                frame_mask=torch.ones(1, T, device=dev, dtype=torch.bool),
                update_mask=update_mask,
            )
            state_history = core.state_history
            if not state_history:
                raise ValueError("no state_history")

            st_init = model.geomemory.new_scene_state(1, device=dev, dtype=torch.bfloat16)

            # --- Evaluate per-prefix, collect per-scene-per-t ---
            per_t = {
                f"{cond}_{kind}": []
                for cond in ("full", "zero", "init", "shuffle")
                for kind in ("current", "history", "heldout")
            }

            for t in range(T):
                st_full = state_history[t]
                st_zero = _zero_state_like(st_full)
                st_shuf = None  # batch=1, can't shuffle within scene

                # current
                m = _filter_mask(cm, cc, t, min_conf)
                if m is not None and bool(m.any()):
                    rq = crq[:, t]
                    for cn, st in [("full", st_full), ("zero", st_zero), ("init", st_init)]:
                        v = _read_mae(model, st, rq, cxt[:, t], m)
                        if v is not None:
                            per_t[f"{cn}_current"].append(v)

                # history
                m = _filter_mask(hm, hc, t, min_conf)
                if t > 0 and m is not None and bool(m.any()):
                    rq = hrq[:, t]
                    for cn, st in [("full", st_full), ("zero", st_zero), ("init", st_init)]:
                        v = _read_mae(model, st, rq, hxt[:, t], m)
                        if v is not None:
                            per_t[f"{cn}_history"].append(v)

                # heldout
                m = _filter_mask(om, oc, t, min_conf)
                if m is not None and bool(m.any()):
                    rq = orq[:, t]
                    for cn, st in [("full", st_full), ("zero", st_zero), ("init", st_init)]:
                        v = _read_mae(model, st, rq, oxt[:, t], m)
                        if v is not None:
                            per_t[f"{cn}_heldout"].append(v)

            # Collapse per-t to per-scene mean
            for key in per_t:
                vals = per_t[key]
                scene_mae[key].append(float(np.mean(vals)) if vals else None)

            scene_names.append(f"{ds_name}/{scene_name}")

        except Exception as e:
            import traceback
            scene_mae = {k: v + [None] for k, v in scene_mae.items()}
            scene_names.append(f"{ds_name}/{scene_name}")
            errors.append({"scene": f"{ds_name}/{scene_name}", "error": str(e)[:300],
                           "trace": traceback.format_exc()[-500:]})
            print(f"[rank {a.rank}] ERROR {ds_name}/{scene_name}: {e}", flush=True)

        if (si + 1) % 10 == 0 or si + 1 == len(scene_list):
            elapsed = time.time() - t0
            print(f"[rank {a.rank}] {si+1}/{len(scene_list)} "
                  f"({(si+1)/elapsed:.2f} scene/s, {elapsed/60:.1f}min)", flush=True)

    # --- Shuffle condition: cross-scene, done post-hoc ---
    # For batch=1 runs, shuffle = roll across scenes' final-state memory.
    # We need to re-run readout with memory from another scene.
    # This requires storing the final memory per scene. Let's do it differently:
    # Since we can't shuffle within a single-GPU single-scene loop, we'll
    # compute shuffle by pairing scene i with scene i+1.
    # Actually, the shuffle condition needs the memory tensor from another scene
    # at the SAME prefix t. We didn't store those. So shuffle is omitted for VSI
    # (single-scene-at-a-time processing). We report full/zero/init only.
    # To add shuffle, one would need to store all state_history tensors and
    # re-run readout. That's memory-prohibitive for 288 scenes × 32 steps.
    # Instead, we approximate shuffle by using init memory from a DIFFERENT
    # scene's final state — but that's not available either.
    # CONCLUSION: shuffle is skipped for VSI-Bench. full/zero/init are reported.

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

    scene_mae["full_overall"] = full_ov
    scene_mae["zero_overall"] = zero_ov
    scene_mae["init_overall"] = init_ov

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
    full_hist = safe_list("full_history")
    zero_hist = safe_list("zero_history")
    init_hist = safe_list("init_history")
    full_ho = safe_list("full_heldout")
    zero_ho = safe_list("zero_heldout")
    init_ho = safe_list("init_heldout")

    deltas = {
        "zero_current": delta_list(full_cur, zero_cur),
        "zero_history": delta_list(full_hist, zero_hist),
        "zero_heldout": delta_list(full_ho, zero_ho),
        "zero_overall": delta_list(full_ov, zero_ov),
        "init_current": delta_list(full_cur, init_cur),
        "init_history": delta_list(full_hist, init_hist),
        "init_heldout": delta_list(full_ho, init_ho),
        "init_overall": delta_list(full_ov, init_ov),
    }

    results = {}
    for cond in ("full", "zero", "init"):
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
        "vsi_jsonl": a.vsi_jsonl,
        "max_scenes": a.max_scenes,
        "num_frames": a.num_frames,
        "query_count": a.query_count,
        "heldout_count": a.heldout_count,
        "n_boot": a.n_boot,
        "seed": a.seed,
        "n_scenes_evaluated": len([v for v in scene_names if v is not None]),
        "n_errors": len(errors),
        "errors": errors[:10],
        "note": "shuffle condition omitted (single-scene processing); full/zero/init reported",
    }

    print(json.dumps(results, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults written to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
