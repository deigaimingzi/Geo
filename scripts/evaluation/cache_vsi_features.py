"""Cache ViT features + CUT3R teacher maps for VSI-Bench test scenes.

Extracts 32 frames per scene, runs Qwen3.5 ViT and CUT3R-512, saves a single
npz per scene containing:
  patch_features [T,N,2560] fp16
  grid_thw [T,3]
  teacher_uv_affine [T,4]
  pts3d [T,H,W,3] fp16
  conf [T,H,W] fp16
  pose [T,4,4] fp32

Then run the ablation script on any checkpoint — no ViT/CUT3R needed.

Usage (8 GPU):
  for r in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$r python scripts/evaluation/cache_vsi_features.py \
      --rank $r --world 8 &
  done; wait
"""
import argparse
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(
    0,
    "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/scripts/evaluation",
)
from eval_vsi_mra import ss_sample_frame_indices  # noqa: E402

from geomemory.position import grid_to_patch_xy  # noqa: E402 (for n_tok calc only)

VSI_BASE = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/benchmark/VSI-Bench"
VSI_JSONL = f"{VSI_BASE}/raw/test.jsonl"
VSI_ROOT = f"{VSI_BASE}/videos"

CUT3R_ROOT = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/depth-model/CUT3R"
CUT3R_CKPT = f"{CUT3R_ROOT}/src/cut3r_512_dpt_4_64.pth"
CUT3R_SIZE = 512

QWEN_MODEL = "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B"

LONG_EDGE = 512
ALIGN = 32
NUM_FRAMES = 32


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_frames(video_path, num_frames=NUM_FRAMES):
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
# ViT feature extraction
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
    pf = torch.stack(feats).numpy()  # [T,N,D]
    grid_np = grid.cpu().numpy()
    orig = np.array(orig_sizes, np.int32)
    proc = np.array([vit_proc_size(*o) for o in orig_sizes], np.int32)
    aff = np.array([teacher_uv_affine(*map(int, o)) for o in orig_sizes], np.float32)
    return pf, grid_np, orig, proc, aff, n_tok


# ---------------------------------------------------------------------------
# CUT3R teacher extraction
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
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsi_jsonl", default=VSI_JSONL)
    ap.add_argument("--vsi_root", default=VSI_ROOT)
    ap.add_argument("--qwen_model", default=QWEN_MODEL)
    ap.add_argument("--out_dir", default="/data/juicefs_sharing_data/11188211/VIT-feather/v8_vit512/features/vsi_bench_test")
    ap.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--tmp_dir", default="/tmp/bluecode/vsi_cache_tmp")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    os.makedirs(a.tmp_dir, exist_ok=True)
    out_dir = os.path.join(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- Load ViT ---
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(a.qwen_model, local_files_only=True)
    vit_full = AutoModelForImageTextToText.from_pretrained(
        a.qwen_model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(dev).eval()
    vit_full.requires_grad_(False)
    vit_model = vit_full.model

    # --- Load CUT3R ---
    cut3r_model = load_cut3r(dev)

    # --- Scene list ---
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
    print(f"[rank {a.rank}] {len(scene_list)} scenes", flush=True)

    idx_path = os.path.join(out_dir, f"scene_index.rank{a.rank}.jsonl")
    idx_fh = open(idx_path, "w", encoding="utf-8")

    t0, done, fail = time.time(), 0, 0
    for si, (ds_name, scene_name) in enumerate(scene_list):
        scene_key = f"{ds_name}_{scene_name}"
        out_path = os.path.join(out_dir, f"{scene_key}.npz")
        if os.path.exists(out_path):
            done += 1
            idx_fh.write(json.dumps({"scene_key": scene_key, "npz": out_path, "cached": True}) + "\n")
            idx_fh.flush()
            continue
        video_path = os.path.join(a.vsi_root, ds_name, f"{scene_name}.mp4")
        try:
            frames = extract_frames(video_path, a.num_frames)
            if len(frames) < 2:
                raise ValueError(f"only {len(frames)} frames")
            T = len(frames)

            pf, grid_np, orig, proc, aff, n_tok = extract_vit_features(
                vit_model, processor, frames, dev
            )
            teacher = infer_cut3r(cut3r_model, frames, dev, a.tmp_dir)

            np.savez(out_path,
                     patch_features=pf,
                     grid_thw=grid_np,
                     orig_size=orig,
                     proc_size=proc,
                     teacher_uv_affine=aff,
                     pts3d=teacher["pts3d"],
                     conf=teacher["conf"],
                     pose=teacher["pose"],
                     scene_key=np.array(scene_key),
                     scene=np.array(scene_key))
            idx_fh.write(json.dumps({
                "scene_key": scene_key, "npz": out_path,
                "num_frames": T, "num_tokens_per_frame": n_tok,
            }) + "\n")
            idx_fh.flush()
            done += 1
        except Exception as e:
            fail += 1
            idx_fh.write(json.dumps({"scene_key": scene_key, "error": str(e)[:300],
                                     "trace": traceback.format_exc()[-500:]}) + "\n")
            idx_fh.flush()
            print(f"[rank {a.rank}] ERROR {scene_key}: {e}", flush=True)

        if (si + 1) % 10 == 0 or si + 1 == len(scene_list):
            elapsed = time.time() - t0
            print(f"[rank {a.rank}] {si+1}/{len(scene_list)} done={done} fail={fail} "
                  f"({(si+1)/elapsed:.2f}/s {elapsed/60:.1f}min)", flush=True)

    idx_fh.close()
    print(f"[rank {a.rank}] finished done={done} fail={fail} in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
