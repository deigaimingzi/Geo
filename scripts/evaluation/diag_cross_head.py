"""Why is Cross (world) XYZ not improving?  Held-out UV, fixed val scenes, per checkpoint.
 A) cross_mae          world head vs CUT3R world XYZ
 B) self->world(stuPose)  student self XYZ moved by the STUDENT pose head  (does Self+Pose already give Cross?)
 C) self->world(gtPose)   student self XYZ moved by the TEACHER pose       (upper bound for that route)
 D) mean_pred_mae      predict the per-scene mean target                    (what 'collapse to the mean' scores)
 E) pred_std/tgt_std   spread of the world-head output vs the target spread (collapse => ratio ~ 0)
 F) corr               per-axis correlation world-head vs target           (structure learned beyond the mean)
"""
import argparse, json, os, sys
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from geomemory.config import GeoMemoryConfig
from geomemory.data.collator import GeoMemoryCollator
from geomemory.data.dataset import GeoMemoryDataset
from geomemory.train import _load_geomemory_checkpoint
from geomemory.training_model import GeoMemoryTrainingModel


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", default="")
    ap.add_argument("--manifest", required=True); ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    cfg = GeoMemoryConfig.from_json(a.config or os.path.join(a.ckpt, "geomemory_config.json"))
    model = GeoMemoryTrainingModel(cfg, language_model=None, freeze_language_model=True)
    _load_geomemory_checkpoint(model, a.ckpt); model.to(dev, torch.bfloat16).eval()
    ds = GeoMemoryDataset(a.manifest, stage="geometry", dynamic_rays=True, uv_holdout="only", seed=1234,
                          max_samples=a.max_samples, query_count=256, relation_pairs=64)
    col = GeoMemoryCollator(tokenizer=None, model_max_length=8192)
    acc = {}
    def add(k, v): acc.setdefault(k, []).append(float(v))
    for i in range(len(ds)):
        b = col([ds[i]]); b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        T = b["patch_features"].shape[1]
        core = model.geomemory(b["patch_features"].to(torch.bfloat16), patch_xy=b["patch_xy"].to(torch.bfloat16),
                               current_ray_queries=b["current_ray_queries"], history_ray_queries=b["history_ray_queries"])
        for t in range(1, T):
            st = core.state_history[t]
            rq = b["current_ray_queries"][:, t].to(torch.bfloat16)
            m = b["current_query_mask"][:, t] & (b["current_query_confidence"][:, t] > 0.05)
            if m.sum() < 8: continue
            p = model.geomemory.read_geometry(st, rq)
            xw = b["current_xyz_targets"][:, t][m]; pw = p.xyz_world.float()[m]; ps = p.xyz_self.float()[m]
            add("A_cross_mae", (pw - xw).abs().mean(-1).mean())
            add("D_mean_pred_mae", (xw - xw.mean(0)).abs().mean(-1).mean())
            add("E_std_ratio", pw.std(0).mean() / xw.std(0).mean().clamp_min(1e-6))
            c = [torch.corrcoef(torch.stack((pw[:, k], xw[:, k])))[0, 1] for k in range(3)]
            add("F_corr", torch.stack([x.nan_to_num(0.0) for x in c]).mean())
            gt = b["pose_targets"][:, t].float()[0]
            stu = core.poses[t].transform_c2w.float()[0] if core.poses else None
            def tf(T_, x): return x @ T_[:3, :3].T + T_[:3, 3]
            add("C_self_gtpose_mae", (tf(gt, ps) - xw).abs().mean(-1).mean())
            if stu is not None:
                add("B_self_stupose_mae", (tf(stu, ps) - xw).abs().mean(-1).mean())
                add("pose_trans_err", (stu[:3, 3] - gt[:3, 3]).norm())
                add("pose_trans_scale", gt[:3, 3].norm())
                cosr = ((stu[:3, :3].T @ gt[:3, :3]).diagonal().sum() - 1) / 2
                add("pose_rot_err_deg", torch.rad2deg(torch.acos(cosr.clamp(-1 + 1e-6, 1 - 1e-6))))
    res = {k: float(np.mean(v)) for k, v in acc.items()}
    print(json.dumps(res, indent=1))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
