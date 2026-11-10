"""Unseen-UV geometry evaluation.  Queries come ONLY from the held-out pixel lattice
(u%8==3 & v%8==3) that Stage A/B training never samples.  Reports, per prefix length, in CUT3R units:
  self_xyz_mae   camera-frame XYZ of the queried pixel (self head)
  cross_xyz_mae  frame-0 XYZ (world head)
  pair_dist_err  |pred - gt| distance between a remembered point and a current-frame point
each next to a trivial reference (constant = per-scene mean target / mean distance)."""
import argparse, json, os, sys
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from geomemory.config import GeoMemoryConfig  # noqa: E402
from geomemory.data.collator import GeoMemoryCollator  # noqa: E402
from geomemory.data.dataset import GeoMemoryDataset  # noqa: E402
from geomemory.train import _load_geomemory_checkpoint  # noqa: E402
from geomemory.training_model import GeoMemoryTrainingModel  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--max_samples", type=int, default=64); ap.add_argument("--config", default=""); ap.add_argument("--uv", default="only", choices=["only","exclude"]); ap.add_argument("--init_seed", type=int, default=-1); ap.add_argument("--query_count", type=int, default=256)
    ap.add_argument("--relation_pairs", type=int, default=64); ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    cfg = GeoMemoryConfig.from_json(a.config or os.path.join(a.ckpt, "geomemory_config.json"))
    model = GeoMemoryTrainingModel(cfg, language_model=None, freeze_language_model=True)
    if a.init_seed >= 0:      # step-0 reference: a freshly initialised model, no checkpoint
        torch.manual_seed(a.init_seed); model = GeoMemoryTrainingModel(cfg, language_model=None, freeze_language_model=True)
    else:
        _load_geomemory_checkpoint(model, a.ckpt)
    model.to(dev, torch.bfloat16).eval()
    ds = GeoMemoryDataset(a.manifest, stage="geometry", dynamic_rays=True, uv_holdout=a.uv, seed=1234,
                          max_samples=a.max_samples, query_count=a.query_count, relation_pairs=a.relation_pairs)
    col = GeoMemoryCollator(tokenizer=None, model_max_length=8192)
    acc = {}
    for i in range(len(ds)):
        b = col([ds[i]])
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        T = b["patch_features"].shape[1]
        core = model.geomemory(
            b["patch_features"].to(torch.bfloat16), patch_xy=b["patch_xy"].to(torch.bfloat16),
            current_ray_queries=b["current_ray_queries"], history_ray_queries=b["history_ray_queries"])
        # read every prefix t with the state that has seen frames 0..t
        for t in range(T):
            st = core.state_history[t] if hasattr(core, "state_history") and core.state_history else None
            if st is None:
                continue
            for kind in ("current", "history"):
                if kind == "history" and t == 0:
                    continue
                rq = b[f"{kind}_ray_queries"][:, t]
                p = model.geomemory.read_geometry(st, rq.to(torch.bfloat16))
                xc, xs = b[f"{kind}_xyz_targets"][:, t], b[f"{kind}_self_xyz_targets"][:, t]
                mc = b[f"{kind}_query_mask"][:, t] & (b[f"{kind}_query_confidence"][:, t] > 0.05)
                ms = b[f"{kind}_query_mask"][:, t] & (b[f"{kind}_self_confidence"][:, t] > 0.05)
                key = f"{kind}"
                if mc.any():
                    acc.setdefault(f"cross_mae_{key}", []).append((p.xyz_world.float() - xc)[mc].abs().mean(-1).mean().item())
                    acc.setdefault(f"cross_ref_{key}", []).append((xc[mc] - xc[mc].mean(0)).abs().mean(-1).mean().item())
                if ms.any():
                    acc.setdefault(f"self_mae_{key}", []).append((p.xyz_self.float() - xs)[ms].abs().mean(-1).mean().item())
                    acc.setdefault(f"self_ref_{key}", []).append((xs[ms] - xs[ms].mean(0)).abs().mean(-1).mean().item())
                    acc.setdefault(f"self_z_rel_{key}", []).append(((p.xyz_self[..., 2].float() - xs[..., 2]).abs() / xs[..., 2].clamp_min(1e-3))[ms].mean().item())
            if t > 0:
                ph = model.geomemory.read_geometry(st, b["history_ray_queries"][:, t].to(torch.bfloat16)).xyz_world.float()
                pc = model.geomemory.read_geometry(st, b["current_ray_queries"][:, t].to(torch.bfloat16)).xyz_world.float()
                pr, rm, dt = b["relation_pairs"][:, t], b["relation_mask"][:, t], b["relation_distance_targets"][:, t]
                if rm.any():
                    pa = torch.gather(ph, 1, pr[..., 0:1].expand(-1, -1, 3)); pb = torch.gather(pc, 1, pr[..., 1:2].expand(-1, -1, 3))
                    d = (pa - pb).norm(dim=-1)
                    acc.setdefault("pair_dist_err", []).append((d - dt).abs()[rm].mean().item())
                    acc.setdefault("pair_dist_ref", []).append((dt[rm] - dt[rm].mean()).abs().mean().item())
                    acc.setdefault("pair_dist_rel", []).append(((d - dt).abs() / dt.clamp_min(1e-3))[rm].mean().item())
    res = {k: float(np.mean(v)) for k, v in acc.items()}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
