"""A1 evaluation on the fixed val scenes (never trained on), per Qwen token, CUT3R units.

For every frame t (t>=1 prefix state) and every reliable token we report:
  patch head (with memory)      xyz MAE, relative Z error, same-frame pair-distance error, scale MAE
  patch head (memory OFF)       same, local features only  -> what the frozen Qwen ViT tokens alone give (P1)
  old UV Self head at centres   the pre-A1 head queried at the token centre u,v                (P2)
  constants                     per-scene-mean xyz, per-scene-mean pair distance
Everything is also split by grid shape.
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from geomemory.config import GeoMemoryConfig
from geomemory.data.collator import GeoMemoryCollator
from geomemory.data.dataset import GeoMemoryDataset
from geomemory.train import _load_geomemory_checkpoint
from geomemory.training_model import GeoMemoryTrainingModel
from geomemory.position import merged_grid_shape


def pair_err(xyz_p, xyz_t, valid, gen, n=256):
    idx = valid.nonzero().squeeze(-1)
    if len(idx) < 4:
        return None
    a = idx[torch.randint(len(idx), (n,), generator=gen, device=idx.device)]
    b = idx[torch.randint(len(idx), (n,), generator=gen, device=idx.device)]
    m = a != b
    dt = (xyz_t[a] - xyz_t[b]).norm(dim=-1)[m]
    dp = (xyz_p[a] - xyz_p[b]).norm(dim=-1)[m]
    dc = (dt - dt.mean()).abs()
    return (dp - dt).abs().mean().item(), dc.mean().item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", default="")
    ap.add_argument("--manifest", required=True); ap.add_argument("--max_samples", type=int, default=120)
    ap.add_argument("--init_seed", type=int, default=-1); ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    cfg = GeoMemoryConfig.from_json(a.config or os.path.join(a.ckpt, "geomemory_config.json"))
    model = GeoMemoryTrainingModel(cfg, language_model=None, freeze_language_model=True)
    if a.init_seed >= 0:
        torch.manual_seed(a.init_seed); model = GeoMemoryTrainingModel(cfg, language_model=None, freeze_language_model=True)
    else:
        _load_geomemory_checkpoint(model, a.ckpt)
    model.to(dev, torch.bfloat16).eval()
    core = model.geomemory
    ds = GeoMemoryDataset(a.manifest, stage="geometry", dynamic_rays=True, patch_targets=True, uv_holdout="only",
                          seed=1234, max_samples=a.max_samples, query_count=8, relation_pairs=4)
    col = GeoMemoryCollator(tokenizer=None, model_max_length=8192)
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    acc = defaultdict(list)
    for i in range(len(ds)):
        it = ds[i]; b = col([it]); b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        grid = tuple(merged_grid_shape(it["grid_thw"][0].tolist(), 2)); gname = f"{grid[0]}x{grid[1]}"
        out = core(b["patch_features"].to(torch.bfloat16), patch_xy=b["patch_xy"].to(torch.bfloat16))
        T = b["patch_features"].shape[1]
        for t in range(1, T):
            st = out.state_history[t]
            tv = torch.full((1,), t, device=dev, dtype=torch.long)
            pf, pxy = b["patch_features"][:, t].to(torch.bfloat16), b["patch_xy"][:, t].to(torch.bfloat16)
            tx, ts, tc, tvld = b["pt_xyz"][0, t], b["pt_scale"][0, t], b["pt_conf"][0, t], b["pt_valid"][0, t]
            valid = tvld & (tc > cfg.loss.min_confidence)
            if valid.sum() < 8:
                continue
            res = {}
            for tag, use_mem in (("mem", True), ("nomem", False)):
                ph = core.patch_head(pf, pxy, tv, st, None, use_memory=use_mem)
                res[tag] = ph
            # old UV self head queried at token centres: rq = (u,v,t/16,1,1,...)
            xy = b["patch_xy"][0, t]
            rq = torch.zeros(1, xy.shape[0], cfg.ray_query_dim, device=dev, dtype=torch.bfloat16)
            rq[0, :, 0], rq[0, :, 1] = xy[:, 0], xy[:, 1]
            rq[0, :, 2], rq[0, :, 3], rq[0, :, 4] = t / 16.0, 1.0, 1.0
            old = core.read_geometry(st, rq).xyz_self.float()[0]
            const = (tx[valid]).mean(0, keepdim=True).expand_as(tx)
            for tag, xyzp in (("A1_mem", res["mem"]["xyz"].float()[0]), ("A1_nomem", res["nomem"]["xyz"].float()[0]),
                              ("oldUV_self", old), ("const", const)):
                for key in ("all", gname):
                    acc[f"{key}/{tag}/xyz_mae"].append((xyzp - tx)[valid].abs().mean().item())
                    acc[f"{key}/{tag}/z_rel"].append(((xyzp[:, 2] - tx[:, 2]).abs() / tx[:, 2].clamp_min(1e-3))[valid].mean().item())
                    pe = pair_err(xyzp, tx, valid, gen)
                    if pe:
                        acc[f"{key}/{tag}/pair_err"].append(pe[0]); 
                        if tag == "const": acc[f"{key}/const/pair_ref"].append(pe[1])
            for key in ("all", gname):
                acc[f"{key}/A1_mem/scale_mae"].append((res["mem"]["scale"].float()[0] - ts).abs()[valid].mean().item())
                acc[f"{key}/zero_scale/scale_mae"].append(ts[valid].abs().mean().item())
    res = {k: float(np.mean(v)) for k, v in acc.items()}
    res["n_frames/all"] = len(acc["all/A1_mem/xyz_mae"])
    print(json.dumps({k: round(v, 4) for k, v in res.items() if k.startswith("all/") or k.startswith("n_")}, indent=1))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
