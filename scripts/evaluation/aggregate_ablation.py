#!/usr/bin/env python3
"""Aggregate Stage B ablation results: per-condition accuracy/MRA + NLL, and
Delta_State = NLL_wrong - NLL_real, Delta_History = NLL_short - NLL_real,
reported per question type and overall."""
import argparse, json, os
from collections import defaultdict
import numpy as np


def load_records(out_dir, world):
    records = []
    for r in range(world):
        p = os.path.join(out_dir, f"ablation_rank{r}.json")
        if os.path.exists(p):
            records.extend(json.load(open(p)))
    return records


def metric_for(qt):
    return "accuracy" if qt in ("object_rel_direction_easy", "object_rel_direction_medium",
                                "object_rel_direction_hard", "object_rel_direction",
                                "object_rel_distance", "route_planning",
                                "obj_appearance_order") else "MRA:.5:.95:.05"


def aggregate(records):
    """Return dict: condition -> {overall_acc, overall_nll, per_type: {qt: {acc, nll, n}}}"""
    conds = sorted({c for r in records for c in r["conditions"]})
    out = {}
    for cond in conds:
        acc_by_type = defaultdict(list)
        nll_by_type = defaultdict(list)
        for r in records:
            qt = r["question_type"]
            c = r["conditions"][cond]
            sc = c.get("score", {})
            m = metric_for(qt)
            if m in sc and sc[m] is not None:
                acc_by_type[qt].append(float(sc[m]))
            nll = c.get("nll")
            if nll is not None and np.isfinite(nll):
                nll_by_type[qt].append(float(nll))
        per_type = {}
        all_acc, all_nll = [], []
        for qt in sorted(set(list(acc_by_type.keys()) + list(nll_by_type.keys()))):
            accs = acc_by_type.get(qt, [])
            nlls = nll_by_type.get(qt, [])
            per_type[qt] = {
                "acc": float(np.mean(accs)) if accs else None,
                "nll": float(np.mean(nlls)) if nlls else None,
                "n": len(accs) if accs else len(nlls),
            }
            all_acc.extend(accs)
            all_nll.extend(nlls)
        out[cond] = {
            "overall_acc": float(np.mean(all_acc)) if all_acc else None,
            "overall_nll": float(np.mean(all_nll)) if all_nll else None,
            "n": len(all_acc) if all_acc else len(all_nll),
            "per_type": per_type,
        }
    return out


def deltas(agg):
    """Compute Delta_State and Delta_History per type and overall."""
    def per_type_delta(a_key, b_key):
        res = {}
        a = agg.get(a_key, {}).get("per_type", {})
        b = agg.get(b_key, {}).get("per_type", {})
        for qt in sorted(set(list(a.keys()) + list(b.keys()))):
            an = a.get(qt, {}).get("nll")
            bn = b.get(qt, {}).get("nll")
            if an is not None and bn is not None:
                res[qt] = an - bn
        return res

    res = {}
    if "wrong" in agg and "real" in agg:
        res["delta_state"] = {
            "overall": agg["wrong"]["overall_nll"] - agg["real"]["overall_nll"]
                        if agg["wrong"]["overall_nll"] and agg["real"]["overall_nll"] else None,
            "per_type": per_type_delta("wrong", "real"),
        }
    if "short" in agg and "real" in agg:
        res["delta_history"] = {
            "overall": agg["short"]["overall_nll"] - agg["real"]["overall_nll"]
                        if agg["short"]["overall_nll"] and agg["real"]["overall_nll"] else None,
            "per_type": per_type_delta("short", "real"),
        }
    if "initial" in agg and "real" in agg:
        res["delta_initial"] = {
            "overall": agg["initial"]["overall_nll"] - agg["real"]["overall_nll"]
                        if agg["initial"]["overall_nll"] and agg["real"]["overall_nll"] else None,
            "per_type": per_type_delta("initial", "real"),
        }
    if "fusion_off" in agg and "real" in agg:
        res["delta_fusion_off"] = {
            "overall": agg["fusion_off"]["overall_nll"] - agg["real"]["overall_nll"]
                        if agg["fusion_off"]["overall_nll"] and agg["real"]["overall_nll"] else None,
            "per_type": per_type_delta("fusion_off", "real"),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--world", type=int, default=8)
    args = ap.parse_args()
    records = load_records(args.out_dir, args.world)
    print(f"Loaded {len(records)} records")
    if not records:
        return
    agg = aggregate(records)
    dl = deltas(agg)
    result = {"conditions": agg, "deltas": dl, "num_records": len(records)}
    out_path = os.path.join(args.out_dir, "ablation_summary.json")
    json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
