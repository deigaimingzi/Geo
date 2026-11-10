#!/usr/bin/env python3
"""Re-score saved OST predictions (no model run) with the current scoring rules."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv, d = sys.argv[:1], sys.argv[1]
from eval_ost_geomemory import score, aggregate
recs = [x for f in sorted(glob.glob(f"{d}/predictions_rank*.json")) for x in json.load(open(f))]
for r in recs:
    r["score"] = score(r)
m = aggregate(recs)
json.dump(m, open(f"{d}/metrics.json", "w"), indent=2)
print(json.dumps(m, indent=2))
