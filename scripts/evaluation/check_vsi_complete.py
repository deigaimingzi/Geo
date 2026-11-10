"""Require a complete, unique prediction set before accepting merged VSI scores."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--vsi_jsonl", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args, _ = parser.parse_known_args()
    with args.vsi_jsonl.resolve().open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.max_samples:
        rows = rows[:args.max_samples]
    expected = {row["id"] for row in rows}
    with (args.out_dir.resolve() / "predictions_merged.json").open(encoding="utf-8") as handle:
        records = json.load(handle)
    ids = [row["id"] for row in records]
    missing, extra = expected - set(ids), set(ids) - expected
    if missing or extra or len(ids) != len(set(ids)):
        raise RuntimeError(f"Incomplete VSI evaluation: expected={len(expected)}, records={len(ids)}, missing={len(missing)}, extra={len(extra)}")
    print(f"VSI-Bench coverage verified: {len(ids)}/{len(expected)}", flush=True)


if __name__ == "__main__":
    main()
