"""Verify every completed current-E0 supplement shard before training."""
import argparse
import json
from pathlib import Path

from full_current_e0_supplement_contract import verify_collection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_collection(args.root, args.plan)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("trajectories", "artifacts_sha256", "fit_sequences", "holdout_sequences")}))


if __name__ == "__main__":
    main()
