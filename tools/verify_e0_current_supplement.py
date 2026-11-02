"""Verify a completed current-E0 evidence smoke against its accepted M2 caches."""
import argparse
import json
from pathlib import Path

from current_e0_supplement_contract import verify_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_run(args.root, args.root / args.run)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
