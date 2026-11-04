"""Use the unchanged reviewed evaluation operator for exact RGBT-only plans."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

from operate_p1b_evaluation import REMOTE

ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL = (
    "work/operate_p1b_evaluation.py", "work/prepare_p1b_evaluation.py",
    "work/audit_p1b_completed_for_evaluation.py",
    "work/prepare_p1b_rgbt_evaluation.py", "work/operate_p1b_rgbt_evaluation.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("benchmark", choices=("rgbt210", "rgbt234"))
    parser.add_argument("action", choices=("deploy", "smoke", "full", "inspect"))
    args = parser.parse_args()
    folder = ROOT / "reports/p1b_rgbt_evaluation_r1_20261104" / args.branch / args.benchmark
    draft_raw = (folder / "draft.json").read_bytes()
    draft = json.loads(draft_raw)
    review_raw = (ROOT / "reports" / PurePosixPath(draft["review"]["path"]).name).read_bytes()
    request = dict(
        draft=draft, draft_hex=draft_raw.hex(),
        draft_sha256=hashlib.sha256(draft_raw).hexdigest(),
        review_hex=review_raw.hex(), review_sha256=hashlib.sha256(review_raw).hexdigest(),
        operational_sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in OPERATIONAL},
        entry_hex=(ROOT / "scripts/eval_p1b_completed.py").read_bytes().hex(),
        action=args.action)
    receipt = folder / (args.action + "_receipt.json")
    if args.action != "inspect":
        assert not receipt.exists()
    result = subprocess.run([
        "ssh", "-T", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
        draft["host"], draft["smoke"]["command"][0], "-B", "-c", shlex.quote(REMOTE)],
        input=json.dumps(request), text=True, capture_output=True, check=True)
    output = json.loads(result.stdout)
    if args.action != "inspect":
        receipt.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
