"""Archive exact RGBT smoke/start evidence using the existing read-only checks."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

from archive_p1b_evaluation_witness import REMOTE

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("benchmark", choices=("rgbt210", "rgbt234"))
    parser.add_argument("kind", choices=("smoke", "full_start"))
    args = parser.parse_args()
    plans = ("reports/p1b_rgbt_evaluation_r2_20261104"
             if (args.branch, args.benchmark) == ("sot", "rgbt234")
             else "reports/p1b_rgbt_evaluation_r1_20261104")
    folder = ROOT / plans / args.branch / args.benchmark
    output = folder / (args.kind + "_witness")
    assert not output.exists()
    draft = json.loads((folder / "draft.json").read_bytes())
    review = ROOT / "reports" / PurePosixPath(draft["review"]["path"]).name
    result = subprocess.run([
        "ssh", "-T", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
        draft["host"], draft["smoke"]["command"][0], "-B", "-c", shlex.quote(REMOTE)],
        input=json.dumps(dict(draft=draft, kind=args.kind,
                              review_sha256=hashlib.sha256(review.read_bytes()).hexdigest())),
        text=True, capture_output=True, check=True)
    audit = json.loads(result.stdout)
    files = audit.pop("files")
    output.mkdir(parents=True, exist_ok=False)
    for name, item in files.items():
        raw = base64.b64decode(item["raw"])
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        (output / name).write_bytes(raw)
    audit["archived_sha256"] = {name: item["sha256"] for name, item in files.items()}
    (output / "witness.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
