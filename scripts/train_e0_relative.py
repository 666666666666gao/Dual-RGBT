"""Train the relative identity head only after full collection and update smoke."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from current_e0_supplement_contract import sha
from relative_e0_full_contract import admission, verify_native
from relative_e0_training import train_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    plan = json.loads((ROOT / args.plan).read_bytes())
    plan_sha = sha(ROOT / args.plan)
    data = admission(ROOT, plan)
    native = json.loads((ROOT / "runs/pipeline_tasks" / plan["task"] / "status.json").read_bytes())
    verify_native(ROOT, plan, native, args.plan)
    assert native["pid"] == os.getppid()
    assert native["command"] == [sys.executable] + sys.argv
    assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["assigned_gpu"]
    provenance = {key: plan[key] for key in (
        "source_review_sha256", "collection_plan_sha256", "collection_acceptance_sha256",
        "initialization_sha256", "update_smoke_plan_sha256", "update_smoke_receipt_sha256",
        "update_smoke_native_sha256")}
    provenance.update(plan_path=args.plan, plan_sha256=plan_sha)
    receipt = train_head(data, plan, ROOT / plan["run"], provenance)
    assert sha(ROOT / args.plan) == plan_sha
    for name, expected in plan["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
