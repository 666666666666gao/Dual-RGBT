"""Native admitted recovery-objective TRAIN, or its terminal CPU verifier."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from context_recovery_execution import data_for_plan, review_sources, verify_run, verify_smoke
from context_recovery_training import train
from current_e0_supplement_contract import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--verify-output")
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    if args.verify_output:
        plan = json.loads((ROOT / args.plan).read_bytes())
        assert plan["phase"] == "full"
        verify_smoke(ROOT, plan)
        result = verify_run(ROOT, args.plan)
        with (ROOT / args.verify_output).open("x") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result), flush=True)
        return
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_bytes())
    review_sources(ROOT, plan)
    assert plan["target_root"] == str(ROOT) and plan["target_hostname"] == socket.gethostname()
    assert sys.executable == plan["interpreter"]
    assert plan["device"] == "cuda"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["assigned_gpu"]
    assert plan["phase"] in ("smoke", "full")
    data = data_for_plan(ROOT, plan)
    if plan["phase"] == "full":
        verify_smoke(ROOT, plan)
    else:
        assert plan["smoke_receipt_sha256"] is None and plan["smoke_native_sha256"] is None
    native = json.loads((ROOT / "runs/pipeline_tasks" / plan["task"] / "status.json").read_bytes())
    assert native["state"] == "RUNNING" and native["task"] == plan["task"]
    assert native["pid"] == os.getppid() and native["hostname"] == plan["target_hostname"]
    assert native["physical_gpu"] == plan["assigned_gpu"]
    assert native["command"] == [plan["interpreter"], str(ROOT / "scripts/train_e0_context_recovery.py"),
                                 "--plan", args.plan]
    plan_sha = sha(plan_path)
    result = train(data, plan, ROOT / plan["run"], plan_sha)
    assert sha(plan_path) == plan_sha
    for name, digest in plan["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
