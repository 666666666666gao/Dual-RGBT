"""Run the admitted context-head smoke or fixed full TRAIN experiment."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from context_e0_execution import training_admission
from context_e0_training import train
from current_e0_supplement_contract import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_bytes())
    assert plan["device"] == "cuda"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["assigned_gpu"]
    plan_sha = sha(plan_path)
    data, native = training_admission(ROOT, plan, args.plan)
    assert native["pid"] == os.getppid()
    receipt = train(data, plan, ROOT / plan["run"], plan_sha)
    assert sha(plan_path) == plan_sha
    for name, digest in plan["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
