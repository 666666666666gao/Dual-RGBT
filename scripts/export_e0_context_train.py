"""Export accepted TRAIN context tensors on the original producer host, CPU only."""
import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import torch
from context_e0_execution import review_sources
from context_e0_train_data import load_context_data
from current_e0_supplement_contract import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_bytes())
    plan_sha = sha(plan_path)
    review_sources(ROOT, plan)
    assert plan["phase"] == "export" and plan["export_root"] == str(ROOT)
    assert plan["export_hostname"] == socket.gethostname()
    parent = ROOT / plan["parent_plan"]
    accepted = ROOT / plan["parent_acceptance"]
    assert sha(parent) == plan["parent_plan_sha256"]
    assert sha(accepted) == plan["parent_acceptance_sha256"]
    prior = json.loads(accepted.read_bytes())
    assert prior["status"] == "RELATIVE_CURRENT_E0_FULL_TRAINING_VERIFIED"
    assert prior["plan_sha256"] == plan["parent_plan_sha256"]
    parent_plan = json.loads(parent.read_bytes())
    for key in ("collection_acceptance_sha256", "initialization_sha256"):
        assert parent_plan[key] == plan[key]
    assert sha(ROOT / plan["core_review"]) == plan["core_review_sha256"]
    core = json.loads((ROOT / plan["core_review"]).read_bytes())
    assert core["stage"] == "M2_context_advantage_core_data"
    assert core["status"] == "SOURCE_PASS" and core["blocking_issues"] == 0
    for name, digest in core["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    torch.set_num_threads(1)
    data = load_context_data(ROOT, plan["parent_plan"])
    output = ROOT / plan["dataset"]
    receipt_path = ROOT / plan["export_receipt"]
    assert not output.exists() and not receipt_path.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    receipt = dict(
        status="CONTEXT_TRAIN_TENSORS_EXPORTED", split="train", test_eligible=False,
        plan_sha256=plan_sha, dataset_sha256=sha(output),
        cuda_initialized=torch.cuda.is_initialized(), optimizer_updates=0,
        shapes={part: {key: list(value.shape) for key, value in fields.items()}
                for part, fields in data.items()},
        **{key: plan[key] for key in (
            "parent_plan_sha256", "parent_acceptance_sha256", "collection_acceptance_sha256",
            "initialization_sha256", "core_review_sha256", "source_sha256")})
    assert receipt["cuda_initialized"] is False
    assert sha(plan_path) == plan_sha
    for name, digest in plan["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    with receipt_path.open("x") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
