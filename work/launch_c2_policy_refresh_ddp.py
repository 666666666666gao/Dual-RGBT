"""Prepare, smoke-test, and launch one independent four-GPU C2 FIT."""
import argparse
import hashlib
import json
import inspect
import subprocess
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from work.prepare_c2_policy_refresh_fit import prepare

DDP_ENTRY = "scripts/train_c2_policy_refresh_ddp.py"
DDP_RUNNER = "scripts/run_task_ddp_four_gpu.py"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def make_plan(old_plan_path, output_dir, host, variant, root, task):
    old = json.loads(Path(old_plan_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_plan = output_dir / "plan.json"
    existing_review = output_dir / "review.json"
    if existing_plan.is_file() and existing_review.is_file():
        plan = json.loads(existing_plan.read_text(encoding="utf-8"))
        assert plan["host"] == host
        assert plan["variant"] == variant
        assert plan["root"] == root
        assert plan["task"] == task
        return plan, existing_plan, existing_review
    assert not existing_plan.exists() and not existing_review.exists()
    archive = ROOT / "reports" / "c2_policy_replay_completed_r1_20261104" / (
        "dropmoe" if host == "2025" else "sot") / variant
    acceptance_raw = (archive / "acceptance.json").read_bytes()
    native_raw = (archive / "native.json").read_bytes()
    plan, _ = prepare(
        acceptance_raw,
        native_raw,
        acceptance_path=old["collection_acceptance"]["path"],
        native_path=old["collection_acceptance_native"]["path"],
        root=root,
        run_root=old["run_root"],
        task=task,
        gpu="0",
        python=old["command"][0],
        review_path=str(PurePosixPath(root) / "reports" / task / "P2_DDP_FIT_REVIEW.json"),
    )
    plan["host"] = host
    plan["variant"] = variant
    plan["root"] = root
    plan["source_root"] = old["root"]
    plan["run"] = str(PurePosixPath(plan["run_root"]) / task)
    plan["task"] = task
    plan["command"] = [
        old["command"][0],
        "-B",
        str(PurePosixPath(root) / DDP_ENTRY),
        "--plan",
        str(PurePosixPath(root) / "reports" / task / "plan.json"),
    ]
    plan["sources"][DDP_ENTRY] = sha((ROOT / DDP_ENTRY).read_bytes())
    plan["sources"][DDP_RUNNER] = sha((ROOT / DDP_RUNNER).read_bytes())
    plan["review"] = {
        "path": str(PurePosixPath(root) / "reports" / task / "P2_DDP_FIT_REVIEW.json"),
        "sha256": "PENDING",
    }
    plan["world_size"] = 4
    plan["local_batch_size"] = 128
    plan["global_batch_size"] = 512
    plan["execution"] = "torchrun_ddp_all_four_gpus"
    review = {
        "stage": plan["stage"],
        "status": "SOURCE_PASS",
        "blocking_issues": 0,
        "host": host,
        "variant": variant,
        "world_size": 4,
        "local_batch_size": 128,
        "global_batch_size": 512,
        "trainer_sha256": plan["sources"][DDP_ENTRY],
        "runner_sha256": plan["sources"][DDP_RUNNER],
        "sources": plan["sources"],
        "scientific_result_audit_accepted": False,
        "engineering_admission": "same-family provisional; benchmark evidence disabled",
    }
    plan_path = output_dir / "plan.json"
    review_path = output_dir / "review.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    return plan, plan_path, review_path


REMOTE = r'''
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

request = json.loads(bytes.fromhex("__REQUEST__").decode())
plan = request["plan"]
root = Path(plan["root"])
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert socket.gethostname() == {"2025": "ubun", "2026": "ubuntu-WS-C621E-SAGE-Series"}[plan["host"]]
assert not Path(plan["run"]).exists()
assert Path(plan["collection_acceptance"]["path"]).is_file()
assert Path(plan["collection_acceptance_native"]["path"]).is_file()
assert root.parent.is_dir()
assert shutil.disk_usage(root.parent).free >= 12 * 2**30
gpu = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,memory.free,memory.used,power.limit,temperature.gpu",
    "--format=csv,noheader,nounits"], text=True).splitlines()
assert len(gpu) == 4
for row in gpu:
    fields = [x.strip() for x in row.split(",")]
    assert int(fields[1]) >= 20480
    assert int(fields[2]) <= 999
    assert float(fields[3]) == (250 if plan["host"] == "2025" else 350)
    assert int(fields[4]) < 85
original_root = Path(plan["source_root"])
if request["mode"] == "smoke":
    assert not root.exists()
    root.mkdir()
    for name, raw_hex in request["sources"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = bytes.fromhex(raw_hex)
        path.write_bytes(raw)
        assert sha(path) == plan["sources"][name], name
    for name in plan["sources"]:
        if name in request["sources"]:
            continue
        source = original_root / name
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        assert sha(path) == plan["sources"][name], name
    review = bytes.fromhex(request["review_hex"])
    review_path = root / "reports" / plan["task"] / "P2_DDP_FIT_REVIEW.json"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_bytes(review)
    plan_path = root / "reports" / plan["task"] / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
else:
    assert root.is_dir()
    plan_path = root / "reports" / plan["task"] / "plan.json"
    review_path = root / "reports" / plan["task"] / "P2_DDP_FIT_REVIEW.json"
    assert plan_path.is_file() and review_path.is_file()
    for name in plan["sources"]:
        assert sha(root / name) == plan["sources"][name], name
    assert json.loads(plan_path.read_text(encoding="utf-8")) == plan
smoke_command = [
    plan["command"][0], "-B", str(root / "__DDP_ENTRY__"), "--plan", str(plan_path), "--smoke"
]
runner = [plan["command"][0], "-B", str(root / "__DDP_RUNNER__"), "--task", plan["task"], "--"] + [
    plan["command"][0], "-B", "-m", "torch.distributed.run",
    "--standalone", "--nproc_per_node=4", str(root / "__DDP_ENTRY__"), "--plan", str(plan_path)
]
if request["mode"] == "smoke":
    subprocess.run(
        [plan["command"][0], "-B", str(root / "__DDP_RUNNER__"), "--task", plan["task"] + "_SMOKE", "--"] + [
            plan["command"][0], "-B", "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node=4", str(root / "__DDP_ENTRY__"),
            "--plan", str(plan_path), "--smoke"
        ], cwd=str(root), check=True)
    print(json.dumps({"status": "SMOKE_PASS", "root": str(root), "plan": str(plan_path)}))
    sys.exit(0)
task_dir = root / "runs" / "pipeline_tasks" / plan["task"]
assert not task_dir.exists()
subprocess.Popen(runner, cwd=str(root), stdin=subprocess.DEVNULL,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 start_new_session=True)
print(json.dumps({"status": "LAUNCHED", "root": str(root), "plan": str(plan_path),
                  "runner": runner}))
'''


def remote_request(plan, review_path, mode):
    sources = {}
    for name in (DDP_ENTRY, DDP_RUNNER):
        sources[name] = (ROOT / name).read_bytes().hex()
    payload = {
        "plan": plan,
        "review_hex": Path(review_path).read_bytes().hex(),
        "sources": sources,
        "mode": mode,
    }
    script = REMOTE.replace("__REQUEST__", json.dumps(payload).encode().hex())
    script = script.replace("__DDP_ENTRY__", DDP_ENTRY).replace("__DDP_RUNNER__", DDP_RUNNER)
    return subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         plan["host"], plan["command"][0], "-B", "-"],
        input=script, text=True, capture_output=True, check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=("smoke", "launch"), required=True)
    args = parser.parse_args()
    plan, plan_path, review_path = make_plan(
        args.old_plan, args.output_dir, args.host, args.variant, args.root, args.task)
    result = remote_request(plan, review_path, args.mode)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


if __name__ == "__main__":
    main()
