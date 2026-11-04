"""Preflight or launch one separately reviewed C2 policy FIT in a new root."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from work.launch_c2_policy_refresh_acceptance import collection_plan_binding

ENTRY = "scripts/train_c2_policy_refresh.py"
VERIFIER = "tools/verify_c2_policy_refresh_collection.py"


def validate_review(plan, review, launcher_sha256):
    assert plan["launch_allowed"] is False
    assert plan["review"]["sha256"] == "PENDING"
    assert review["stage"] == plan["stage"] == "P2_C2_policy_refresh_fit"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == plan["sources"]
    assert collection_plan_binding(plan) in review["admitted_plan_bindings"]
    assert review["operational_sources"][
        "work/launch_c2_policy_refresh_fit.py"] == launcher_sha256
    assert review["actual_model"] == "UNATTESTED"
    assert review["scientific_result_audit_accepted"] is False


def validate_terminal_replay(plan, accepted, native, python):
    assert plan["stage"] == "P2_C2_policy_refresh_fit"
    assert plan["mode"] == "fit" and plan["smoke"] is False
    assert plan["launch_allowed"] is False
    assert plan["review"]["sha256"] == "PENDING"
    assert plan["test_enabled"] is plan["dev_enabled"] is False
    assert accepted["stage"] == "P2_C2_policy_refresh_full_collection_acceptance"
    assert accepted["status"] == "PASS" and accepted["cuda_initialized"] is True
    assert (accepted["sequences"], accepted["tracked_frames"],
            accepted["records"]) == (882, 458553, 91091)
    assert accepted["benchmark_evidence"] is False
    assert accepted["policy_fit_admitted"] is False
    for key in ("host", "variant", "c2_task_id", "executor_binding"):
        assert accepted[key] == plan[key], key
    assert plan["host"] in ("2025", "2026")
    assert plan["variant"] in ("zero", "film")
    assert plan["root"] != accepted["root"]
    assert plan["sources"] == dict(
        accepted["sources"], **{ENTRY: plan["sources"][ENTRY]})
    assert accepted["sources"][VERIFIER] == accepted["verifier_sha256"]
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    collections = accepted["collections"]
    evidence = accepted["native_and_artifact_evidence"]
    assert len(collections) == len(evidence) == 4
    paths = [row["plan"]["command"][-1] for row in collections]
    assert len(set(paths)) == 4
    replay_root = PurePosixPath(accepted["root"])
    assert plan["collection_acceptance_native"]["path"] == str(
        replay_root / "runs/pipeline_tasks" / native["task"] / "status.json")
    assert native["command"] == [
        python, "-B", str(replay_root / VERIFIER), "--plans",
        *paths, "--output", plan["collection_acceptance"]["path"],
    ]
    totals = dict(sequences=0, tracked_frames=0, records=0)
    for shard, (collection, witness) in enumerate(zip(collections, evidence)):
        original = collection["plan"]
        assert original["shard"] == shard and original["shards"] == 4
        for key in ("host", "root", "variant", "c2_task_id",
                    "executor_binding", "sources"):
            assert original[key] == accepted[key], key
        for key in ("seed", "cost_weight", "label_stride"):
            assert original[key] == plan[key], key
        assert witness["plan_path"] == paths[shard]
        assert witness["plan_sha256"] == collection["plan_sha256"]
        for key in totals:
            totals[key] += witness[key]
    assert totals == {key: accepted[key] for key in totals}


def validate_capacity(plan, gpu_rows, process_rows, disk_free_bytes):
    rows = {
        fields[0].strip(): [field.strip() for field in fields]
        for fields in (row.split(",") for row in gpu_rows)
    }
    row = rows[plan["gpu"]]
    assert int(row[1]) >= 20480 and int(row[2]) <= 999
    assert float(row[3]) == (250 if plan["host"] == "2025" else 350)
    assert int(row[4]) < 85
    assert not any(line.split(",")[0].strip() == row[5]
                   for line in process_rows if line.strip())
    assert disk_free_bytes >= 12 * 2**30
    return row


REMOTE = r'''
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import subprocess
import sys
import time

ENTRY = "scripts/train_c2_policy_refresh.py"
VERIFIER = "tools/verify_c2_policy_refresh_collection.py"
__FUNCTIONS__

request = json.loads(bytes.fromhex("__REQUEST__").decode("utf-8"))
plan = request["plan"]
root = Path(plan["root"])
python = plan["command"][0]
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
acceptance_path = Path(plan["collection_acceptance"]["path"])
native_path = Path(plan["collection_acceptance_native"]["path"])
assert sha(acceptance_path) == plan["collection_acceptance"]["sha256"]
assert sha(native_path) == plan["collection_acceptance_native"]["sha256"]
accepted = json.loads(acceptance_path.read_bytes())
native = json.loads(native_path.read_bytes())
validate_terminal_replay(plan, accepted, native, python)
assert socket.gethostname() == {
    "2025": "ubun", "2026": "ubuntu-WS-C621E-SAGE-Series"
}[plan["host"]]
assert Path(python).is_file()
assert root.is_absolute() and root.parent.is_dir() and not root.exists()
assert not Path(plan["run"]).exists()
assert plan["run"] == str(Path(plan["run_root"]) / plan["task"])
assert plan["gpu"] in ("0", "1", "2", "3")
assert Path(plan["task"]).name == plan["task"]
plan_path = Path(plan["command"][-1])
assert plan_path == root / "reports" / plan["task"] / "plan.json"
assert plan["command"] == [
    python, "-B", str(root / ENTRY), "--plan", str(plan_path)]
review_path = Path(plan["review"]["path"])
review_path.relative_to(root)
remote_witness = Path(plan["run_root"]) / (plan["task"] + "_ADMISSION.json")
assert not remote_witness.exists()

original_root = Path(accepted["root"])
source_bytes = {}
for name, digest in plan["sources"].items():
    raw = (bytes.fromhex(request["entry_hex"]) if name == ENTRY
           else (original_root / name).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == digest, name
    source_bytes[name] = raw
for item in plan["executor_binding"]["inputs"].values():
    assert sha(item["path"]) == item["sha256"], item["path"]
for collection, evidence in zip(accepted["collections"],
                                accepted["native_and_artifact_evidence"]):
    original = collection["plan"]
    assert sha(original["command"][-1]) == collection["plan_sha256"]
    assert json.loads(Path(original["command"][-1]).read_bytes()) == original
    assert sha(collection["path"]) == collection["sha256"]
    receipt = json.loads(Path(collection["path"]).read_bytes())
    assert receipt["status"] == "COMPLETE" and receipt["smoke"] is False
    assert receipt["benchmark_evidence"] is False
    assert receipt["executor_preserved"] is True
    assert receipt["plan_sha256"] == collection["plan_sha256"]
    assert receipt["executor_sha256"] == plan["executor_binding"]["inputs"][
        "executor"]["sha256"]
    assert receipt["artifacts"] == evidence["artifact_hashes"]
    assert sha(evidence["native_path"]) == evidence["native_sha256"]
    collection_native = json.loads(Path(evidence["native_path"]).read_bytes())
    assert collection_native["state"] == "COMPLETE"
    assert collection_native["exit_code"] == 0
    assert collection_native["command"] == original["command"]
    assert collection_native["task"] == original["task"]
    assert collection_native["physical_gpu"] == original["gpu"]
    assert sha(evidence["precision_path"]) == evidence["precision_sha256"]
    assert json.loads(Path(evidence["precision_path"]).read_bytes()) == dict(
        matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)

sys.path.insert(0, str(original_root))
from dual_rgbt.c2_policy_refresh import completed_executor_state
saved = completed_executor_state(plan["executor_binding"])
del saved

gpu_rows = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.free,memory.used,power.limit,temperature.gpu,uuid",
    "--format=csv,noheader,nounits"], text=True).splitlines()
process_rows = subprocess.check_output([
    "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
    "--format=csv,noheader,nounits"], text=True).splitlines()
disk_free_bytes = shutil.disk_usage(root.parent).free
gpu_row = validate_capacity(plan, gpu_rows, process_rows, disk_free_bytes)
witness = dict(
    status="PREFLIGHT_PASS", host=plan["host"], hostname=socket.gethostname(),
    task=plan["task"], gpu=plan["gpu"], gpu_row=gpu_row,
    compute_processes=process_rows, disk_free_bytes=disk_free_bytes,
    draft_plan_sha256=request["draft_plan_sha256"],
    review_sha256=request["review_sha256"],
    launcher_sha256=request["launcher_sha256"],
    collection_acceptance=plan["collection_acceptance"],
    collection_acceptance_native=plan["collection_acceptance_native"],
    sources=plan["sources"], observed_at_epoch=time.time())
if request["preflight_only"]:
    print(json.dumps(witness))
    sys.exit(0)

review_raw = bytes.fromhex(request["review_hex"])
assert hashlib.sha256(review_raw).hexdigest() == request["review_sha256"]
validate_review(plan, json.loads(review_raw), request["launcher_sha256"])
root.mkdir(exist_ok=False)
for name, raw in source_bytes.items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    assert sha(path) == plan["sources"][name]
review_path.parent.mkdir(parents=True, exist_ok=True)
review_path.write_bytes(review_raw)
plan["review"]["sha256"] = request["review_sha256"]
plan["launch_allowed"] = True
plan_path.parent.mkdir(parents=True, exist_ok=True)
plan_path.write_text(json.dumps(plan, indent=2) + "\n")

# Load the actual deployed entry without reusing modules from the source root.
for name in list(sys.modules):
    if name == "dual_rgbt" or name.startswith(("dual_rgbt.", "lib.")) or name == "lib":
        del sys.modules[name]
sys.path.remove(str(original_root))
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("c2_fit_entry", root / ENTRY)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
assert entry.load_plan(plan_path) == plan
entry.validate_collection_acceptance(plan)
os.chdir(root)
command = [
    python, "-B", str(root / "scripts/run_task_four_gpu.py"),
    "--task", plan["task"], "--gpu", plan["gpu"], "--",
] + plan["command"]
remote_witness.parent.mkdir(parents=True, exist_ok=True)
witness["admitted_plan_sha256"] = sha(plan_path)
remote_witness.write_text(json.dumps(witness, indent=2) + "\n")
process = subprocess.Popen(
    command, cwd=str(root), stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
witness.update(status="LAUNCHED", wrapper_pid=process.pid, command=command)
remote_witness.write_text(json.dumps(witness, indent=2) + "\n")
print(json.dumps(witness))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--witness", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    plan_raw = Path(args.plan).read_bytes()
    plan = json.loads(plan_raw)
    launcher_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    relative_review = PurePosixPath(plan["review"]["path"]).relative_to(
        PurePosixPath(plan["root"]))
    local_review = ROOT.joinpath(*relative_review.parts)
    review_raw = b"" if args.preflight_only else local_review.read_bytes()
    if not args.preflight_only:
        validate_review(plan, json.loads(review_raw), launcher_sha)
    witness_path = Path(args.witness)
    assert not witness_path.exists()
    payload = dict(
        plan=plan, draft_plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
        launcher_sha256=launcher_sha, preflight_only=args.preflight_only,
        entry_hex=(ROOT / ENTRY).read_bytes().hex(),
        review_hex=review_raw.hex(),
        review_sha256=(None if args.preflight_only
                       else hashlib.sha256(review_raw).hexdigest()))
    script = REMOTE.replace("__FUNCTIONS__", "\n".join(
        inspect.getsource(function) for function in (
            collection_plan_binding, validate_review,
            validate_terminal_replay, validate_capacity,
        ))).replace("__REQUEST__", json.dumps(payload).encode().hex())
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         plan["host"], plan["command"][0], "-B", "-"],
        input=script, text=True, capture_output=True, check=True)
    witness = json.loads(result.stdout)
    assert witness["status"] == (
        "PREFLIGHT_PASS" if args.preflight_only else "LAUNCHED")
    witness_path.write_text(json.dumps(witness, indent=2) + "\n")
    print(json.dumps(witness, indent=2))


if __name__ == "__main__":
    main()
