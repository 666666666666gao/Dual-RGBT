"""Run one reviewed C2 TRAIN replay acceptance with exact remote preflight."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess


def collection_plan_binding(plan):
    return hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items()
         if key not in ("launch_allowed", "review")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_collection_source_pin(read_source, pin, entry, collection_plan, review):
    source_raw = read_source(pin["source_plan"])
    assert hashlib.sha256(source_raw).hexdigest() == (
        pin["source_plan_sha256"])
    source_plan = json.loads(source_raw)
    binding = collection_plan_binding(source_plan)

    assert pin["sha256"] == entry["review_sha256"]
    assert Path(collection_plan["root"]) / pin["path"] == (
        Path(collection_plan["review"]["path"]))
    assert pin["binding_sha256"] == binding
    assert collection_plan_binding(collection_plan) == binding
    assert source_plan["task"] == collection_plan["task"]
    assert source_plan["gpu"] == collection_plan["gpu"]
    assert source_plan["variant"] == collection_plan["variant"]
    assert source_plan["shard"] == collection_plan["shard"]
    assert review["sources"] == source_plan["sources"]

    reviewed_drafts = review["reviewed_drafts"][collection_plan["variant"]]
    draft = next(row for row in reviewed_drafts
                 if row["shard"] == collection_plan["shard"])
    assert draft["plan_path"] == pin["source_plan"]
    assert draft["plan_sha256"] == pin["source_plan_sha256"]
    assert draft["admitted_plan_binding"] == binding
    assert draft["task"] == collection_plan["task"]
    assert draft["gpu"] == collection_plan["gpu"]

    original_fields = (
        "original_source_plan",
        "original_source_plan_sha256",
        "original_binding_sha256",
    )
    draft_original_fields = (
        "original_source_plan_path",
        "original_source_plan_sha256",
        "original_source_binding",
    )
    has_original_pin = tuple(field in pin for field in original_fields)
    has_original_draft = tuple(
        field in draft for field in draft_original_fields)
    assert has_original_pin == has_original_draft
    if all(has_original_pin):
        original_raw = read_source(pin["original_source_plan"])
        assert hashlib.sha256(original_raw).hexdigest() == (
            pin["original_source_plan_sha256"])
        original_plan = json.loads(original_raw)
        original_binding = collection_plan_binding(original_plan)
        assert original_binding == pin["original_binding_sha256"]
        assert original_plan["variant"] == source_plan["variant"]
        assert original_plan["shard"] == source_plan["shard"]
        assert original_plan["sources"] == source_plan["sources"]
        assert draft["original_source_plan_path"] == pin["original_source_plan"]
        assert draft["original_source_plan_sha256"] == (
            pin["original_source_plan_sha256"])
        assert draft["original_source_binding"] == (
            pin["original_binding_sha256"])


def validate_expected_counts(totals, expected):
    count_fields = ("sequences", "tracked_frames", "records")
    assert totals == {field: expected[field] for field in count_fields}
    assert expected["benchmark_evidence"] is False
    assert expected["policy_fit_admitted"] is False


REMOTE = r'''
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

__VALIDATION_FUNCTIONS__

request = json.loads(bytes.fromhex("__PAYLOAD_HEX__").decode("utf-8"))
plan = request["plan"]
root = Path(plan["root"])
assert socket.gethostname() == plan["hostname"]
os.chdir(str(root))

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

assert sha256(root / plan["verifier"]["path"]) == plan["verifier"]["sha256"]
assert sha256(root / plan["verifier_test"]["path"]) == plan["verifier_test"]["sha256"]
assert sha256(plan["executor"]["checkpoint"]) == plan["executor"]["checkpoint_sha256"]
assert sha256(plan["executor"]["manifest"]) == plan["executor"]["manifest_sha256"]
assert len(plan["collection_plans"]) == 4
assert [entry["task"] for entry in plan["collection_plans"]] == [
    json.loads(Path(entry["path"]).read_text())["task"]
    for entry in plan["collection_plans"]
]

all_sources = None
totals = {"sequences": 0, "tracked_frames": 0, "records": 0}
checked = []
for entry in plan["collection_plans"]:
    collection_plan = json.loads(Path(entry["path"]).read_text())
    assert sha256(entry["path"]) == entry["sha256"]
    assert collection_plan["task"] == entry["task"]
    assert collection_plan["gpu"] == entry["gpu"]
    assert collection_plan["host"] == plan["host"]
    assert collection_plan["root"] == str(root)
    assert collection_plan["variant"] == plan["executor"]["variant"]
    assert collection_plan["c2_task_id"] == plan["executor"]["c2_task_id"]
    assert collection_plan["smoke"] is False
    assert collection_plan["launch_allowed"] is True
    review_path = root / collection_plan["review"]["path"]
    assert sha256(review_path) == entry["review_sha256"]
    review = json.loads(review_path.read_text())
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == collection_plan["sources"]
    if all_sources is None:
        all_sources = collection_plan["sources"]
    else:
        assert collection_plan["sources"] == all_sources

    binding = collection_plan_binding(collection_plan)
    pinned_review = next(
        row for row in plan["collection_source_reviews"]
        if row["binding_sha256"] == binding)

    run = Path(collection_plan["run"])
    receipt_path = run / "COMPLETE.json"
    receipt = json.loads(receipt_path.read_text())
    assert sha256(receipt_path) == entry["receipt_sha256"]
    assert receipt["status"] == "COMPLETE" and receipt["smoke"] is False
    assert receipt["benchmark_evidence"] is False
    assert receipt["executor_preserved"] is True
    assert receipt["plan_sha256"] == entry["sha256"]
    assert receipt["c2_task_id"] == plan["executor"]["c2_task_id"]
    assert receipt["executor_sha256"] == plan["executor"]["checkpoint_sha256"]
    assert len(receipt["sequence_names"]) == entry["sequences"]
    assert receipt["tracked_frames"] == entry["tracked_frames"]
    assert receipt["records"] == entry["records"]

    native_path = Path(entry["native_status_path"])
    native = json.loads(native_path.read_text())
    assert sha256(native_path) == entry["native_status_sha256"]
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == entry["task"]
    assert native["physical_gpu"] == entry["gpu"]
    assert native["hostname"] == plan["hostname"]
    assert native["command"] == collection_plan["command"]

    precision_path = run / "precision.json"
    precision = json.loads(precision_path.read_text())
    assert sha256(precision_path) == entry["precision_sha256"]
    assert precision == {
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
    }
    for field in totals:
        totals[field] += entry[field]
    checked.append({
        "task": entry["task"],
        "plan_sha256": entry["sha256"],
        "receipt_sha256": entry["receipt_sha256"],
        "native_status_sha256": entry["native_status_sha256"],
        "precision_sha256": entry["precision_sha256"],
        **{field: entry[field] for field in totals},
    })

    validate_collection_source_pin(
        lambda path: bytes.fromhex(request["source_files"][path]),
        pinned_review, entry, collection_plan, review)

validate_expected_counts(totals, plan["expected_acceptance"])
assert len({entry["task"] for entry in plan["collection_plans"]}) == 4
assert len({entry["path"] for entry in plan["collection_plans"]}) == 4
for name, digest in all_sources.items():
    assert sha256(root / name) == digest, name

gpu_rows = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.free,memory.used,power.limit,temperature.gpu,uuid",
    "--format=csv,noheader,nounits",
], text=True).splitlines()
gpu = {
    row.split(",")[0].strip(): [value.strip() for value in row.split(",")]
    for row in gpu_rows
}
gpu_row = gpu[plan["gpu"]]
process_rows = subprocess.check_output([
    "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
    "--format=csv,noheader,nounits",
], text=True).splitlines()
busy_uuids = {
    row.split(",")[0].strip() for row in process_rows if row.strip()
}
limits = plan["runtime_admission_required"]
assert int(gpu_row[1]) >= limits["gpu_free_mib_min"], gpu_rows
assert int(gpu_row[2]) <= limits["gpu_used_mib_max"], gpu_rows
assert float(gpu_row[3]) == limits["power_limit_w_exact"], gpu_rows
assert int(gpu_row[4]) < limits["temperature_c_max_exclusive"], gpu_rows
assert gpu_row[5] not in busy_uuids, process_rows
assert shutil.disk_usage(root).free >= limits["disk_free_gib_min"] * 2**30

task_dir = root / "runs/pipeline_tasks" / plan["task"]
output_path = Path(plan["output"])
remote_witness = Path(plan["remote_witness"])
assert not task_dir.exists()
assert not output_path.exists()
assert not remote_witness.exists()

command = plan["native_command"]
assert Path(plan["python"]).is_file()
assert command == [
    plan["python"], "-B",
    str(root / "scripts/run_task_four_gpu.py"),
    "--task", plan["task"], "--gpu", plan["gpu"], "--",
    plan["python"], "-B", str(root / plan["verifier"]["path"]),
    "--plans",
] + [entry["path"] for entry in plan["collection_plans"]] + [
    "--output", plan["output"],
]

witness = {
    "status": "PREFLIGHT_PASS",
    "host": plan["host"],
    "hostname": socket.gethostname(),
    "task": plan["task"],
    "gpu": plan["gpu"],
    "gpu_row": gpu_row,
    "compute_processes": process_rows,
    "disk_free_bytes": shutil.disk_usage(root).free,
    "plan_sha256": request["plan_sha256"],
    "review_sha256": request["review_sha256"],
    "launcher_sha256": request["launcher_sha256"],
    "checkpoint_sha256": plan["executor"]["checkpoint_sha256"],
    "manifest_sha256": plan["executor"]["manifest_sha256"],
    "collection_evidence": checked,
    "counts": totals,
    "observed_at_epoch": time.time(),
}
if request["preflight_only"]:
    print(json.dumps(witness))
    sys.exit(0)
remote_witness.write_text(json.dumps(witness, indent=2) + "\n")
process = subprocess.Popen(
    command, cwd=str(root), stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    start_new_session=True,
)
witness.update(status="LAUNCHED", wrapper_pid=process.pid)
remote_witness.write_text(json.dumps(witness, indent=2) + "\n")
print(json.dumps(witness))
'''


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_review(plan, review, plan_sha256, launcher_sha256):
    assert plan["status"] == "DISABLED_PENDING_EXACT_REVIEW"
    assert plan["launch_allowed"] is False
    assert review["status"] == "SOURCE_PASS"
    assert review["blocking_issues"] == 0
    assert review["target"]["plan_sha256"] == plan_sha256
    assert review["target"]["launcher_sha256"] == launcher_sha256
    assert review["scientific_result_audit_accepted"] is False
    assert review["actual_model"] == "UNATTESTED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launcher_sha256 = file_sha(__file__)
    assert plan["launcher"]["path"] == Path(__file__).as_posix()
    assert plan["launcher"]["sha256"] == launcher_sha256
    plan_sha256 = file_sha(plan_path)

    review_path = Path(plan["review"]["path"])
    if not args.preflight_only:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        validate_review(plan, review, plan_sha256, launcher_sha256)

    local_witness = Path(plan["local_witness"])
    preflight_witness = Path(plan["preflight_witness"])
    assert not (preflight_witness if args.preflight_only
                else local_witness).exists()
    payload = {
        "plan": plan,
        "plan_sha256": plan_sha256,
        "review_sha256": (
            None if args.preflight_only else file_sha(review_path)),
        "launcher_sha256": launcher_sha256,
        "preflight_only": args.preflight_only,
        "source_files": {},
    }
    for pin in plan["collection_source_reviews"]:
        names = [pin["source_plan"]]
        if "original_source_plan" in pin:
            names.append(pin["original_source_plan"])
        for name in names:
            payload["source_files"][name] = Path(name).read_bytes().hex()
    remote_script = REMOTE.replace(
        "__VALIDATION_FUNCTIONS__",
        "\n".join(inspect.getsource(function) for function in (
            collection_plan_binding,
            validate_collection_source_pin,
            validate_expected_counts,
        )),
    ).replace(
        "__PAYLOAD_HEX__", json.dumps(payload, separators=(",", ":")).encode().hex())
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         plan["host"], "python3", "-"],
        input=remote_script, text=True, capture_output=True, check=True,
    )
    witness = json.loads(result.stdout)
    assert witness["status"] == (
        "PREFLIGHT_PASS" if args.preflight_only else "LAUNCHED")
    if args.preflight_only:
        preflight_witness.write_text(result.stdout, encoding="utf-8")
    else:
        local_witness.write_text(
            json.dumps(witness, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(witness, indent=2))


if __name__ == "__main__":
    main()
