"""Validate exact archived native evidence for the two completed R3 runs."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for branch in ("dropmoe", "sot"):
    folder = ROOT / "reports/p1b_capacity_r3_20261102" / branch
    spec = json.loads((folder / "spec.json").read_bytes())
    native = json.loads((folder / "runtime/status.json").read_bytes())
    result = json.loads((folder / "measurement.json").read_bytes())
    review_path = ROOT / "reports/E0_EXECUTOR_V2_P1B_CAPACITY_ADMISSION_R3_20261102.json"
    review = json.loads(review_path.read_bytes())
    assert sha(review_path) == spec["review"]["sha256"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    payload = {key: value for key, value in spec.items()
               if key not in ("review", "launch_allowed")}
    binding = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert binding in review["admitted_spec_bindings"]
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == spec["task"] and native["physical_gpu"] == spec["gpu"]
    assert native["command"] == spec["command"]
    assert result["stage"] == spec["stage"] and result["status"] == "COMPLETE"
    assert result["spec_sha256"] == sha(folder / "spec.json")
    assert result["spec_binding_sha256"] == binding
    assert result["tool_sha256"] == spec["tool_sha256"]
    assert result["branch"] == branch and result["batch_size"] == 512
    assert result["learning_rates"] == [spec["preview_lr"], spec["quality_gain_lr"]]
    assert result["executor_preserved"] and result["checkpoint_written"] is False
    assert all(result[key] is False for key in (
        "full_collection_acceptance", "full_fit_admitted", "benchmark_evidence"))
    assert [row["step"] for row in result["updates"]] == [1, 2]
    assert all(math.isfinite(row[key]) for row in result["updates"]
               for key in ("loss", "gradient_norm", "seconds"))
    assert result["free_mib_before_cuda"] >= spec["minimum_free_mib"]
    assert 0 < result["peak_allocated_bytes"] <= result["peak_reserved_bytes"] < result["total_gpu_bytes"]
    print(json.dumps(dict(
        branch=branch, validation="PASS",
        measurement_sha256=sha(folder / "measurement.json"),
        allocated_gib=result["peak_allocated_bytes"] / 2**30,
        reserved_gib=result["peak_reserved_bytes"] / 2**30,
        native_seconds=native["completed"] - native["started"],
        updates=result["updates"])))
