"""Bind actual accepted full TRAIN evidence and launch one exact admitted task."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/data2/gb/Dual-RGBT")
PYTHON = str(ROOT / "conda-envs/dual_rgbt/bin/python")
FOLDER = "reports/e0_relative_full_training_r1_20261102"
REVIEW_SHA = "b4c5337cc45e55aa45d8604e8316efe6d779fb63e382561c5565eca6d6592e2a"
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from current_e0_supplement_contract import sha
from full_current_e0_supplement_contract import verify_collection
from relative_e0_full_contract import admission, bind_draft, native_command, recipe
from relative_e0_smoke_contract import verify_completed


def main():
    draft_path = ROOT / FOLDER / "draft.json"
    assert sha(draft_path) == "499d301bab553c0b9384bb86b9af8c214292247fe29a71d057e0e267de84ec46"
    draft = json.loads(draft_path.read_bytes())
    recipe(draft)
    review_path = ROOT / draft["source_review"]
    assert sha(review_path) == REVIEW_SHA
    review = json.loads(review_path.read_bytes())
    assert review["stage"] == draft["stage"] and review["status"] == "SOURCE_PASS"
    assert review["blocking_issues"] == 0
    assert review["target_host"] == "2026" and review["target_root"] == str(ROOT)
    assert review["source_sha256"] == draft["source_sha256"]
    assert review["reviewed_drafts"][draft["reviewed_draft"]] == sha(draft_path)
    for name, expected in draft["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    assert draft["task"] == "train_e0_relative_full_r1_p26_20261102"
    assert draft["run"] == "runs/e0_relative_full_r1_p26_20261102"
    assert draft["assigned_gpu"] == "0"
    assert not (ROOT / draft["run"]).exists()
    assert not (ROOT / "runs/pipeline_tasks" / draft["task"]).exists()
    collection = json.loads((ROOT / draft["collection_plan"]).read_bytes())
    for job in collection["jobs"]:
        status = json.loads((ROOT / "runs/pipeline_tasks" / job["task"] / "status.json").read_bytes())
        assert status["state"] == "COMPLETE" and status["exit_code"] == 0, job["task"]
    acceptance_path = ROOT / draft["collection_acceptance"]
    assert json.loads(acceptance_path.read_bytes()) == verify_collection(ROOT, draft["collection_plan"])
    smoke = verify_completed(ROOT, draft["update_smoke_plan"])
    assert smoke["plan_sha256"] == "dea09e34eaf3d0810f1bb6daec7d9d006a10098557fe7d9d4432ac18f2826d7a"
    assert smoke["receipt_sha256"] == "f30c8693b2407f042b62ac4d5e2ce69e7fe7899c24d1f2738418f97b3d2fd828"
    assert smoke["native_sha256"] == "2bee929423db8ad3a582dc89bb7e83c63fad6377be0bf6400af05b667a7d9276"
    plan = dict(draft, launch_allowed=True, source_review_sha256=REVIEW_SHA,
                collection_acceptance_sha256=sha(acceptance_path),
                update_smoke_plan_sha256=smoke["plan_sha256"],
                update_smoke_receipt_sha256=smoke["receipt_sha256"],
                update_smoke_native_sha256=smoke["native_sha256"])
    bind_draft(plan, draft)
    free = int(subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], universal_newlines=True).strip())
    assert free >= 4096, free
    available_kib = int(next(line for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:")).split()[1])
    assert available_kib >= 16 * 1024 * 1024, available_kib
    plan_path = ROOT / FOLDER / "plan.json"
    with plan_path.open("x") as handle:
        json.dump(plan, handle, indent=2)
    data = admission(ROOT, plan)
    del data
    relative = FOLDER + "/plan.json"
    native = native_command(ROOT, relative)
    record = dict(plan=relative, plan_sha256=sha(plan_path), review_sha256=REVIEW_SHA,
                  started=time.time(), gpu_free_before_mib=free,
                  memory_available_before_kib=available_kib, native_command=native)
    output = plan_path.parent / "launch.json"
    with output.open("x") as handle:
        json.dump(record, handle, indent=2)
    wrapper = [PYTHON, str(ROOT / "scripts/run_task_four_gpu.py"),
               "--task", plan["task"], "--gpu", "0", "--"] + native
    process = subprocess.Popen(
        wrapper, cwd=str(ROOT), env=dict(os.environ, PYTHONUNBUFFERED="1"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    record["wrapper_pid"] = process.pid
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
