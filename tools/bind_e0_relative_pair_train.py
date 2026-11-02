"""Bind completed qualifying TRAIN evidence and submit the fixed paired check."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/data2/gb/Dual-RGBT")
PYTHON = str(ROOT / "conda-envs/dual_rgbt/bin/python")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from relative_e0_pair_contract import (
    FOLDER, admission, bind_draft, native_command, recipe, sha)

DRAFT_SHA = "44fcab8eebfb37e8eaea0cdf8d8da435049592e635c403b1d0d935c2825b0e97"
REVIEW_SHA = "bc174285da2b01abcaf7434692d9ee982a07ee8d9a666dce6401c503c8abf4e0"


def main():
    assert Path.cwd() == ROOT
    draft_path = ROOT / FOLDER / "draft.json"
    assert sha(draft_path) == DRAFT_SHA
    draft = json.loads(draft_path.read_bytes())
    recipe(draft)
    assert sha(ROOT / draft["source_review"]) == REVIEW_SHA
    acceptance_path = ROOT / draft["training_acceptance"]
    trained = json.loads(acceptance_path.read_bytes())
    assert trained["status"] == "RELATIVE_CURRENT_E0_FULL_TRAINING_VERIFIED"
    assert trained["retain_protected_e0"] is False
    assert type(trained["best_qualifying_epoch"]) is int
    assert trained["artifacts_sha256"]["best.pth"] == sha(ROOT / draft["relative_checkpoint"])
    assert trained["plan_sha256"] == sha(ROOT / draft["training_plan"])
    plan_path = ROOT / FOLDER / "plan.json"
    assert not plan_path.exists()
    assert not (ROOT / draft["run"]).exists()
    assert not (ROOT / "runs/pipeline_tasks" / draft["task"]).exists()
    assert not (ROOT / FOLDER / "launch.json").exists()
    plan = dict(
        draft, launch_allowed=True, source_review_sha256=REVIEW_SHA,
        training_plan_sha256=trained["plan_sha256"],
        training_acceptance_sha256=sha(acceptance_path),
        relative_checkpoint_sha256=trained["artifacts_sha256"]["best.pth"],
        qualifying_epoch=trained["best_qualifying_epoch"])
    bind_draft(plan, draft)
    # The canonical admission replays actual terminal TRAIN verification.
    rows, verified = admission(ROOT, plan)
    assert verified == trained and len(rows) == 8
    free = int(subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], text=True).strip())
    assert free >= 8192, free
    available_kib = int(next(
        line for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")).split()[1])
    assert available_kib >= 16 * 1024 * 1024, available_kib
    with plan_path.open("x") as handle:
        json.dump(plan, handle, indent=2)
    command = native_command(ROOT, FOLDER + "/plan.json")
    record = dict(
        plan=FOLDER + "/plan.json", plan_sha256=sha(plan_path),
        review_sha256=REVIEW_SHA, qualifying_epoch=plan["qualifying_epoch"],
        checkpoint_sha256=plan["relative_checkpoint_sha256"],
        training_acceptance_sha256=plan["training_acceptance_sha256"],
        native_command=command, gpu_free_before_mib=free,
        memory_available_before_kib=available_kib, started=time.time())
    output = ROOT / FOLDER / "launch.json"
    with output.open("x") as handle:
        json.dump(record, handle, indent=2)
    wrapper = [PYTHON, str(ROOT / "scripts/run_task_four_gpu.py"),
               "--task", plan["task"], "--gpu", "0", "--"] + command
    process = subprocess.Popen(
        wrapper, cwd=str(ROOT), env=dict(os.environ, PYTHONUNBUFFERED="1"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    record["wrapper_pid"] = process.pid
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
