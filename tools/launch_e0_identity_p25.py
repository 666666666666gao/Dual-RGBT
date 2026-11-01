"""Finalize reviewed host25 smoke/full plans and submit fresh owned native tasks."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path("/data2/gb/Dual-RGBT")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from benchmark_inputs import sha
from collect_e0_identity_train import check_admission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--review-sha256", required=True)
    args = parser.parse_args()
    report_path = ROOT / "reports/E0_M2_IDENTITY_COLLECTOR_SOURCE_REVIEW_P25_20261101.json"
    assert sha(report_path) == args.review_sha256
    review = json.loads(report_path.read_bytes())
    assert review["stage"] == "M2_identity_collector"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["target_host"] == "2025" and review["target_root"] == str(ROOT)
    draft = "reports/E0_M2_IDENTITY_%s_DRAFT_R7_20261031.json" % args.phase.upper()
    assert review["reviewed_drafts"][draft] == sha(ROOT / draft)
    plan = json.loads((ROOT / draft).read_bytes())
    assert review["source_sha256"] == plan["source_sha256"]
    plan.update(launch_allowed=True, source_review_sha256=args.review_sha256)
    output = "reports/E0_M2_IDENTITY_%s_PLAN_P25_R1_20261101.json" % args.phase.upper()
    if args.phase == "full":
        smoke_run = ROOT / "runs" / plan["smoke_run"]
        smoke_native = ROOT / "runs/pipeline_tasks" / plan["smoke_task"] / "status.json"
        plan.update(smoke_plan_sha256=sha(ROOT / plan["smoke_plan"]),
                    smoke_receipt_sha256=sha(smoke_run / "COMPLETE.json"),
                    smoke_native_sha256=sha(smoke_native))
    check_admission(ROOT, plan)
    assert shutil.disk_usage(ROOT).free > 15 * 2**30
    usage = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    memory = {int(row.split(",")[0]): int(row.split(",")[1]) for row in usage.strip().splitlines()}
    assert memory[2] < 500 and memory[3] < 500
    assert not (ROOT / output).exists()
    for job in plan["jobs"]:
        assert not (ROOT / "runs" / job["run"]).exists()
        assert not (ROOT / "runs/pipeline_tasks" / job["task"]).exists()
    (ROOT / output).write_text(json.dumps(plan, indent=2) + "\n")
    python = str(ROOT / "conda-envs/dual_rgbt/bin/python")
    for job in plan["jobs"]:
        argv = [python, str(ROOT / "scripts/collect_e0_identity_train.py"),
                "--plan", output, "--run", job["run"], "--task", job["task"],
                "--shard", str(job["shard"])]
        command = [python, str(ROOT / "scripts/run_task_four_gpu.py"), "--task", job["task"],
                   "--gpu", job["assigned_gpu"], "--"] + argv
        child = subprocess.Popen(
            command, cwd=str(ROOT), env=dict(os.environ, OPENCV_FOR_THREADS_NUM="1"),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps(dict(task=job["task"], gpu=job["assigned_gpu"], wrapper_pid=child.pid,
                              run=job["run"], command=command, plan_sha256=sha(ROOT / output))),
              flush=True)


if __name__ == "__main__":
    main()
