"""Launch one reviewed weight-decay control without changing admitted M1 source."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/data2/gb/Dual-RGBT")
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_inputs import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-sha256", required=True)
    args = parser.parse_args()
    draft_path = ROOT / "reports/E0_M1_WD1E3_CONTROL_DRAFT_R1_20261101.json"
    review_path = ROOT / "reports/E0_M1_WD1E3_CONTROL_PLAN_REVIEW_R1_20261101.json"
    assert sha(review_path) == args.review_sha256
    review = json.loads(review_path.read_bytes())
    assert review["stage"] == "M1_weight_decay_control_plan"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["target_host"] == "2026" and review["target_root"] == str(ROOT)
    assert review["reviewed_draft_sha256"] == sha(draft_path)
    draft = json.loads(draft_path.read_bytes())
    assert draft["launch_allowed"] is False and draft["test_eligible"] is False
    assert draft["source_sha256"] == review["source_sha256"]
    baseline = json.loads((ROOT / "runs/protected_e0_offline_m1_20261031/config.json").read_bytes())["plan"]
    changed = {name for name in draft if draft[name] != baseline[name]}
    assert changed == {"launch_allowed", "weight_decay", "planned_output"}
    assert set(draft) == set(baseline)
    assert baseline["weight_decay"] == .0001 and draft["weight_decay"] == .001
    assert sha(ROOT / draft["source_review"]) == draft["source_review_sha256"]
    assert sha(ROOT / draft["collection_acceptance"]) == draft["collection_acceptance_sha256"]
    for name, digest in draft["source_sha256"].items():
        assert sha(ROOT / name) == digest
    task = "train_protected_e0_m1_wd1e3_r1_p26_20261101"
    output = "runs/protected_e0_offline_m1_wd1e3_r1_20261101"
    assert draft["planned_output"] == output
    assert not (ROOT / output).exists()
    assert not (ROOT / "runs/pipeline_tasks" / task).exists()
    usage = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        universal_newlines=True)
    free = {row.split(",")[0].strip(): int(row.split(",")[1])
            for row in usage.strip().splitlines()}
    assert free["3"] > 8 * 1024
    plan_path = ROOT / "reports/E0_M1_WD1E3_CONTROL_FINAL_PLAN_R1_20261101.json"
    assert not plan_path.exists()
    plan = dict(draft, launch_allowed=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    python = str(ROOT / "conda-envs/dual_rgbt/bin/python")
    command = [python, str(ROOT / "scripts/run_task_four_gpu.py"),
        "--task", task, "--gpu", "3", "--",
        python, str(ROOT / "scripts/train_e0_reliability.py"),
        "--plan", plan_path.relative_to(ROOT).as_posix(), "--output", output]
    process = subprocess.Popen(command, cwd=str(ROOT),
        env=dict(os.environ, OPENCV_FOR_THREADS_NUM="1"), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    result = dict(task=task, wrapper_pid=process.pid, physical_gpu="3",
        output=output, command=command, review_sha256=args.review_sha256,
        draft_sha256=sha(draft_path), plan_sha256=sha(plan_path),
        launch_script_sha256=sha(__file__))
    receipt = ROOT / "reports/E0_M1_WD1E3_CONTROL_LAUNCH_R1_20261101.json"
    assert not receipt.exists()
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
