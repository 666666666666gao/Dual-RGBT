"""Prepare and launch one source-bound TRAIN-only M1 seed replication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REMOTE = Path("/data2/gb/Dual-RGBT")
DRAFT = "reports/E0_M1_SEED2028_DRAFT_R1_20261101.json"
REVIEW = "reports/E0_M1_SEED2028_PLAN_REVIEW_R1_20261101.json"
TASK = "train_protected_e0_m1_seed2028_r1_p26_20261101"
OUTPUT = "runs/protected_e0_offline_m1_seed2028_r1_20261101"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(root):
    run = root / "runs/protected_e0_offline_m1_20261031"
    native = root / "runs/pipeline_tasks/train_protected_e0_offline_m1_r5_20261101/status.json"
    status = json.loads(native.read_bytes())
    assert status["state"] == "COMPLETE" and status["exit_code"] == 0
    config = json.loads((run / "config.json").read_bytes())
    receipt = json.loads((run / "COMPLETE.json").read_bytes())
    assert all(sha(run / name) == value for name, value in receipt["artifacts_sha256"].items())
    plan = config["plan"]
    assert all(sha(root / name) == value for name, value in plan["source_sha256"].items())
    assert sha(root / plan["collection_acceptance"]) == plan["collection_acceptance_sha256"]
    assert sha(root / plan["source_review"]) == plan["source_review_sha256"]
    assert plan["seed"] == 2027 and plan["weight_decay"] == .0001
    assert plan["epochs"] == 12 and plan["batch_states"] == 4096
    return dict(plan=plan, original_config_sha256=sha(run / "config.json"),
                original_native_sha256=sha(native),
                original_complete_sha256=sha(run / "COMPLETE.json"))


def prepare():
    root = Path(__file__).resolve().parents[1]
    source = Path(__file__).read_text()
    probe = source.split("def prepare():")[0] + "\nprint(json.dumps(baseline(REMOTE)))\n"
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "2026",
         (REMOTE / "conda-envs/dual_rgbt/bin/python").as_posix() + " -"],
        input=probe.encode(), stdout=subprocess.PIPE, check=True)
    evidence = json.loads(result.stdout)
    original = evidence.pop("plan")
    plan = dict(original, launch_allowed=False, seed=2028, planned_output=OUTPUT)
    path = root / DRAFT
    assert not path.exists()
    path.write_text(json.dumps(plan, indent=2) + "\n")
    evidence.update(stage="M1_seed_replication_plan", draft=DRAFT,
        reviewed_draft_sha256=sha(path), launcher_sha256=sha(__file__),
        source_sha256=plan["source_sha256"], intended_task=TASK,
        intended_gpu="2", launch_allowed=False, training_only=True,
        test_launch_allowed=False, test_outcomes_already_observed=True,
        test_eligible=False, scientific_result_audit_accepted=False,
        scope="One fixed TRAIN seed replication after the original full TEST "
              "results were observed. Same architecture/cache/split/loss/recipe "
              "except seed2027->2028. Select epoch only on TRAIN component "
              "holdout. No TEST inference, threshold change or TEST-based "
              "checkpoint/model selection admitted by this plan.")
    output = root / "reports/E0_M1_SEED2028_EVIDENCE_R1_20261101.json"
    assert not output.exists()
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)


def launch(review_sha):
    root = REMOTE
    report = root / REVIEW
    assert sha(report) == review_sha
    review = json.loads(report.read_bytes())
    assert review["stage"] == "M1_seed_replication_plan"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["target_host"] == "2026" and review["target_root"] == str(root)
    assert review["reviewed_draft_sha256"] == sha(root / DRAFT)
    assert review["launcher_sha256"] == sha(__file__)
    draft = json.loads((root / DRAFT).read_bytes())
    original = baseline(root)["plan"]
    assert set(draft) == set(original)
    assert {key for key in draft if draft[key] != original[key]} == {
        "launch_allowed", "seed", "planned_output"}
    assert draft["launch_allowed"] is False and draft["seed"] == 2028
    assert draft["planned_output"] == OUTPUT
    assert draft["source_sha256"] == review["source_sha256"]
    assert draft["split"] == "train" and draft["test_eligible"] is False
    usage = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"], universal_newlines=True)
    free = {int(row.split(",")[0]): int(row.split(",")[1])
            for row in usage.strip().splitlines()}
    assert free[2] > 8 * 1024
    assert not (root / OUTPUT).exists()
    assert not (root / "runs/pipeline_tasks" / TASK).exists()
    final = root / "reports/E0_M1_SEED2028_FINAL_PLAN_R1_20261101.json"
    assert not final.exists()
    final.write_text(json.dumps(dict(draft, launch_allowed=True), indent=2) + "\n")
    python = str(root / "conda-envs/dual_rgbt/bin/python")
    command = [python, str(root / "scripts/run_task_four_gpu.py"),
        "--task", TASK, "--gpu", "2", "--", python,
        str(root / "scripts/train_e0_reliability.py"),
        "--plan", final.relative_to(root).as_posix(), "--output", OUTPUT]
    child = subprocess.Popen(command, cwd=str(root),
        env=dict(os.environ, OPENCV_FOR_THREADS_NUM="1"), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print(json.dumps(dict(task=TASK, wrapper_pid=child.pid, command=command,
        plan_sha256=sha(final), review_sha256=review_sha,
        launcher_sha256=sha(__file__))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "launch"), required=True)
    parser.add_argument("--review-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        assert args.review_sha256
        launch(args.review_sha256)
