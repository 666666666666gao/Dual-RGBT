"""Score the ten assigned completed evaluations once with existing CPU tools."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("/home/gaob/Dual-RGBT")
DIRECTORY = ROOT / "reports/e0_p25_score_handoff_r1_20261102"
PYTHON = str(ROOT / "conda-envs/cpu-scoring/bin/python")
OWNERSHIP = ROOT / "reports/E0_EP20_DUAL_SERVER_OWNERSHIP_20261102.json"
QUEUE = ROOT / "reports/E0_EP20_FINAL_FIVE_EVALUATION_QUEUE_P25_20261101.json"
PROFILE = ROOT / "reports/E0_FINETUNED_ROUTE_MAC_CPU_CROSSCHECK_P25_20261101.json"
SCORER = ROOT / "tools/score_e0_finetuned.py"
COMPUTE = ROOT / "tools/summarize_e0_full_compute.py"
PROFILE_SHA = "708d3aaafb486a5f21a28bf134207a3ae2bf59e0c61607336e9a6addd4a8989e"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4194304), b""):
            digest.update(block)
    return digest.hexdigest()


def save(state, **values):
    record = dict(state=state, pid=os.getpid(), updated=time.time(), **values)
    (DIRECTORY / "status.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def execute(command, task, phase):
    with (DIRECTORY / (task + "_" + phase + ".log")).open("xb") as log:
        result = subprocess.run(
            command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                     MKL_NUM_THREADS="1"))
    if result.returncode:
        save("BLOCKED_" + phase.upper(), task=task, exit_code=result.returncode)
        return False
    return True


def main():
    assert sha(OWNERSHIP) == "49595047e2b33665e75d4e4d4567fb2d124d3cb6d24daa143c60b27d72fca676"
    assert sha(QUEUE) == "3f24a30b570dab2c39d9e5cfadb2cc8d864c86049bef29ca7e2cfd7d8041efd6"
    assert sha(PROFILE) == PROFILE_SHA
    assert sha(SCORER) == "e3a206a61867991e6547ad41b563356f363f90dbe10dddde2cf96f9359776a68"
    assert sha(COMPUTE) == "efd27f1d4ab4de69e77c91e2033eb644a7b52904c319428c108dce2cc54aa77e"
    jobs = json.loads(OWNERSHIP.read_bytes())["p25_jobs"]
    queue = json.loads(QUEUE.read_bytes())
    assert len(jobs) == 10
    assert sha(queue["source_review"]) == queue["review_sha256"]
    review = json.loads(Path(queue["source_review"]).read_bytes())
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["source_sha256"] == queue["source_sha256"]
    for name, expected in queue["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    completed = []
    save("WAITING_FOR_FULL_NATIVE_COMPLETION", completed=completed, poll_seconds=180)
    while len(completed) < len(jobs):
        for job in jobs:
            task = job["task"]
            if task in completed:
                continue
            native_path = ROOT / "runs/pipeline_tasks" / task / "status.json"
            native = json.loads(native_path.read_bytes())
            assert native["task"] == task and native["physical_gpu"] == job["gpu"]
            assert native["state"] in ("RUNNING", "COMPLETE", "FAILED")
            if native["state"] == "FAILED":
                save("BLOCKED_EVALUATION_FAILED", task=task, native=native, completed=completed)
                return
            if native["state"] == "RUNNING":
                continue
            assert native["exit_code"] == 0
            run = Path(job["run"])
            receipt = json.loads((run / "COMPLETE.json").read_bytes())
            assert receipt["sequences"] == job["sequences"] and receipt["frames"] == job["frames"]
            score = Path(job["score_output"])
            if not score.exists():
                save("SCORING", task=task, completed=completed)
                command = [
                    PYTHON, "-B", str(SCORER), "--run", str(run),
                    "--toolkit", str(ROOT / "vendor/cadtrack/toolkit"),
                    "--protocol", str(ROOT / "reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json"),
                    "--output", str(score)]
                if not execute(command, task, "official_score"):
                    return
            scored = json.loads(score.read_bytes())
            assert scored["status"] == "SCORED_COMPLETE"
            assert scored["native_status_sha256"] == sha(native_path)
            assert scored["checkpoint_sha256"] == job["checkpoint_sha256"]
            assert scored["sequences"] == job["sequences"] and scored["frames"] == job["frames"]
            assert scored["test_eligible"] is False and scored["exploratory_test"] is True
            output = score.with_name(score.stem + "_compute.json")
            if not output.exists():
                save("SUMMARIZING_COMPUTE", task=task, completed=completed)
                command = [
                    PYTHON, "-B", str(COMPUTE), "--root", str(ROOT), "--run", str(run),
                    "--score", str(score), "--score-sha256", sha(score),
                    "--profile", str(PROFILE), "--profile-sha256", PROFILE_SHA,
                    "--output", str(output)]
                if not execute(command, task, "compute"):
                    return
            compute = json.loads(output.read_bytes())
            assert compute["status"] == "COMPLETE_FULL_RUN_COMPUTE_SUMMARY"
            assert compute["task"] == task and compute["score_sha256"] == sha(score)
            assert compute["profile_sha256"] == PROFILE_SHA
            completed.append(task)
            save("RESULT_READY", task=task, completed=completed,
                 score_sha256=sha(score), compute_sha256=sha(output))
        if len(completed) < len(jobs):
            time.sleep(180)
    save("COMPLETE", completed=completed)


if __name__ == "__main__":
    main()
