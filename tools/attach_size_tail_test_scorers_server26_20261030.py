"""Attach CPU scorers after size-tail full-test workers are verified."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path("/data2/gb/Dual-RGBT/snapshots/cached_parent_loss_20261030")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", choices=("r1", "r2", "r3"), default="r1")
    args = parser.parse_args()
    path = ROOT / (
        "reports/SIZE_TAIL_FULL_TEST_DISPATCH_SERVER26_20261030_%s.json" % args.wave.upper())
    receipt = json.loads(path.read_text())
    for job in receipt["jobs"]:
        assert "scorer_pid" not in job
        native = json.loads((ROOT / "runs/pipeline_tasks" / job["task"] / "status.json").read_text())
        config = json.loads((ROOT / "runs" / job["run"] / "config.json").read_text())
        assert native["state"] == "RUNNING" and native["pid"] == job["pid"]
        assert Path("/proc", str(job["pid"])).exists()
        assert native["command"] == config["native_command"]
        assert config["checkpoint_sha256"] == job["checkpoint_sha256"]
        assert config["split"] == "test" and config["smoke_frames"] == 0
        assert config["evaluation_class"] == "size_tail_diagnostic"
        assert config["test_eligible"] is False and config["exploratory_test"] is True
    for job in receipt["jobs"]:
        command = [
            str(ROOT / "conda-envs/cpu-scoring/bin/python"),
            str(ROOT / "tools/score_cached_gain_test.py"),
            "--run", job["run"], "--output", job["score"], "--wait",
        ]
        with (ROOT / ("reports/" + job["run"] + "_scorer.log")).open("x") as log:
            process = subprocess.Popen(
                command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        job.update(scorer_pid=process.pid, scorer_command=command)
        receipt.update(status="INFERENCE_VERIFIED_SCORERS_ATTACHED", updated=time.time())
        with path.open("w") as handle:
            json.dump(receipt, handle, indent=2)
            handle.write("\n")
    print(json.dumps({job["run"]: job["scorer_pid"] for job in receipt["jobs"]}, indent=2))


if __name__ == "__main__":
    main()
