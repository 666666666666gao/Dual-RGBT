"""Wait for exact native collectors, then run existing full acceptance/admission."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path("/data2/gb/Dual-RGBT")
PYTHON = str(ROOT / "conda-envs/dual_rgbt/bin/python")
DIRECTORY = ROOT / "reports/relative_full_handoff_r1_20261102"
COLLECTION = ROOT / "reports/e0_current_full_r1_20261102/plan.json"
FULL = ROOT / "reports/e0_relative_full_training_r1_20261102"
ACCEPTANCE = ROOT / "reports/E0_M2_CURRENT_FULL_ACCEPTANCE_R1_20261102.json"
TASK = ROOT / "runs/pipeline_tasks/train_e0_relative_full_r1_p26_20261102/status.json"


def save(state, **evidence):
    record = dict(state=state, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                       time.gmtime()), **evidence)
    (DIRECTORY / "status.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def main():
    assert hashlib.sha256(COLLECTION.read_bytes()).hexdigest() == (
        "61f3dc02d5727093740547666cbf2af95765efa683736f3fa074e7cc7167feef")
    assert hashlib.sha256((FULL / "draft.json").read_bytes()).hexdigest() == (
        "499d301bab553c0b9384bb86b9af8c214292247fe29a71d057e0e267de84ec46")
    collection = json.loads(COLLECTION.read_bytes())
    save("WAITING_FOR_ALL_EIGHT_COLLECTORS", poll_seconds=180)
    while True:
        if TASK.exists():
            save("FULL_TASK_ALREADY_PRESENT", native=json.loads(TASK.read_bytes()))
            return
        if (FULL / "plan.json").exists():
            save("BLOCKED_FULL_PLAN_ALREADY_PRESENT_WITHOUT_NATIVE")
            return
        native_rows = [json.loads(
            (ROOT / "runs/pipeline_tasks" / job["task"] / "status.json").read_bytes())
            for job in collection["jobs"]]
        assert len(native_rows) == 8
        for job, native in zip(collection["jobs"], native_rows):
            assert native["task"] == job["task"]
            assert native["state"] in ("RUNNING", "COMPLETE", "FAILED")
            if native["state"] == "FAILED":
                save("BLOCKED_COLLECTOR_FAILED", native=native)
                return
            if native["state"] == "COMPLETE":
                assert native["exit_code"] == 0
            else:
                live = Path("/proc", str(native["pid"])).exists()
                if not live:
                    # The wrapper saves terminal status before exiting.
                    fresh = json.loads((ROOT / "runs/pipeline_tasks" / job["task"] /
                                        "status.json").read_bytes())
                    if fresh["state"] == "COMPLETE" and fresh["exit_code"] == 0:
                        continue
                    save("BLOCKED_COLLECTOR_PROCESS_MISSING", native=fresh)
                    return
        if all(native["state"] == "COMPLETE" for native in native_rows):
            break
        time.sleep(180)
    save("VERIFYING_COMPLETE_COLLECTION")
    if not ACCEPTANCE.exists():
        verified = subprocess.run([
            PYTHON, "-B", str(ROOT / "tools/verify_e0_current_full.py"),
            "--root", str(ROOT), "--plan", str(COLLECTION.relative_to(ROOT)),
            "--output", str(ACCEPTANCE)], cwd=str(ROOT))
        if verified.returncode != 0:
            save("BLOCKED_FULL_COLLECTION_VERIFICATION", exit_code=verified.returncode)
            return
    save("CHECKING_BOUND_FULL_TRAIN_ADMISSION")
    launched = subprocess.run([PYTHON, "-B", str(DIRECTORY / "bind_launch.py")],
                              cwd=str(ROOT))
    if launched.returncode != 0:
        save("BLOCKED_FULL_TRAIN_ADMISSION_OR_LAUNCH", exit_code=launched.returncode)
        return
    save("FULL_TRAIN_WRAPPER_SUBMITTED",
         launch=json.loads((FULL / "launch.json").read_bytes()),
         collection_acceptance_sha256=hashlib.sha256(ACCEPTANCE.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
