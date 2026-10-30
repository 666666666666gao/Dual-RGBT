"""Register and dispatch retained utility-focal weights on server25."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/home/gaob/Dual-RGBT")
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_inputs import sha
from joint_evaluation_contract import check_files


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    assert json.loads(path.read_text()) == value


def main():
    review_path = ROOT / "reports/CACHED_GAIN_TEST_SOURCE_REVIEW_20261030.json"
    mapping_path = ROOT / "reports/CACHED_GAIN_SERVER25_TEST_MAPPING_20261030.json"
    review = json.loads(review_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    check_files(ROOT, review["source_sha256"])
    assert mapping["status"] == "TEST_MAPPING_VERIFIED"
    assert (mapping["sequences"], mapping["frames"], mapping["checked_image_paths"]) == (
        245, 220703, 441406)
    formal = ROOT / "runs/cached_query_gain_utility_focal_formal_20261030/COMPLETE.json"
    native_path = ROOT / (
        "runs/pipeline_tasks/train_cached_query_gain_utility_focal_formal_20261030/status.json")
    trained = json.loads(formal.read_text())
    native = json.loads(native_path.read_text())
    assert trained["status"] == "TRAIN_COMPLETE_PENDING_DEV"
    assert trained["strict_reload_bit_equal"]
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert [item["epoch"] for item in trained["retained"]] == [6, 7, 8, 9, 10]
    receipt_path = ROOT / "reports/UTILITY_FOCAL_FULL_TEST_DISPATCH_SERVER25_20261030.json"
    assert not receipt_path.exists()
    python = str(ROOT / "conda-envs/dual_rgbt/bin/python")
    cpu = str(ROOT / "conda-envs/cpu-scoring/bin/python")
    jobs = []
    for retained, gpu in zip(trained["retained"], ("1", "2", "3", "1", "2")):
        epoch = retained["epoch"]
        plan = json.loads((ROOT / (
            "reports/CACHED_QUERY_GAIN_UTILITY_FOCAL_E%d_DEV_PLAN_20261030.json" % epoch
        )).read_text())
        assert plan["checkpoint"] == retained["checkpoint"]
        assert sha(ROOT / plan["checkpoint"]) == retained["sha256"] == plan["checkpoint_sha256"]
        assert sha(formal) == plan["training_result_sha256"]
        assert sha(native_path) == plan["training_native_sha256"]
        smoke_path = ROOT / plan["smoke_result"]
        smoke = json.loads(smoke_path.read_text())
        smoke_native_path = ROOT / plan["smoke_native_status"]
        smoke_native = json.loads(smoke_native_path.read_text())
        assert sha(smoke_path) == plan["smoke_result_sha256"]
        assert sha(smoke_native_path) == plan["smoke_native_sha256"]
        assert smoke_native["state"] == "COMPLETE" and smoke_native["exit_code"] == 0
        assert smoke["checkpoint_sha256"] == retained["sha256"]
        assert (smoke["sequences"], smoke["frames"]) == (1, 11)
        plan.update(
            source_sha256=review["source_sha256"],
            source_review=review_path.relative_to(ROOT).as_posix(),
            source_review_sha256=sha(review_path),
            full_evaluation_allowed=True, exploratory_test=True,
            runtime_provider=dict(ssh_alias="2025", project=str(ROOT)),
            dataset_mapping_receipt=mapping_path.relative_to(ROOT).as_posix(),
            dataset_mapping_receipt_sha256=sha(mapping_path))
        run = "utility_focal_e%03d_lasher_test_p25_exploratory_20261030" % epoch
        task = "eval_" + run
        plan_path = ROOT / ("reports/" + run + "_plan.json")
        score = "reports/utility_focal_test_p25_20261030/e%03d.json" % epoch
        assert not plan_path.exists()
        assert not (ROOT / "runs" / run).exists()
        assert not (ROOT / "runs/pipeline_tasks" / task).exists()
        assert not (ROOT / score).exists()
        jobs.append(dict(epoch=epoch, gpu=gpu, run=run, task=task,
                         plan=plan_path.relative_to(ROOT).as_posix(), plan_data=plan,
                         checkpoint=retained["checkpoint"],
                         checkpoint_sha256=retained["sha256"], score=score))
    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        universal_newlines=True).splitlines()
    assert all(int(memory[int(gpu)]) < 500 for gpu in ("1", "2", "3"))
    for job in jobs:
        plan_path = ROOT / job["plan"]
        write_new(plan_path, job.pop("plan_data"))
        job["plan_sha256"] = sha(plan_path)
    receipt = dict(status="REGISTERED", pid=os.getpid(), jobs=jobs,
                   poll_seconds=180, registered=time.time(),
                   script_sha256=sha(Path(__file__)),
                   exploratory_test=True, test_eligible=False,
                   observed_memory_mib=memory,
                   duplicate_check="Server25 and server26 full-test plans checked before dispatch.")
    write_new(receipt_path, receipt)
    for job in jobs:
        command = [
            python, str(ROOT / "scripts/run_task_four_gpu.py"),
            "--task", job["task"], "--gpu", job["gpu"], "--",
            python, str(ROOT / "scripts/eval_cached_query_gain_test.py"),
            "--plan", job["plan"], "--run", job["run"], "--task", job["task"],
            "--split", "test", "--smoke-frames", "0"]
        with (ROOT / ("reports/" + job["run"] + "_wrapper.log")).open("x") as log:
            process = subprocess.Popen(
                command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OPENCV_FOR_THREADS_NUM="1"), start_new_session=True)
        job.update(pid=process.pid, command=command, dispatched=time.time())
        with receipt_path.open("w") as handle:
            json.dump(receipt, handle, indent=2)
            handle.write("\n")
    print(json.dumps(dict(status="DISPATCHED", jobs=[
        {key: job[key] for key in ("epoch", "gpu", "pid", "run")} for job in jobs
    ]), indent=2))


if __name__ == "__main__":
    main()
