"""Register and dispatch retained size-tail checkpoints on server26."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/data2/gb/Dual-RGBT/snapshots/cached_parent_loss_20261030")
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_inputs import sha
from joint_evaluation_contract import check_files, evaluation_plan


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    assert json.loads(path.read_text()) == value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", choices=("r1", "r2", "r3"), default="r1")
    args = parser.parse_args()
    parent_path = "reports/joint_candidate_evaluation/joint_quality_k8_cost0p1_server25_20261027_formal/plan.json"
    parent, spec, _ = evaluation_plan(ROOT, parent_path)
    assert parent["checkpoint_sha256"] == "21e07cd1f645c6f7237e063c81cf5d60f6a14bca710e0180e8a3ca29dc827073"
    manifest = ROOT / spec["datasets"]["LasHeR"]["test"]
    assert sha(manifest) == parent["input_sha256"]["data/benchmarks/lasher_test.json"]
    mapping_path = ROOT / "reports/CACHED_PARENT_LOSS_SERVER26_TEST_MAPPING_20261030.json"
    mapping = json.loads(mapping_path.read_text())
    assert mapping["status"] == "TEST_MAPPING_VERIFIED"
    assert (mapping["sequences"], mapping["frames"]) == (245, 220703)
    evaluator = ROOT / "scripts/eval_size_tail_test.py"
    assert sha(evaluator) == "85e714c04b93de8ea7ad2acb74da13073d1c12e9a2055fd3db3f7fde04e46990"
    assert sha(ROOT / "scripts/eval_size_tail.py") == "ccf05b16cd77a816454cd0a65c17c63ee68a9f46984b5a018ab0702b70e32613"
    review_path = ROOT / "reports/SIZE_TAIL_FULL_TEST_SOURCE_REVIEW_20261030.json"
    review = json.loads(review_path.read_text())
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["actual_model"] == "UNATTESTED"
    assert review["review_independence"] == "same-family"
    assert review["acceptance_status"] == "provisional"
    check_files(ROOT, review["source_sha256"])
    records = {
        "r1": (
            (576, 5, "2", "cd6ced8d8e8bc3bf6ce866c9d0970410d82d387d9e90d9f3a980b538b9c661e6"),
            (640, 4, "3", "b3bee645ba73a4b34d308905f50270812faea5d17ea68330cf0a1e0d9b06f1e4"),
        ),
        "r2": (
            (576, 3, "0", "0344b86208f6bc8222a0879fa0d418edaf08b271b2992968468f227eb97b3790"),
            (640, 3, "0", "3e43f0a4b430282d7864f9e8025094a78d27185fa0425ff5e037bd88a6162de0"),
        ),
        "r3": (
            (576, 1, "1", "902937bb379f743420544f2adeda86daeef19c7780a00b07090575926870cc4f"),
            (576, 2, "2", "954545f519085dcbefdf9d8f53b98339504091460d2d7920833fb6a64e59d118"),
            (576, 4, "3", "61405e38b7f364e136616f8e2ee4ed9af0f426b306dd6e4ab604e2fa55165b1b"),
            (640, 1, "1", "5d8d4af88501929d37b5b5054cbaaf50af1240447919c324cd3db8189d8420b3"),
            (640, 2, "2", "df3e94931c18a5ae981824321593a1704324177707afbad0e85e3d30e8ccb912"),
        ),
    }[args.wave]
    jobs = []
    for batch, epoch, gpu, checkpoint_hash in records:
        run = "size_tail_b%03d_e%03d_lasher_test_p26_exploratory_20261030_%s" % (
            batch, epoch, args.wave)
        task = "eval_" + run
        checkpoint = "runs/size_tail_b%03d_formal_epochs_20261030_train_e%03d/epoch_%03d.pth" % (
            batch, epoch, epoch)
        complete = checkpoint.rsplit("/", 1)[0] + "/COMPLETE.json"
        native_training = "runs/pipeline_tasks/train_size_tail_b%03d_formal_epochs_20261030_train_e%03d/status.json" % (
            batch, epoch)
        assert sha(ROOT / checkpoint) == checkpoint_hash
        training = json.loads((ROOT / complete).read_text())
        native = json.loads((ROOT / native_training).read_text())
        assert training["status"] == "EPOCH_COMPLETE_PENDING_DEV"
        assert training["epoch"] == epoch and training["checkpoint_sha256"] == checkpoint_hash
        assert training["strict_reload_bit_equal"] and training["trainable_parameters"] == 66
        assert native["state"] == "COMPLETE" and native["exit_code"] == 0
        plan = dict(
            mode="size_tail_diagnostic",
            parent_evaluation_plan=parent_path,
            parent_checkpoint_sha256=parent["checkpoint_sha256"],
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_hash,
            reference_run="joint_quality_k8_cost0p1_server25_20261027_formal_lasher_dev_cost0p15_s0of1",
            cost_weight=0.15,
            test_eligible=False,
            source_sha256=dict(review["source_sha256"]),
            source_review=review_path.relative_to(ROOT).as_posix(),
            source_review_sha256=sha(review_path),
            training_task=native["task"],
            training_complete=complete,
            training_complete_sha256=sha(ROOT / complete),
            training_native_status=native_training,
            training_native_sha256=sha(ROOT / native_training),
            training_scope="Completed size-only diagnostic epoch; full-test metrics remain exploratory.",
            full_evaluation_allowed=True,
            exploratory_test=True,
            runtime_provider=dict(ssh_alias="2026", project=str(ROOT)),
            dataset_mapping_receipt=mapping_path.relative_to(ROOT).as_posix(),
            dataset_mapping_receipt_sha256=sha(mapping_path),
        )
        plan_path = "reports/" + run + "_plan.json"
        score = "reports/size_tail_test_p26_20261030/b%03d_e%03d_%s.json" % (
            batch, epoch, args.wave)
        assert not (ROOT / plan_path).exists()
        assert not (ROOT / "runs" / run).exists()
        assert not (ROOT / "runs/pipeline_tasks" / task).exists()
        assert not (ROOT / score).exists()
        jobs.append(dict(gpu=gpu, run=run, task=task, plan=plan_path,
                         plan_data=plan, score=score, checkpoint=checkpoint,
                         checkpoint_sha256=checkpoint_hash))
    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        universal_newlines=True).splitlines()
    admission_mib = 3000 if args.wave == "r3" else 500
    assert all(int(memory[int(job["gpu"])]) < admission_mib for job in jobs)
    for job in jobs:
        path = ROOT / job["plan"]
        write_new(path, job.pop("plan_data"))
        job["plan_sha256"] = sha(path)
    receipt_path = ROOT / (
        "reports/SIZE_TAIL_FULL_TEST_DISPATCH_SERVER26_20261030_%s.json" % args.wave.upper())
    receipt = dict(status="REGISTERED", pid=os.getpid(), jobs=jobs, poll_seconds=180,
                   registered=time.time(), exploratory_test=True, test_eligible=False,
                   observed_memory_mib=memory, review_sha256=sha(review_path))
    write_new(receipt_path, receipt)
    python = str(ROOT / "conda-envs/dual_rgbt/bin/python")
    for job in jobs:
        command = [
            python, str(ROOT / "scripts/run_task_four_gpu.py"),
            "--task", job["task"], "--gpu", job["gpu"], "--",
            python, str(ROOT / "tools/run_size_tail_server26_test.py"),
            "--plan", job["plan"], "--run", job["run"], "--task", job["task"],
            "--split", "test", "--smoke-frames", "0",
        ]
        with (ROOT / ("reports/" + job["run"] + "_wrapper.log")).open("x") as log:
            process = subprocess.Popen(
                command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OPENCV_FOR_THREADS_NUM="1"), start_new_session=True)
        job.update(pid=process.pid, command=command, dispatched=time.time())
        with receipt_path.open("w") as handle:
            json.dump(receipt, handle, indent=2)
            handle.write("\n")
    print(json.dumps(dict(status="DISPATCHED", jobs=[
        {key: job[key] for key in ("gpu", "pid", "run")} for job in jobs
    ]), indent=2))


if __name__ == "__main__":
    main()
