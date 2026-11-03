"""Read completed native TRAIN evidence; never load models or initialize CUDA."""
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import csv
import hashlib
import json
from pathlib import Path
import time

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

parent = Path("/home/gaob/Dual-RGBT")
training = []
for branch in ("dropmoe", "sot"):
    task = "train_executor_v2_%s_p1_r4_20261102" % branch
    directory = parent / "runs/pipeline_tasks" / task
    native_path = directory / "status.json"
    native = json.loads(native_path.read_text())
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    command = native["command"]
    run = Path(command[command.index("--run") + 1])
    csv_path = run / "steps.csv"
    stats = {}
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            loss = float(row["loss"])
            assert __import__("math").isfinite(loss)
            item = stats.setdefault(epoch, dict(updates=0, loss_sum=0., routes={}))
            item["updates"] += 1
            item["loss_sum"] += loss
            route = row["modality"] + ":" + row["action"]
            item["routes"][route] = item["routes"].get(route, 0) + 1
    durations = {}
    log_path = directory / "run.log"
    for line in log_path.read_text().splitlines():
        words = line.split()
        if words and words[0] == "EPOCH_COMPLETE":
            durations[int(words[1])] = float(words[2])
    assert set(stats) == set(durations) == set(range(1, 21))
    epochs = []
    for epoch in range(1, 21):
        item = stats[epoch]
        assert item["updates"] == 2500 and len(item["routes"]) == 10
        assert set(item["routes"].values()) == {250}
        epochs.append(dict(epoch=epoch, updates=item["updates"],
                           mean_train_loss=item["loss_sum"] / item["updates"],
                           seconds=durations[epoch], route_updates=item["routes"]))
    training.append(dict(
        branch=branch, execution_host="2025", native=native,
        native_path=str(native_path), native_sha256=sha(native_path),
        run=str(run), steps_sha256=sha(csv_path), log_sha256=sha(log_path),
        native_seconds=native["completed"] - native["started"],
        mean_epoch_seconds=sum(durations.values()) / 20, epochs=epochs))

snapshot = parent / "snapshots/p1b_dropmoe_p25_r1_20261102"
task = "collect_p1b_dropmoe_full_s0_r1_p25_20261102"
plan_path = snapshot / "reports/p1b_dropmoe_full_collection_r1_20261102/s0/plan.json"
plan = json.loads(plan_path.read_text())
run = Path(plan["run"])
native_path = snapshot / "runs/pipeline_tasks" / task / "status.json"
native = json.loads(native_path.read_text())
receipt_path = run / "COMPLETE.json"
receipt = json.loads(receipt_path.read_text())
assert native["state"] == "COMPLETE" and native["exit_code"] == 0
assert native["command"] == plan["command"] and native["task"] == task
assert receipt["plan_sha256"] == sha(plan_path)
assert json.loads((run / "config.json").read_text()) == plan
assert len(receipt["sequence_names"]) == len(receipt["artifacts"]) == 221
assert receipt["tracked_frames"] == 98613 and receipt["records"] == 19574
verified = {}
for name, expected in receipt["artifacts"].items():
    path = run / name
    assert path.resolve().parent == run.resolve()
    actual = sha(path)
    assert actual == expected, name
    verified[name] = actual
print(json.dumps(dict(
    observed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    stage="P1A_actual_train_timing_and_P1B_first_shard_byte_audit",
    training=training, shard=dict(
        task=task, native=native, native_sha256=sha(native_path),
        plan_sha256=sha(plan_path), receipt=receipt,
        receipt_sha256=sha(receipt_path), artifact_hashes_verified=verified,
        full_collection_acceptance=False),
    gpu_work=False, test_used=False, convergence_proven=False), indent=2))
'''

output = ROOT / "reports/p1a_train_timing_first_shard_r1_20261102"
output.mkdir(exist_ok=False)
result = subprocess.run(
    ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "2025",
     "python3", "-c", shlex.quote(REMOTE)],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(output / "stderr.log").write_bytes(result.stderr)
result.check_returncode()
report = json.loads(result.stdout)
(output / "audit.json").write_bytes(result.stdout)
print(json.dumps(dict(
    training=[dict(branch=row["branch"], host=row["execution_host"],
                   native_seconds=row["native_seconds"],
                   mean_epoch_seconds=row["mean_epoch_seconds"],
                   epoch1_loss=row["epochs"][0]["mean_train_loss"],
                   epoch20_loss=row["epochs"][-1]["mean_train_loss"])
              for row in report["training"]],
    shard_artifacts_verified=len(report["shard"]["artifact_hashes_verified"]),
    output=str(output / "audit.json")), indent=2))
