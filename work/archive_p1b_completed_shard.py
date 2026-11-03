"""Archive a terminal collector and verify its exact native and cache bytes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("host", choices=("2025", "2026"))
parser.add_argument("shard", type=int, choices=range(4))
args = parser.parse_args()
branch = "dropmoe" if args.host == "2025" else "sot"
target = ("/home/gaob/Dual-RGBT/snapshots/p1b_dropmoe_p25_r1_20261102"
          if args.host == "2025"
          else "/data2/gb/Dual-RGBT/snapshots/p1b_direct_r3_20261102")
folder = ROOT / ("reports/p1b_completed_shards_r1_20261102/%s/s%d" %
                 (branch, args.shard))
assert not folder.exists()
REMOTE = r'''
import base64, hashlib, json, socket, sys, time
from pathlib import Path
root, branch, shard = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
plan_path = root / ("reports/p1b_%s_full_collection_r1_20261102/s%d/plan.json" %
                    (branch, shard))
plan = json.loads(plan_path.read_text())
native_dir = root / "runs/pipeline_tasks" / plan["task"]
native_path = native_dir / "status.json"
native = json.loads(native_path.read_text())
run = Path(plan["run"])
receipt_path = run / "COMPLETE.json"
receipt = json.loads(receipt_path.read_text())
assert native["state"] == "COMPLETE" and native["exit_code"] == 0
assert native["task"] == plan["task"] and native["command"] == plan["command"]
assert native["hostname"] == socket.gethostname()
assert native["physical_gpu"] == plan["gpu"]
assert receipt["status"] == "COMPLETE" and receipt["smoke"] is False
assert receipt["stage"] == "P1B_direct_E0_collection"
assert receipt["benchmark_evidence"] is False
assert receipt["executor_sha256"] == plan["inputs"]["executor"]["sha256"]
assert receipt["plan_sha256"] == sha(plan_path)
assert json.loads((run / "config.json").read_text()) == plan
assert json.loads((run / "precision.json").read_text()) == dict(
    matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)
manifest = json.loads(Path(plan["inputs"]["manifest"]["path"]).read_text())
expected = manifest["train"][shard::4]
assert receipt["sequence_names"] == [row["name"] for row in expected]
assert receipt["tracked_frames"] == sum(len(row["boxes"]) - 1 for row in expected)
assert receipt["records"] == sum(sum(
    frame > 0 and frame % plan["label_stride"] == 0
    for frame in row["valid_frames"]) for row in expected)
assert len(receipt["artifacts"]) == len(expected)
verified = {}
for name, digest in receipt["artifacts"].items():
    path = run / name
    assert path.resolve().parent == run.resolve()
    assert sha(path) == digest, name
    verified[name] = dict(sha256=digest, bytes=path.stat().st_size)
files = {}
for name, path in [
    ("plan.json", plan_path), ("native.json", native_path),
    ("run.log", native_dir / "run.log"), ("COMPLETE.json", receipt_path),
    ("config.json", run / "config.json"), ("precision.json", run / "precision.json"),
    ("progress.json", run / "progress.json")]:
    raw = path.read_bytes()
    files[name] = dict(sha256=hashlib.sha256(raw).hexdigest(),
                       base64=base64.b64encode(raw).decode())
print(json.dumps(dict(
    observed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    task=plan["task"], state=native["state"], exit_code=native["exit_code"],
    sequences=len(expected), tracked_frames=receipt["tracked_frames"],
    records=receipt["records"], artifacts=verified, files=files,
    scope="terminal shard native/binding/count/cache byte audit",
    full_collection_acceptance=False, gpu_work=False, benchmark_evidence=False)))
'''
result = subprocess.run(
    ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
     args.host, "python3", "-", target, branch, str(args.shard)],
    input=REMOTE.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
result.check_returncode()
report = json.loads(result.stdout)
files = report.pop("files")
folder.mkdir(parents=True)
for name, item in files.items():
    raw = base64.b64decode(item["base64"])
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    (folder / name).write_bytes(raw)
report["archived_sha256"] = {name: item["sha256"] for name, item in files.items()}
(folder / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: report[key] for key in (
    "task", "state", "exit_code", "sequences", "tracked_frames", "records")},
    indent=2))
