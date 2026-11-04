"""Deploy reviewed P1B evidence/inference sources and run LasHeR first."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]

REMOTE = r'''
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

request = json.loads(sys.stdin.read())
d = request["draft"]
root = Path(d["root"])
original = Path(d["source_root"])
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_bytes())
assert socket.gethostname() == {
    "2025": "ubun", "2026": "ubuntu-WS-C621E-SAGE-Series"}[d["host"]]
review_raw = bytes.fromhex(request["review_hex"])
assert hashlib.sha256(review_raw).hexdigest() == request["review_sha256"]
review = json.loads(review_raw)
assert review["stage"] == "E0_finetuned_evaluation"
assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
assert review["source_sha256"] == d["source_sha256"]
assert review["checkpoint"] == d["checkpoint"] and review["factory"] == d["factory"]
assert review["target_root"] == str(root) and review["target_host"] == d["host"]
assert review["reviewed_draft"]["sha256"] == request["draft_sha256"]
assert review["operational_sources"] == request["operational_sources"]
assert review["p1b_fit_evidence"] == d["fit_evidence"]
assert review["p1b_fit_plan_binding"] == d["fit_plan_binding"]
for item in d["fit_evidence"].values():
    assert sha(item["remote_path"]) == item["sha256"]
for item in (d["checkpoint"], d["factory"], d["protocol"]):
    assert sha(item["path"]) == item["sha256"]
for job in [d["smoke"]] + d["jobs"]:
    assert sha(job["manifest"]["path"]) == job["manifest"]["sha256"]
for name, digest in d["source_sha256"].items():
    if name != "scripts/eval_p1b_completed.py":
        assert sha(original / name) == digest, name
entry_raw = bytes.fromhex(request["entry_hex"])
assert hashlib.sha256(entry_raw).hexdigest() == d["source_sha256"][
    "scripts/eval_p1b_completed.py"]
for name, digest in read(d["protocol"]["path"])["toolkit_code_sha256"].items():
    assert sha(Path(d["toolkit"]) / name.replace("\\", "/")) == digest, name

action = request["action"]
if action == "deploy":
    assert root.is_absolute() and root.parent.is_dir() and not root.exists()
    root.mkdir(exist_ok=False)
    for name, digest in d["source_sha256"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = entry_raw if name == "scripts/eval_p1b_completed.py" else (original / name).read_bytes()
        path.write_bytes(raw)
        assert sha(path) == digest, name
    path = Path(d["review"]["path"])
    path.relative_to(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(review_raw)
    (root / "reports/draft.json").write_bytes(bytes.fromhex(request["draft_hex"]))
else:
    assert sha(d["review"]["path"]) == request["review_sha256"]
    assert sha(root / "reports/draft.json") == request["draft_sha256"]
    for name, digest in d["source_sha256"].items():
        assert sha(root / name) == digest, name
os.chdir(root)
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scripts"))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from eval_p1b_completed import validate_fit
validate_fit(review, d["checkpoint"]["path"], d["checkpoint"]["sha256"])
if action == "deploy":
    import torch
    factory_state = torch.load(d["factory"]["path"], map_location="cpu")
    assert "net" in factory_state and not torch.cuda.is_initialized()

def command(job):
    return [request["review_sha256"] if word == "PENDING_REVIEW_SHA256" else word
            for word in job["command"]]

def native_path(job):
    return root / "runs/pipeline_tasks" / job["task"] / "status.json"

def completed_smoke():
    job = d["smoke"]
    native = read(native_path(job))
    run = Path(job["run"])
    config = read(run / "config.json")
    receipt = read(run / "SMOKE.json")
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["command"] == config["native_command"] == command(job)
    assert native["task"] == config["task"] == receipt["task"] == job["task"]
    assert native["physical_gpu"] == config["physical_gpu"] == job["gpu"]
    assert receipt["sequences"] == 1 and receipt["frames"] == 11
    assert receipt["smoke_only"] is True and receipt["state_unchanged"] is True
    assert receipt["optimizer_updates"] == receipt["backward_calls"] == 0
    assert receipt["checkpoint_sha256"] == config["checkpoint_sha256"] == d["checkpoint"]["sha256"]
    assert receipt["config_sha256"] == sha(run / "config.json")
    assert config["source_review_sha256"] == request["review_sha256"]
    assert config["source_sha256"] == d["source_sha256"]
    assert read(run / "precision.json") == dict(
        matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)

output = dict(action=action, host=d["host"], root=str(root), status="VERIFIED")
if action == "deploy":
    output["factory_pickle_cpu_loaded"] = True
if action in ("smoke", "full"):
    if action == "full":
        completed_smoke()
    job = d["smoke"] if action == "smoke" else d["jobs"][0]
    assert not native_path(job).parent.exists() and not Path(job["run"]).exists()
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,memory.used,temperature.gpu,power.limit,uuid",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    row = next([word.strip() for word in line.split(",")] for line in rows
               if line.split(",")[0].strip() == job["gpu"])
    assert int(row[1]) >= 6000 and int(row[2]) < 500
    assert int(row[3]) < 85 and float(row[4]) == (250 if d["host"] == "2025" else 350)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    assert not any(line.split(",")[0].strip() == row[5] for line in processes)
    assert shutil.disk_usage(root).free >= 5 * 2**30
    argv = [job["command"][0], "-B", str(root / "scripts/run_task_four_gpu.py"),
            "--task", job["task"], "--gpu", job["gpu"], "--"] + command(job)
    child = subprocess.Popen(
        argv, cwd=str(root), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), start_new_session=True)
    output.update(status="LAUNCHED", task=job["task"], wrapper_pid=child.pid,
                  gpu_row=row, command=argv)
elif action == "inspect":
    output["tasks"] = []
    for job in [d["smoke"]] + d["jobs"]:
        path = native_path(job)
        row = dict(task=job["task"], native_exists=path.exists())
        if path.exists():
            native = read(path)
            row["native"] = native
            row["wrapper_live"] = Path("/proc/%s" % native["pid"]).exists()
            row["children"] = subprocess.run(
                ["pgrep", "-P", str(native["pid"])],
                stdout=subprocess.PIPE, text=True).stdout.split()
            for name in ("progress.json", "SMOKE.json", "COMPLETE.json"):
                if (Path(job["run"]) / name).exists():
                    row[name] = read(Path(job["run"]) / name)
        output["tasks"].append(row)
print(json.dumps(output))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("action", choices=("deploy", "smoke", "full", "inspect"))
    args = parser.parse_args()
    folder = ROOT / "reports/p1b_evaluation_r3_20261104" / args.branch
    draft_raw = (folder / "draft.json").read_bytes()
    draft = json.loads(draft_raw)
    review_raw = (ROOT / "reports" / PurePosixPath(draft["review"]["path"]).name).read_bytes()
    operational = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in ("work/operate_p1b_evaluation.py", "work/prepare_p1b_evaluation.py",
                                "work/audit_p1b_completed_for_evaluation.py")}
    request = dict(
        draft=draft, draft_hex=draft_raw.hex(),
        draft_sha256=hashlib.sha256(draft_raw).hexdigest(),
        review_hex=review_raw.hex(), review_sha256=hashlib.sha256(review_raw).hexdigest(),
        operational_sources=operational,
        entry_hex=(ROOT / "scripts/eval_p1b_completed.py").read_bytes().hex(),
        action=args.action)
    receipt = folder / (args.action + "_receipt.json")
    if args.action != "inspect":
        assert not receipt.exists()
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         draft["host"], draft["smoke"]["command"][0], "-B", "-c", shlex.quote(REMOTE)],
        input=json.dumps(request), text=True, capture_output=True, check=True)
    output = json.loads(result.stdout)
    if args.action != "inspect":
        receipt.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
