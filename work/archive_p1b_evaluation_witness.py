"""Read and archive native P1B smoke/full evidence without starting jobs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import base64, hashlib, json, subprocess, sys
from pathlib import Path
r = json.loads(sys.stdin.read())
d = r["draft"]
root = Path(d["root"])
job = d["smoke"] if r["kind"] == "smoke" else d["jobs"][0]
run = Path(job["run"])
native_dir = root / "runs/pipeline_tasks" / job["task"]
files = {}
for name, path in [
        ("native.json", native_dir / "status.json"),
        ("run.log", native_dir / "run.log"),
        ("config.json", run / "config.json"),
        ("precision.json", run / "precision.json"),
        ("SMOKE.json", run / "SMOKE.json"),
        ("COMPLETE.json", run / "COMPLETE.json"),
        ("progress.json", run / "progress.json")]:
    if path.exists():
        raw = path.read_bytes()
        files[name] = dict(raw=base64.b64encode(raw).decode(),
                          sha256=hashlib.sha256(raw).hexdigest())
native = json.loads((native_dir / "status.json").read_bytes())
expected = [r["review_sha256"] if word == "PENDING_REVIEW_SHA256" else word
            for word in job["command"]]
assert native["command"] == expected and native["task"] == job["task"]
assert native["physical_gpu"] == job["gpu"]
children = subprocess.run(["pgrep", "-P", str(native["pid"])],
                          stdout=subprocess.PIPE, text=True).stdout.split()
gpu = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,uuid,memory.used,temperature.gpu,power.draw,power.limit",
    "--format=csv,noheader,nounits"], text=True)
processes = subprocess.check_output([
    "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
    "--format=csv,noheader,nounits"], text=True)
if r["kind"] == "smoke":
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    receipt = json.loads((run / "SMOKE.json").read_bytes())
    assert (receipt["sequences"], receipt["frames"]) == (1, 11)
    assert receipt["state_unchanged"] is True
print(json.dumps(dict(task=job["task"], state=native["state"], children=children,
                      gpu=gpu, gpu_processes=processes, files=files)))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("kind", choices=("smoke", "full_start"))
    args = parser.parse_args()
    folder = ROOT / "reports/p1b_evaluation_r3_20261104" / args.branch
    output = folder / (args.kind + "_witness")
    assert not output.exists()
    draft = json.loads((folder / "draft.json").read_bytes())
    review = ROOT / "reports" / PurePosixPath(draft["review"]["path"]).name
    result = subprocess.run([
        "ssh", "-T", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
        draft["host"], draft["smoke"]["command"][0], "-B", "-c", shlex.quote(REMOTE)],
        input=json.dumps(dict(draft=draft, kind=args.kind,
                              review_sha256=hashlib.sha256(review.read_bytes()).hexdigest())),
        text=True, capture_output=True, check=True)
    audit = json.loads(result.stdout)
    files = audit.pop("files")
    output.mkdir(parents=True, exist_ok=False)
    for name, item in files.items():
        raw = base64.b64decode(item["raw"])
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        (output / name).write_bytes(raw)
    audit["archived_sha256"] = {name: item["sha256"] for name, item in files.items()}
    (output / "witness.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
