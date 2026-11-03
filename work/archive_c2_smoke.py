"""Archive completed C2 TRAIN smokes without admitting full training."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import base64, csv, hashlib, json, sys
from pathlib import Path
request=json.loads(sys.stdin.read())
root=Path(request["root"])
plan_path=root/"reports"/request["task"]/"plan.json"
plan=json.loads(plan_path.read_bytes())
native_path=root/"runs/pipeline_tasks"/plan["task"]/"status.json"
native=json.loads(native_path.read_bytes())
assert native["task"]==plan["task"] and native["command"]==plan["command"]
if native["state"]!="COMPLETE":
    print(json.dumps(dict(status="NOT_READY",native=native,
        log_tail=(native_path.parent/"run.log").read_text()[-1500:])))
    sys.exit(0)
assert native["exit_code"]==0 and native["physical_gpu"]==plan["gpu"]
assert not Path("/proc/%d"%native["pid"]).exists()
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
run=Path(plan["run"])
receipt=json.loads((run/"SMOKE.json").read_bytes())
assert receipt["status"]=="PASS" and receipt["task"]==plan["task"]
assert receipt["updates"]==10
assert receipt["route_counts"]=={str(i):1 for i in range(10)}
assert receipt["strict_reload"] and receipt["policy_preserved"]
assert receipt["config_sha256"]==sha(run/"config.json")
assert json.loads((run/"config.json").read_bytes())==plan
assert set(receipt["artifacts"])=={"epoch_001.pth"}
for name,digest in receipt["artifacts"].items():
    assert sha(run/name)==digest
rows=list(csv.DictReader((run/"steps.csv").open()))
assert len(rows)==10
assert {(int(row["modality"]),int(row["action"])) for row in rows}=={
    (m,a) for m in range(2) for a in range(5)}
for name,digest in plan["sources"].items():
    assert sha(root/name)==digest
assert sha(plan["review"]["path"])==plan["review"]["sha256"]
paths=dict(native=native_path,plan=plan_path,config=run/"config.json",
           receipt=run/"SMOKE.json",precision=run/"precision.json",
           source_review=Path(plan["review"]["path"]))
files={}
for name,path in paths.items():
    raw=path.read_bytes()
    files[name+".json"]=dict(raw=base64.b64encode(raw).decode(),
                            sha256=hashlib.sha256(raw).hexdigest())
for name,path in (("steps.csv",run/"steps.csv"),("run.log",native_path.parent/"run.log")):
    raw=path.read_bytes()
    files[name]=dict(raw=base64.b64encode(raw).decode(),
                     sha256=hashlib.sha256(raw).hexdigest())
print(json.dumps(dict(status="ARCHIVED_SMOKE_PASS",task=plan["task"],
    updates=10,peak_allocated_bytes=receipt["peak_allocated_bytes"],
    peak_reserved_bytes=receipt["peak_reserved_bytes"],files=files,
    native_training=True,full_training_admitted=False,benchmark_evidence=False)))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("variant", choices=("zero", "film", "shared", "routed"))
    args = parser.parse_args()
    folder = ROOT / "reports/c2_training_smoke_r2_20261103" / args.branch
    plan = json.loads((folder / args.variant / "plan.json").read_bytes())
    archive = folder / args.variant / "runtime"
    assert not archive.exists()
    request = dict(root=plan["root"], task=plan["task"])
    response = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         plan["host"], plan["command"][0], "-B", "-c", shlex.quote(REMOTE)],
        input=json.dumps(request), encoding="utf-8", stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True)
    report = json.loads(response.stdout)
    if report["status"] == "NOT_READY":
        print(json.dumps(report))
        return
    archive.mkdir()
    files = report.pop("files")
    for name, item in files.items():
        raw = base64.b64decode(item["raw"])
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        (archive / name).write_bytes(raw)
    report["archived_sha256"] = {name: item["sha256"] for name, item in files.items()}
    (archive / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
