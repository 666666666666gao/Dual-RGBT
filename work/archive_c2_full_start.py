"""Archive actual C2 full TRAIN ownership and first flushed optimizer updates."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import base64,csv,hashlib,json,math,subprocess,sys,time
from pathlib import Path
request=json.loads(sys.stdin.read())
root=Path(request["root"])
task=request["task"]
native_path=root/"runs/pipeline_tasks"/task/"status.json"
native=json.loads(native_path.read_bytes())
plan_path=root/"reports"/task/"plan.json"
plan=json.loads(plan_path.read_bytes())
assert native["state"]=="RUNNING" and native["task"]==plan["task"]==task
assert native["command"]==plan["command"] and native["physical_gpu"]==plan["gpu"]
assert plan["smoke"] is False and plan["launch_allowed"] is True
assert Path("/proc/%d"%native["pid"]).exists()
children=subprocess.check_output(["pgrep","-P",str(native["pid"])],universal_newlines=True).split()
assert len(children)==1
child=children[0]
argv=[part.decode() for part in Path("/proc/"+child+"/cmdline").read_bytes().split(b"\0") if part]
assert argv==plan["command"]
run=Path(plan["run"])
assert json.loads((run/"config.json").read_bytes())==plan
rows=list(csv.DictReader((run/"steps.csv").open()))
if len(rows)<25:
    print(json.dumps(dict(status="NOT_READY",task=task,flushed_updates=len(rows))))
    sys.exit(0)
assert [int(row["step"]) for row in rows]==list(range(1,len(rows)+1))
assert all(math.isfinite(float(row[key])) for row in rows for key in ("loss","gradient_norm","seconds"))
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(plan["review"]["path"])==plan["review"]["sha256"]
uuid=subprocess.check_output(["nvidia-smi","--id="+plan["gpu"],"--query-gpu=uuid",
    "--format=csv,noheader"],universal_newlines=True).strip()
compute=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory",
    "--format=csv,noheader"],universal_newlines=True)
assert any(row.split(",")[0].strip()==child and row.split(",")[1].strip()==uuid
           for row in compute.splitlines())
paths=dict(native=native_path,plan=plan_path,config=run/"config.json",
           precision=run/"precision.json",source_review=Path(plan["review"]["path"]))
files={}
for name,path in paths.items():
    raw=path.read_bytes()
    files[name+".json"]=dict(raw=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
for name,path in (("steps.csv",run/"steps.csv"),("run.log",native_path.parent/"run.log")):
    raw=path.read_bytes()
    files[name]=dict(raw=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
print(json.dumps(dict(status="RUNNING_WITH_REAL_UPDATES",observed_utc=time.strftime(
    "%Y-%m-%dT%H:%M:%SZ",time.gmtime()),task=task,wrapper_pid=native["pid"],
    training_pid=int(child),gpu=plan["gpu"],gpu_uuid=uuid,compute=compute,
    gpu_state=subprocess.check_output(["nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.limit",
        "--format=csv,noheader"],universal_newlines=True),
    flushed_updates=len(rows),last_step=rows[-1],completed_epoch_claimed=False,
    benchmark_evidence=False,files=files)))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("variant", choices=("zero", "film"))
    args = parser.parse_args()
    folder = ROOT / "reports/c2_training_full_r1_20261103" / args.branch / args.variant
    plan = json.loads((folder / "plan.json").read_bytes())
    archive = folder / "start_witness"
    assert not archive.exists()
    response = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         plan["host"], plan["command"][0], "-B", "-c", shlex.quote(REMOTE)],
        input=json.dumps(dict(root=plan["root"], task=plan["task"])),
        encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
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
    (archive / "observation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
