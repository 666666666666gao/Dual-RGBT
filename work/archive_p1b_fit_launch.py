"""Archive real native FIT launch ownership without claiming an epoch result."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import base64,hashlib,json,subprocess,time
from pathlib import Path
request=json.loads(__import__("sys").stdin.read())
root=Path(request["root"])
plan_path=Path(request["plan"])
plan=json.loads(plan_path.read_text())
assert plan["launch_allowed"] is True
native_dir=root/"runs/pipeline_tasks"/plan["task"]
native_path=native_dir/"status.json"
native=json.loads(native_path.read_text())
assert native["state"]=="RUNNING" and native["command"]==plan["command"]
assert native["physical_gpu"]==plan["gpu"] and Path("/proc/%d"%native["pid"]).exists()
children=subprocess.check_output(["pgrep","-P",str(native["pid"])],universal_newlines=True).split()
assert children
child=children[0]
cmdline=Path("/proc/"+child+"/cmdline").read_bytes().split(b"\0")
assert [part.decode() for part in cmdline if part]==plan["command"]
run=Path(plan["run"])
assert json.loads((run/"config.json").read_text())==plan
assert json.loads((run/"precision.json").read_text())==dict(
    matmul_allow_tf32=False,cudnn_allow_tf32=False,cudnn_benchmark=False)
gpu_uuid=subprocess.check_output(
    ["nvidia-smi","--id="+plan["gpu"],"--query-gpu=uuid",
     "--format=csv,noheader"],universal_newlines=True).strip()
compute=subprocess.check_output(
    ["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory",
     "--format=csv,noheader"],universal_newlines=True)
assert any(row.split(",")[0].strip()==child and row.split(",")[1].strip()==gpu_uuid
           for row in compute.splitlines())
files={}
for name,path in dict(native=native_path,config=run/"config.json",
                      precision=run/"precision.json",plan=plan_path,
                      review=Path(plan["review"]["path"])).items():
    raw=path.read_bytes()
    files[name+".json"]=dict(raw=base64.b64encode(raw).decode(),
                             sha256=hashlib.sha256(raw).hexdigest())
assert files["review.json"]["sha256"]==plan["review"]["sha256"]
print(json.dumps(dict(
    observed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
    task=plan["task"],native_state=native["state"],wrapper_pid=native["pid"],
    training_pid=int(child),gpu=plan["gpu"],gpu_uuid=gpu_uuid,compute=compute,
    process=subprocess.check_output(
        ["ps","-p",child,"-o","pid,stat,time,etime,args"],universal_newlines=True),
    gpu_state=subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.limit",
         "--format=csv,noheader"],universal_newlines=True),
    steps_csv_bytes=(run/"steps.csv").stat().st_size if (run/"steps.csv").exists() else 0,
    completed_epochs_claimed=0,benchmark_evidence=False,files=files)))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    args = parser.parse_args()
    folder = ROOT / "reports/p1b_full_fit_r1_20261102" / args.branch
    witness = folder / "launch_witness"
    assert not witness.exists()
    plan = json.loads((folder / "admitted/plan.json").read_bytes())
    request = dict(root=plan["root"], plan=plan["command"][-1])
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         plan["host"], "python3", "-c", shlex.quote(REMOTE)],
        input=json.dumps(request), encoding="utf-8", stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True)
    report = json.loads(result.stdout)
    files = report.pop("files")
    witness.mkdir()
    for name, item in files.items():
        raw = base64.b64decode(item["raw"])
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        (witness / name).write_bytes(raw)
    report["archived_sha256"] = {name: item["sha256"] for name, item in files.items()}
    (witness / "observation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
