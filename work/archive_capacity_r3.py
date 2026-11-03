"""Read capacity native state and preserve exact remote terminal evidence."""
import base64
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import base64, datetime, hashlib, json, subprocess, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
root = Path(spec["root"])
native = root/"runs/pipeline_tasks"/spec["task"]
state = json.loads((native/"status.json").read_text())
result = dict(time_utc=datetime.datetime.utcnow().isoformat()+"Z",native=state)
result["wrapper_live"] = Path("/proc/%d"%state["pid"]).exists()
result["children"] = subprocess.getoutput("pgrep -P %d"%state["pid"])
result["gpu"] = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.free,temperature.gpu,power.limit","--format=csv,noheader"],universal_newlines=True)
result["compute"] = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],universal_newlines=True)
result["log_tail"] = (native/"run.log").read_text(errors="replace")[-2200:]
result["files"] = {}
for name,path in [("runtime/status.json",native/"status.json"),("runtime/run.log",native/"run.log"),("measurement.json",Path(spec["output"]))]:
 if path.exists():
  result["files"][name] = base64.b64encode(path.read_bytes()).decode()
result["collectors"] = []
for i in range(4):
 task = "collect_p1b_%s_full_s%d_r1_p%s_20261102"%(spec["branch"],i,spec["host"][-2:])
 p = json.loads((root/("reports/p1b_%s_full_collection_r1_20261102/s%d/plan.json"%(spec["branch"],i))).read_text())
 n = json.loads((root/"runs/pipeline_tasks"/task/"status.json").read_text())
 progress = json.loads((Path(p["run"])/"progress.json").read_text())
 result["collectors"].append(dict(task=task,native=n,wrapper_live=Path("/proc/%d"%n["pid"]).exists(),children=subprocess.getoutput("pgrep -P %d"%n["pid"]),progress=progress))
print(json.dumps(result))
'''
for branch,host in (("dropmoe","2025"),("sot","2026")):
    folder = ROOT / ("reports/p1b_capacity_r3_20261102/"+branch)
    spec = json.loads((folder/"spec.json").read_text())
    result = subprocess.run(["ssh","-o","ClearAllForwardings=yes","-o","BatchMode=yes",
                             host,spec["command"][0],"-",spec["command"][-1]],
                            input=REMOTE,encoding="utf-8",stdout=subprocess.PIPE,check=True)
    data = json.loads(result.stdout)
    for name,content in data.pop("files").items():
        path = folder/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(base64.b64decode(content))
    (folder/"live_audit.json").write_text(json.dumps(data,indent=2)+"\n")
    print(host,json.dumps(data),flush=True)
