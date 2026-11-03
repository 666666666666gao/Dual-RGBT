"""Read actual snapshot dependencies and benchmark inputs for fixed P1A controls."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/p1a_evaluation_r1_20261102"
NAMES = list(json.loads((ROOT / "reports/E0_FINETUNED_EVALUATION_SOURCE_REVIEW_20261101.json")
                       .read_bytes())["source_sha256"])
REMOTE = r'''
import hashlib, json, os, subprocess, sys
from pathlib import Path
request = json.loads(sys.stdin.read())
root, parent = Path(request["root"]), Path(request["parent"])
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as handle:
  for block in iter(lambda:handle.read(1048576),b""):h.update(block)
 return h.hexdigest()
os.chdir(str(root));sys.path.insert(0,str(root))
from dual_rgbt.e0_policy_refresh import load_plan
plan=load_plan(request["collector_plan"],"collect")
source={}
for name in request["names"]:
 source[name]=dict(present=(root/name).is_file())
 if source[name]["present"]:
  source[name]["sha256"]=sha(root/name)
 else:
  source[name]["parent_present"]=(parent/name).is_file()
  if source[name]["parent_present"]:
   source[name]["parent_sha256"]=sha(parent/name)
manifests={}
for key,path in request["manifests"].items():
 p=Path(path);row=dict(path=path,present=p.is_file())
 if row["present"]:
  data=json.loads(p.read_bytes())
  row.update(sha256=sha(p),**{k:data[k] for k in ("benchmark","split","sequence_count","frame_count")})
 manifests[key]=row
native=json.loads(Path(plan["inputs"]["p1a_native"]["path"]).read_text())
terminal=json.loads(Path(plan["inputs"]["p1a_complete"]["path"]).read_text())
assert native["state"]=="COMPLETE" and native["exit_code"]==0
assert terminal["status"]=="COMPLETE" and terminal["epochs"]==20 and terminal["steps"]==50000
print(json.dumps(dict(host=request["host"],root=str(root),parent=str(parent),
 source_files=source,original98_unchanged=True,original_inputs=plan["inputs"],
 manifests=manifests,p1a_native=native,p1a_complete=terminal,
 protocol=dict(path=str(parent/"reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json"),
               sha256=sha(parent/"reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json")),
 toolkit=str(parent/"vendor/cadtrack/toolkit"))))
'''

for branch, host, parent, snapshot, prefix in (
    ("dropmoe", "2025", "/home/gaob/Dual-RGBT", "p1b_dropmoe_p25_r1_20261102", ""),
    ("sot", "2026", "/data2/gb/Dual-RGBT", "p1b_direct_r3_20261102", "server26/"),
):
    root = parent + "/snapshots/" + snapshot
    request = dict(host=host, root=root, parent=parent, names=NAMES,
                   collector_plan=root + "/reports/p1b_%s_full_collection_r1_20261102/s0/plan.json" % branch,
                   manifests={key:parent + "/data/benchmarks/" + prefix + key + ".json"
                              for key in ("lasher_train_smoke", "lasher_test", "rgbt210_test", "rgbt234_test")})
    interpreter = parent + "/conda-envs/dual_rgbt/bin/python"
    command = ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
               host, interpreter, "-c", "'" + REMOTE.replace("'", "'\"'\"'") + "'"]
    result = subprocess.run(command, input=json.dumps(request), encoding="utf-8",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(host, result.stderr, flush=True)
    result.check_returncode()
    audit = json.loads(result.stdout)
    folder = FOLDER / branch
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "host_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    absent = [key for key, row in audit["source_files"].items() if not row["present"]]
    print(json.dumps(dict(host=host, missing_sources=absent,
                          manifests=audit["manifests"],
                          p1a_status=audit["p1a_native"]["state"])), flush=True)
