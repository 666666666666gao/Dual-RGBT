"""Deploy exact reviewed fixed-epoch controls, then admit smoke and full runs."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("branch", choices=("dropmoe", "sot"))
parser.add_argument("action", choices=("deploy", "smoke", "full", "inspect"))
args = parser.parse_args()
folder = ROOT / "reports/p1a_evaluation_r1_20261102" / args.branch
draft_raw = (folder / "draft.json").read_bytes()
draft = json.loads(draft_raw)
review_name = PurePosixPath(draft["review"]["path"]).name
review_path = ROOT / "reports" / review_name
review_raw = review_path.read_bytes()
review = json.loads(review_raw)
review_sha = hashlib.sha256(review_raw).hexdigest()
assert review["stage"] == "E0_finetuned_evaluation"
assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
assert review["target_root"] == draft["root"]
assert review["checkpoint"] == draft["checkpoint"]
assert review["factory"] == draft["factory"]
assert review["source_sha256"] == draft["source_sha256"]
assert review["reviewed_draft"]["sha256"] == hashlib.sha256(draft_raw).hexdigest()
assert len(review["source_sha256"]) == 49
ssh = ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", draft["host"]]
remote_folder = "reports/p1a_evaluation_r1_20261102/" + args.branch
if args.action == "deploy":
    names = draft["additive_source_files"]
    assert len(names) == 24
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name in names:
            path = folder / "source_payload" / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == draft["source_sha256"][name]
            archive.add(str(path), arcname=name)
        archive.add(str(review_path), arcname="reports/" + review_name)
        archive.add(str(folder / "draft.json"), arcname=remote_folder + "/draft.json")
    absent = names + ["reports/" + review_name, remote_folder + "/draft.json"]
    check = " && ".join("test ! -e " + shlex.quote(draft["root"] + "/" + name)
                        for name in absent)
    subprocess.run(ssh + [check], check=True)
    subprocess.run(ssh + ["tar", "-C", draft["root"], "-xf", "-"],
                   input=payload.getvalue(), check=True)

REMOTE = r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
request=json.loads(sys.stdin.read())
d=request["draft"];root=Path(d["root"])
os.chdir(str(root));sys.path.insert(0,str(root))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1048576),b""):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
review=read(d["review"]["path"])
assert sha(d["review"]["path"])==request["review_sha"]
assert sha(root/request["remote_folder"]/"draft.json")==request["draft_sha"]
assert review["source_sha256"]==d["source_sha256"]
assert all(sha(root/name)==digest for name,digest in d["source_sha256"].items())
from dual_rgbt.e0_policy_refresh import load_plan
p=load_plan(str(root/("reports/p1b_%s_full_collection_r1_20261102/s0/plan.json"%d["branch"])),"collect")
assert p["inputs"]==d["original98_inputs"]
assert sha(d["checkpoint"]["path"])==d["checkpoint"]["sha256"]
assert sha(d["factory"]["path"])==d["factory"]["sha256"]
assert read(p["inputs"]["p1a_native"]["path"])==d["p1a_native"]
assert read(p["inputs"]["p1a_complete"]["path"])==d["p1a_complete"]
for job in [d["smoke"]]+d["jobs"]:
 assert sha(job["manifest"]["path"])==job["manifest"]["sha256"]
assert sha(d["protocol"]["path"])==d["protocol"]["sha256"]
toolkit=Path(d["toolkit"])
for name,digest in d["source_sha256"].items():
 prefix="vendor/cadtrack/toolkit/"
 if name.startswith(prefix):assert sha(toolkit/name[len(prefix):])==digest
for name,digest in read(d["protocol"]["path"])["toolkit_code_sha256"].items():
 assert sha(toolkit/name.replace("\\","/"))==digest
parent=Path(d["jobs"][0]["command"][0]).parents[3]
cpu=parent/"conda-envs/cpu-scoring/bin/python"
def command(job):
 return [request["review_sha"] if v=="PENDING_REVIEW_SHA256" else v for v in job["command"]]
def native(job):return root/"runs/pipeline_tasks"/job["task"]/"status.json"
def inspect(job):
 path=native(job);row=dict(task=job["task"],native_exists=path.exists())
 if path.exists():
  n=read(path);row["native"]=n;row["wrapper_live"]=Path("/proc/%d"%n["pid"]).exists()
  row["children"]=subprocess.run(["pgrep","-P",str(n["pid"])],stdout=subprocess.PIPE,universal_newlines=True).stdout.split()
  run=Path(job["run"])
  for name in ("progress.json","SMOKE.json","COMPLETE.json"):
   if (run/name).exists():row[name]=read(run/name)
  traces=list((run/"traces").glob("*.csv"))
  if traces:
   latest=max(traces,key=lambda p:p.stat().st_mtime)
   row["latest_trace"]=str(latest);row["trace_age"]=time.time()-latest.stat().st_mtime
 return row
action=request["action"];output=dict(action=action,host=d["host"],observed=time.time(),
 original98_unchanged=True,input_count=len(p["inputs"]),review_sha256=request["review_sha"])
if action=="deploy":
 code="import sys,runpy;sys.path.insert(0,%r);from rgbt import LasHeR,RGBT210,RGBT234;print([len(c().seqs_name) for c in (LasHeR,RGBT210,RGBT234)]);runpy.run_path(%r,run_name='score_import_witness')"%(str(toolkit),str(root/"tools/score_e0_finetuned.py"))
 result=subprocess.run([str(cpu),"-B","-c",code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 output["cpu_import_stdout"]=result.stdout;output["cpu_import_stderr"]=result.stderr
 assert result.returncode==0,output
elif action in ("smoke","full"):
 jobs=[d["smoke"]] if action=="smoke" else d["jobs"]
 if action=="full":
  s=d["smoke"];n=read(native(s));run=Path(s["run"])
  r=read(run/"SMOKE.json");c=read(run/"config.json")
  assert n["state"]=="COMPLETE" and n["exit_code"]==0
  assert n["task"]==s["task"]==r["task"]==c["task"]
  assert n["command"]==c["native_command"]==command(s)
  assert n["physical_gpu"]==c["physical_gpu"]==s["gpu"]
  assert r["sequences"]==1 and r["frames"]==11 and r["smoke_only"]
  assert r["state_unchanged"] and r["optimizer_updates"]==r["backward_calls"]==0
  assert r["config_sha256"]==sha(run/"config.json")
  assert c["source_sha256"]==d["source_sha256"]
  assert c["source_review_sha256"]==request["review_sha"]
  assert c["checkpoint_sha256"]==r["checkpoint_sha256"]==d["checkpoint"]["sha256"]
  assert c["manifest_sha256"]==s["manifest"]["sha256"]
  assert c["exploratory_test"] is r["exploratory_test"] is False
  assert c["test_eligible"] is r["test_eligible"] is False
  output["smoke_verified"]=True
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free,temperature.gpu,power.limit","--format=csv,noheader,nounits"],universal_newlines=True)
 rows={row.split(",")[0].strip():row.split(",") for row in gpu.splitlines()}
 for j in jobs:
  assert int(rows[j["gpu"]][1])>=6000,gpu
  if d["host"]=="2025":assert float(rows[j["gpu"]][3])==250.,gpu
  assert not native(j).parent.exists() and not Path(j["run"]).exists(),j["task"]
 output["gpu_before"]=gpu
 launches=[]
 for job in jobs:
  argv=[job["command"][0],str(root/"scripts/run_task_four_gpu.py"),"--task",job["task"],"--gpu",job["gpu"],"--"]+command(job)
  log=root/request["remote_folder"]/(job["task"]+"_wrapper.log")
  with log.open("xb") as f:
   child=subprocess.Popen(argv,cwd=str(root),start_new_session=True,stdout=f,stderr=subprocess.STDOUT)
  launches.append(dict(task=job["task"],wrapper_pid=child.pid,gpu=job["gpu"],command=argv,run=job["run"]))
 output["launches"]=launches
 path=root/request["remote_folder"]/(action+"_launch.json")
 assert not path.exists();path.write_text(json.dumps(output,indent=2)+"\n")
else:
 output["tasks"]=[inspect(j) for j in [d["smoke"]]+d["jobs"]]
 output["gpu"]=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.limit","--format=csv,noheader"],universal_newlines=True)
 output["compute"]=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],universal_newlines=True)
print(json.dumps(output,indent=2))
'''
request = dict(draft=draft, review_sha=review_sha, action=args.action,
               remote_folder=remote_folder, draft_sha=hashlib.sha256(draft_raw).hexdigest())
interpreter = draft["smoke"]["command"][0]
result = subprocess.run(ssh + [interpreter, "-B", "-c", shlex.quote(REMOTE)],
                        input=json.dumps(request), stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, universal_newlines=True)
print(result.stderr, end="", flush=True)
print(result.stdout, end="", flush=True)
result.check_returncode()
if args.action != "inspect":
    with (folder / (args.action + "_receipt.json")).open("x") as handle:
        handle.write(result.stdout)
