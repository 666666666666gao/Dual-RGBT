"""Copy reviewed CPU-only compute dependencies and verify actual smoke bindings."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DRAFT = "reports/p1a_compute_r1_20261102/draft.json"
draft_raw = (ROOT / DRAFT).read_bytes()
draft = json.loads(draft_raw)
branch = sys.argv[1]
target = next(row for row in draft["targets"] if row["branch"] == branch)
review_raw = (ROOT / draft["review"]).read_bytes()
review = json.loads(review_raw)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
assert review["stage"] == draft["stage"]
assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
assert review["reviewed_draft"] == dict(path=DRAFT, sha256=sha(draft_raw))
assert review["source_sha256"] == {
    draft["tool"]: draft["tool_sha256"],
    draft["transport"]: sha((ROOT / draft["transport"]).read_bytes())}
assert review["targets"] == draft["targets"]
assert review["profile_sha256"] == draft["profile_sha256"]
assert sha((ROOT / draft["tool"]).read_bytes()) == draft["tool_sha256"]
assert sha((ROOT / draft["profile_local"]).read_bytes()) == draft["profile_sha256"]
for name, expected in review["source_sha256"].items():
    assert sha((ROOT / name).read_bytes()) == expected
folder = ROOT / "reports/p1a_compute_r1_20261102" / branch
assert not folder.exists()
folder.mkdir()
request = dict(draft=draft, target=target, review_sha256=sha(review_raw),
               draft_sha256=sha(draft_raw))
ssh = ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", target["host"]]
CHECK = r'''
import hashlib,json,sys
from pathlib import Path
r=json.loads(sys.stdin.read());d=r["draft"];root=Path(r["target"]["root"])
expected={d["tool"]:d["tool_sha256"],d["profile_target"]:d["profile_sha256"]}
missing=[]
for name,digest in expected.items():
 p=root/name
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
 else:missing.append(name)
assert not (root/d["review"]).exists()
assert not (root/"reports/p1a_compute_r1_20261102/draft.json").exists()
print(json.dumps(missing))
'''
result = subprocess.run(ssh + [target["python"], "-B", "-c", shlex.quote(CHECK)],
                        input=json.dumps(request), stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, universal_newlines=True, check=True)
missing = json.loads(result.stdout)
payload = io.BytesIO()
with tarfile.open(fileobj=payload, mode="w") as archive:
    for name in missing:
        source = draft["profile_local"] if name == draft["profile_target"] else name
        archive.add(str(ROOT / source), arcname=name)
    for name in (draft["review"], DRAFT):
        archive.add(str(ROOT / name), arcname=name)
subprocess.run(ssh + ["tar", "-C", target["root"], "-xf", "-"],
               input=payload.getvalue(), check=True)
VERIFY = r'''
import hashlib,importlib.util,json,os,sys
from pathlib import Path
r=json.loads(sys.stdin.read());d=r["draft"];t=r["target"];root=Path(t["root"])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
assert sha(root/"reports/p1a_compute_r1_20261102/draft.json")==r["draft_sha256"]
assert sha(root/d["review"])==r["review_sha256"]
assert sha(root/d["tool"])==d["tool_sha256"]
assert sha(root/d["profile_target"])==d["profile_sha256"]
assert sha(root/t["evaluation_draft"])==t["evaluation_draft_sha256"]
assert sha(root/t["evaluation_review"])==t["evaluation_review_sha256"]
evaluation=read(root/t["evaluation_review"])
assert evaluation["stage"]=="E0_finetuned_evaluation"
assert evaluation["status"]=="SOURCE_PASS" and evaluation["blocking_issues"]==0
assert all(sha(root/k)==v for k,v in evaluation["source_sha256"].items())
profile=read(root/d["profile_target"]);run=Path(t["smoke_run"])
config=read(run/"config.json");receipt=read(run/"SMOKE.json")
native=read(config["native_status"])
assert native["state"]=="COMPLETE" and native["exit_code"]==0
assert native["command"]==config["native_command"]
assert native["task"]==receipt["task"]==config["task"]
assert config["source_sha256"]==evaluation["source_sha256"]
assert config["source_review_sha256"]==t["evaluation_review_sha256"]
assert receipt["frames"]==11 and receipt["sequences"]==1 and receipt["smoke_only"]
assert receipt["state_unchanged"] and receipt["optimizer_updates"]==receipt["backward_calls"]==0
assert sha(run/"config.json")==t["smoke_config_sha256"]==receipt["config_sha256"]
assert config["total_parameters"]==profile["parameter_counts"]["e0"]==d["total_parameters"]
assert profile["status"]==d["profile_stage"] and profile["model_unchanged"]
assert profile["cuda_initialized"] is False
assert len(profile["routes"])==10
assert all(row["independent_formula_match"] for row in profile["routes"])
spec=importlib.util.spec_from_file_location("p1_actual_compute",root/d["tool"])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
common=module.bind_sources(profile,config)
result=dict(status="CPU_COMPUTE_BINDING_READY",branch=t["branch"],root=str(root),
 source_sha256=d["tool_sha256"],profile_sha256=d["profile_sha256"],
 source_review_sha256=r["review_sha256"],evaluation_sources_verified=49,
 common_source_paths=common,parameters=config["total_parameters"],
 actual_smoke_verified=True,full_metrics_computed=False,scorer_rerun=False,
 gpu_allocated=False,test_eligible=False)
output=root/("reports/p1a_compute_r1_20261102/"+t["branch"]+"_readiness.json")
assert not output.exists();output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
'''
result = subprocess.run(ssh + [target["python"], "-B", "-c", shlex.quote(VERIFY)],
                        input=json.dumps(request), stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, universal_newlines=True)
(folder / "run.log").write_text(result.stderr + result.stdout)
print(result.stderr + result.stdout, end="", flush=True)
result.check_returncode()
(folder / "readiness.json").write_text(result.stdout)
