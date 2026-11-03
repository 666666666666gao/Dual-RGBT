"""Deploy the exact reviewed supplemental capacity tool and launch once."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVIEW = "reports/E0_EXECUTOR_V2_P1B_CAPACITY_ADMISSION_R3_20261102.json"
REMOTE = r'''
import hashlib, json, os, subprocess, sys
from pathlib import Path
spec_path = Path(sys.argv[1])
spec = json.loads(spec_path.read_text())
root = Path(spec["root"])
os.chdir(str(root))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(spec["review"]["path"]) == spec["review"]["sha256"]
review = json.loads(Path(spec["review"]["path"]).read_text())
assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
assert spec["launch_allowed"] is True
payload = {k:v for k,v in spec.items() if k not in ("review","launch_allowed")}
digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
assert digest in review["admitted_spec_bindings"]
assert sha(spec["command"][1]) == spec["tool_sha256"]
sys.path.insert(0,str(root))
from dual_rgbt.e0_policy_refresh import load_plan
plan = load_plan(spec["collection_plan"]["path"], "collect")
assert sha(spec["collection_plan"]["path"]) == spec["collection_plan"]["sha256"]
assert len(plan["sources"]) == 98 and len(plan["inputs"]) == 7
for i in range(4):
 name = "collect_p1b_%s_full_s%d_r1_p%s_20261102" % (spec["branch"], i, spec["host"][-2:])
 native = json.loads((root/"runs/pipeline_tasks"/name/"status.json").read_text())
 assert native["state"] == "RUNNING"
 assert Path("/proc/%d"%native["pid"]).exists()
 children = subprocess.check_output(["pgrep","-P",str(native["pid"])],universal_newlines=True).strip()
 assert children
 print(name, "live", children, flush=True)
assert not (root/"runs/pipeline_tasks"/spec["task"]).exists()
assert not Path(spec["output"]).exists()
free = int(subprocess.check_output(["nvidia-smi","--id=3","--query-gpu=memory.free","--format=csv,noheader,nounits"],universal_newlines=True).strip())
assert free >= spec["minimum_free_mib"]
command = [spec["command"][0], str(root/"scripts/run_task_four_gpu.py"),
           "--task",spec["task"],"--gpu",spec["gpu"],"--"] + spec["command"]
print(json.dumps(dict(preflight="PASS",free_mib=free,binding=digest,command=command)),flush=True)
child = subprocess.Popen(command,cwd=str(root),stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(launched=True,wrapper_pid=child.pid,task=spec["task"])),flush=True)
'''

for branch, host in (("dropmoe", "2025"), ("sot", "2026")):
    local_spec = ROOT / ("reports/p1b_capacity_r3_20261102/%s/spec.json" % branch)
    spec = json.loads(local_spec.read_text())
    review = json.loads((ROOT / REVIEW).read_text())
    assert hashlib.sha256((ROOT / REVIEW).read_bytes()).hexdigest() == spec["review"]["sha256"]
    tool = ROOT / "tools/measure_e0_policy_refresh_capacity.py"
    assert hashlib.sha256(tool.read_bytes()).hexdigest() == spec["tool_sha256"]
    ssh = ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", host]
    remote_spec = spec["command"][-1]
    subprocess.run(ssh + ["mkdir", "-p", str(Path(remote_spec).parent).replace("\\", "/")], check=True)
    for source, target in ((tool, spec["command"][1]), (ROOT / REVIEW, spec["review"]["path"]),
                           (local_spec, remote_spec)):
        subprocess.run(["scp", "-q", "-o", "ClearAllForwardings=yes", str(source),
                        host + ":" + target], check=True)
    result = subprocess.run(ssh + [spec["command"][0], "-", remote_spec], input=REMOTE,
                            encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (local_spec.parent / "launch.log").write_text(result.stdout)
    print(host, result.stdout, flush=True)
    result.check_returncode()
