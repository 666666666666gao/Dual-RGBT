"""Deploy only exact admitted full plans and launch one C2 job per free GPU."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import base64,hashlib,importlib.util,json,shutil,subprocess,sys
from pathlib import Path
request=json.loads(sys.stdin.read())
root=Path(request["root"])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,digest in request["sources"].items():
    assert sha(root/name)==digest,name
for item in request["files"].values():
    assert hashlib.sha256(base64.b64decode(item["raw"])).hexdigest()==item["sha256"]
for name in request["files"]:
    assert not (root/name).exists(),name
for plan in request["plans"]:
    for item in plan["inputs"].values():
        assert sha(item["path"])==item["sha256"]
    for variant in ("zero","film","shared","routed"):
        smoke_task="smoke_c2_%s_%s_r2_p%s_20261103"%(
            request["branch"],variant,plan["host"][-2:])
        native=json.loads((root/"runs/pipeline_tasks"/smoke_task/"status.json").read_bytes())
        assert native["state"]=="COMPLETE" and native["exit_code"]==0
        assert not Path("/proc/%d"%native["pid"]).exists()
    for variant in ("zero","film","shared","routed"):
        full_task="train_c2_%s_%s_full_r1_p%s_20261103"%(
            request["branch"],variant,plan["host"][-2:])
        native_path=root/"runs/pipeline_tasks"/full_task/"status.json"
        if native_path.exists():
            native=json.loads(native_path.read_bytes())
            if native["physical_gpu"]==plan["gpu"]:
                assert native["state"]=="COMPLETE" and native["exit_code"]==0
                assert not Path("/proc/%d"%native["pid"]).exists()
assert len({plan["gpu"] for plan in request["plans"]})==len(request["plans"])
rows=subprocess.check_output(
    ["nvidia-smi","--query-gpu=index,memory.free,power.limit",
     "--format=csv,noheader,nounits"],universal_newlines=True)
gpu={row.split(",")[0].strip():row.split(",") for row in rows.splitlines()}
for plan in request["plans"]:
    assert int(gpu[plan["gpu"]][1])>=20480,rows
    if plan["host"]=="2025":assert float(gpu[plan["gpu"]][2])==250.
    assert shutil.disk_usage(Path(plan["run"]).parent).free>=12*2**30
    assert not Path(plan["run"]).exists()
    assert not (root/"runs/pipeline_tasks"/plan["task"]).exists()
for name,item in request["files"].items():
    path=root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(base64.b64decode(item["raw"]))
spec=importlib.util.spec_from_file_location("c2_full_entry",str(root/"scripts/train_c2_executor.py"))
entry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
for plan in request["plans"]:
    assert entry.load_plan(plan["command"][-1])==plan
print(json.dumps(dict(status="CANONICAL_FULL_ADMISSION_PASS",gpu=rows)),flush=True)
launches=[]
for plan in request["plans"]:
    argv=[plan["command"][0],str(root/"scripts/run_task_four_gpu.py"),
          "--task",plan["task"],"--gpu",plan["gpu"],"--"]+plan["command"]
    log=root/"reports"/(plan["task"]+"_wrapper.log")
    with log.open("xb") as handle:
        child=subprocess.Popen(argv,cwd=str(root),stdin=subprocess.DEVNULL,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    launches.append(dict(task=plan["task"],gpu=plan["gpu"],wrapper_pid=child.pid))
print(json.dumps(dict(status="LAUNCHED",launches=launches)),flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("--variants", nargs="+", required=True,
                        choices=("zero", "film", "shared", "routed"))
    args = parser.parse_args()
    assert len(args.variants) == len(set(args.variants))
    folder = ROOT / "reports/c2_training_full_r1_20261103" / args.branch
    plans = [json.loads((folder / variant / "plan.json").read_bytes())
             for variant in args.variants]
    first = plans[0]
    review_name = PurePosixPath(first["review"]["path"]).name
    review_raw = (ROOT / "reports" / review_name).read_bytes()
    review_sha = hashlib.sha256(review_raw).hexdigest()
    review = json.loads(review_raw)
    assert review["stage"] == first["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == first["sources"]
    files = {"reports/" + review_name: review_raw}
    for plan in plans:
        assert plan["launch_allowed"] is False and plan["smoke"] is False
        assert plan["sources"] == first["sources"] and plan["root"] == first["root"]
        binding = {key: value for key, value in plan.items()
                   if key not in ("launch_allowed", "review")}
        digest = hashlib.sha256(json.dumps(
            binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert digest in review["admitted_plan_bindings"]
        plan["launch_allowed"] = True
        plan["review"]["sha256"] = review_sha
        relative = PurePosixPath(plan["command"][-1]).relative_to(
            PurePosixPath(plan["root"])).as_posix()
        files[relative] = (json.dumps(plan, indent=2) + "\n").encode()
    request = dict(branch=args.branch, root=first["root"], sources=first["sources"],
                   plans=plans, files={name: dict(
                       raw=base64.b64encode(raw).decode(),
                       sha256=hashlib.sha256(raw).hexdigest())
                       for name, raw in files.items()})
    log = folder / ("launch_" + "_".join(args.variants) + ".log")
    assert not log.exists()
    response = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         first["host"], first["command"][0], "-B", "-c", shlex.quote(REMOTE)],
        input=json.dumps(request), encoding="utf-8", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    log.write_text(response.stdout)
    print(response.stdout, flush=True)
    response.check_returncode()


if __name__ == "__main__":
    main()
