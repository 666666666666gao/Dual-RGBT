"""Deploy an exactly admitted full FIT plan and launch its native wrapper once."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import base64, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
request = json.loads(sys.stdin.read())
plan = request["plan"]
root = Path(plan["root"])
os.chdir(str(root))
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
for name, digest in plan["sources"].items():
    assert sha(root / name) == digest, name
for key, item in plan["inputs"].items():
    assert sha(item["path"]) == item["sha256"], key
acceptance = plan["collection_acceptance"]
assert sha(acceptance["path"]) == acceptance["sha256"]
accepted = json.loads(Path(acceptance["path"]).read_text())
assert accepted["status"] == "PASS" and accepted["sequences"] == 882
assert accepted["records"] == 91091 and accepted["cuda_initialized"] is False
assert accepted["stage"] == "P1B_direct_E0_full_collection_acceptance"
for key in ("host", "root", "branch", "round", "inputs", "sources",
            "label_stride", "collections"):
    assert accepted[key] == plan[key], key
for collection in plan["collections"]:
    original = collection["plan"]
    native = json.loads((root / "runs/pipeline_tasks" / original["task"] /
                         "status.json").read_text())
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["command"] == original["command"]
assert not Path(plan["run"]).exists()
assert not (root / "runs/pipeline_tasks" / plan["task"]).exists()
free_mib = int(subprocess.check_output([
    "nvidia-smi", "--id=" + plan["gpu"], "--query-gpu=memory.free",
    "--format=csv,noheader,nounits"], universal_newlines=True).strip())
assert free_mib >= 20000
assert shutil.disk_usage(Path(plan["run"]).parent).free >= 3 * 2**30
for name in request["files"]:
    assert not (root / name).exists(), name
for name, item in request["files"].items():
    raw = base64.b64decode(item["raw"])
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    assert sha(path) == item["sha256"]
sys.path.insert(0, str(root))
from dual_rgbt.e0_policy_refresh import load_plan
admitted = load_plan(request["plan_path"], "fit")
assert admitted == plan
review = json.loads(Path(plan["review"]["path"]).read_text())
assert review["supplemental_sources"] == plan["supplemental_sources"]
for name, digest in plan["supplemental_sources"].items():
    assert sha(root / name) == digest, name
command = [plan["command"][0], str(root / "scripts/run_task_four_gpu.py"),
           "--task", plan["task"], "--gpu", plan["gpu"], "--"] + plan["command"]
print(json.dumps(dict(admission="PASS", free_mib=free_mib,
                      command=command)), flush=True)
child = subprocess.Popen(command, cwd=str(root), stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
print(json.dumps(dict(launched=True, task=plan["task"], wrapper_pid=child.pid)), flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    args = parser.parse_args()
    folder = ROOT / "reports/p1b_full_fit_r1_20261102" / args.branch
    plan = json.loads((folder / "draft/plan.json").read_bytes())
    assert plan["launch_allowed"] is False
    review_name = PurePosixPath(plan["review"]["path"]).name
    review_raw = (ROOT / "reports" / review_name).read_bytes()
    review = json.loads(review_raw)
    assert review["stage"] == "P1B_direct_E0"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == plan["sources"]
    assert review["supplemental_sources"] == plan["supplemental_sources"]
    binding = {key: value for key, value in plan.items()
               if key not in ("review", "launch_allowed")}
    digest = hashlib.sha256(json.dumps(
        binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest in review["admitted_plan_bindings"]
    plan["review"]["sha256"] = hashlib.sha256(review_raw).hexdigest()
    plan["launch_allowed"] = True
    final = folder / "admitted"
    final.mkdir(exist_ok=False)
    plan_raw = (json.dumps(plan, indent=2) + "\n").encode()
    (final / "plan.json").write_bytes(plan_raw)
    plan_path = plan["command"][-1]
    assert plan["command"][-2] == "--plan"
    relative = PurePosixPath(plan_path).relative_to(PurePosixPath(plan["root"])).as_posix()
    files = {relative: plan_raw, "reports/" + review_name: review_raw}
    for name, expected in plan["supplemental_sources"].items():
        raw = (ROOT / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        files[name] = raw
    request = dict(plan=plan, plan_path=plan_path, files={
        name: dict(raw=base64.b64encode(raw).decode(),
                   sha256=hashlib.sha256(raw).hexdigest())
        for name, raw in files.items()})
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         plan["host"], plan["command"][0], "-c", shlex.quote(REMOTE)],
        input=json.dumps(request), encoding="utf-8", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (final / "launch.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    result.check_returncode()


if __name__ == "__main__":
    main()
