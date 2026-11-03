"""Deploy an admitted isolated C2 working copy and run explicit TRAIN smokes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import base64, hashlib, importlib.util, json, shutil, subprocess, sys
from pathlib import Path
request = json.loads(sys.stdin.read())
root = Path(request["root"])
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
if request["action"] == "deploy":
    assert not root.exists()
    snapshot = Path(request["snapshot"])
    original = json.loads((snapshot / request["collection_plan"]).read_text())
    assert len(original["sources"]) == 98
    for name, digest in original["sources"].items():
        assert sha(snapshot / name) == digest
        assert request["sources"][name] == digest
    for item in request["original_inputs"].values():
        assert sha(item["path"]) == item["sha256"]
    root.mkdir(parents=True, exist_ok=False)
    for name in original["sources"]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / name, target)
    for name, item in request["files"].items():
        raw = base64.b64decode(item["raw"])
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        path = root / name
        assert not path.exists(), name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    assert not __import__("torch").cuda.is_initialized()
for name, digest in request["sources"].items():
    assert sha(root / name) == digest, name
assert sha(root / request["review_path"]) == request["review_sha256"]
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location(
    "c2_train_entry", str(root / "scripts/train_c2_executor.py"))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
plans = [entry.load_plan(str(root / item)) for item in request["plans"]]
assert all(plan["smoke"] is True for plan in plans)
if request["action"] == "smoke":
    assert len({plan["gpu"] for plan in plans}) == len(plans)
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,power.limit",
         "--format=csv,noheader,nounits"], universal_newlines=True)
    gpu = {row.split(",")[0].strip(): row.split(",") for row in rows.splitlines()}
    for plan in plans:
        assert int(gpu[plan["gpu"]][1]) >= 20000, rows
        if plan["host"] == "2025":
            assert float(gpu[plan["gpu"]][2]) == 250., rows
        assert shutil.disk_usage(Path(plan["run"]).parent.parent).free >= 3 * 2**30
        assert not Path(plan["run"]).exists()
        assert not (root / "runs/pipeline_tasks" / plan["task"]).exists()
    launches = []
    for plan in plans:
        argv = [plan["command"][0], str(root / "scripts/run_task_four_gpu.py"),
                "--task", plan["task"], "--gpu", plan["gpu"], "--"] + plan["command"]
        log = root / "reports" / (plan["task"] + "_wrapper.log")
        with log.open("xb") as handle:
            child = subprocess.Popen(argv, cwd=str(root), stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        launches.append(dict(task=plan["task"], wrapper_pid=child.pid, gpu=plan["gpu"]))
    print(json.dumps(dict(status="LAUNCHED", launches=launches)), flush=True)
elif request["action"] == "inspect":
    result = []
    for plan in plans:
        path = root / "runs/pipeline_tasks" / plan["task"] / "status.json"
        row = dict(task=plan["task"], native_exists=path.exists())
        if path.exists():
            native = json.loads(path.read_text())
            row["native"] = native
            row["wrapper_live"] = Path("/proc/%d" % native["pid"]).exists()
            row["children"] = subprocess.run(
                ["pgrep", "-P", str(native["pid"])], stdout=subprocess.PIPE,
                universal_newlines=True).stdout.split()
            receipt = Path(plan["run"]) / "SMOKE.json"
            row["smoke"] = json.loads(receipt.read_text()) if receipt.exists() else None
            row["log_tail"] = (path.parent / "run.log").read_text()[-1200:]
        result.append(row)
    print(json.dumps(dict(status="INSPECTED", tasks=result)))
else:
    assert request["action"] == "deploy"
    assert not __import__("torch").cuda.is_initialized()
    print(json.dumps(dict(status="CPU_ADMISSION_PASS", sources=len(request["sources"]),
                          plans=len(plans), cuda_initialized=False)))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("action", choices=("deploy", "smoke", "inspect"))
    parser.add_argument("--variants", nargs="+", choices=("zero", "film", "shared", "routed"),
                        required=True)
    args = parser.parse_args()
    assert len(args.variants) == len(set(args.variants))
    folder = ROOT / "reports/c2_training_smoke_r1_20261103" / args.branch
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
    files = {}
    relative_plans = []
    for plan in plans:
        assert plan["launch_allowed"] is False
        assert plan["sources"] == first["sources"] and plan["root"] == first["root"]
        binding = {key: value for key, value in plan.items()
                   if key not in ("review", "launch_allowed")}
        digest = hashlib.sha256(json.dumps(
            binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert digest in review["admitted_plan_bindings"]
        plan["launch_allowed"] = True
        plan["review"]["sha256"] = review_sha
        relative = PurePosixPath(plan["command"][-1]).relative_to(
            PurePosixPath(plan["root"])).as_posix()
        relative_plans.append(relative)
        files[relative] = (json.dumps(plan, indent=2) + "\n").encode()
    if args.action == "deploy":
        assert set(args.variants) == {"zero", "film", "shared", "routed"}
        metadata = json.loads((folder / "bindings.json").read_bytes())
        for name in metadata["new_sources"]:
            raw = (ROOT / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == first["sources"][name]
            files[name] = raw
        files["evidence/c2_cpu_result_r2.json"] = (
            ROOT / "reports/c2_executor_cpu_r2_20261103/result.json").read_bytes()
        files["reports/" + review_name] = review_raw
    request = dict(
        action=args.action, root=first["root"], snapshot=first["source_snapshot"],
        collection_plan=first["source_collection_plan"], sources=first["sources"],
        original_inputs={key: first["inputs"][key] for key in ("factory", "executor", "manifest")},
        review_path="reports/" + review_name, review_sha256=review_sha,
        plans=relative_plans, files={name: dict(
            raw=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
            for name, raw in files.items()})
    response = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         first["host"], first["command"][0], "-B", "-c", shlex.quote(REMOTE)],
        input=json.dumps(request), encoding="utf-8", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    if args.action != "inspect":
        receipt = folder / (args.action + "_" + "_".join(args.variants) + ".log")
        with receipt.open("x") as handle:
            handle.write(response.stdout)
    print(response.stdout, flush=True)
    response.check_returncode()


if __name__ == "__main__":
    main()
