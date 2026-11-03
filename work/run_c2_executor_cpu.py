"""Deploy exact reviewed C2 files to a new CPU-only working copy on P25."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVIEW = "reports/P2_C2_EXECUTOR_CPU_SOURCE_REVIEW_R2_20261103.json"
FILES = ("dual_rgbt/c2_executor.py", "tools/check_c2_executor_cpu.py",
         "dual_rgbt/route_lora.py", "dual_rgbt/missing_modality.py")
REMOTE = r'''
import base64, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
request = json.loads(sys.stdin.read())
source = Path(request["snapshot"])
destination = Path(request["root"])
assert not destination.exists()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
original_plan = json.loads((source / request["plan"]).read_text())
assert len(original_plan["sources"]) == 98
for name, digest in original_plan["sources"].items():
    assert sha(source / name) == digest, name
for name, digest in request["reviewed_dependencies"].items():
    assert original_plan["sources"][name] == digest, name
for key in ("factory", "executor", "manifest"):
    item = original_plan["inputs"][key]
    assert sha(item["path"]) == item["sha256"]
assert all(not (source / name).exists() for name in request["files"])
destination.mkdir(parents=True, exist_ok=False)
for name in original_plan["sources"]:
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / name, target)
for name, item in request["files"].items():
    raw = base64.b64decode(item["raw"])
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    assert sha(target) == item["sha256"]
review = base64.b64decode(request["review"])
assert hashlib.sha256(review).hexdigest() == request["review_sha256"]
(destination / "source_review.json").write_bytes(review)
inputs = original_plan["inputs"]
command = [request["python"], "-B", str(destination / "tools/check_c2_executor_cpu.py"),
           "--factory", inputs["factory"]["path"],
           "--executor", inputs["executor"]["path"],
           "--manifest", inputs["manifest"]["path"],
           "--output", str(destination / "result.json")]
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   OPENCV_FOR_THREADS_NUM="1", PYTHONDONTWRITEBYTECODE="1")
binding = dict(command=command, original_sources=original_plan["sources"],
               files={name:item["sha256"] for name,item in request["files"].items()},
               review_sha256=request["review_sha256"],
               inputs={key:inputs[key] for key in ("factory", "executor", "manifest")},
               cuda_visible_devices="", gpu_execution=False)
(destination / "invocation.json").write_text(json.dumps(binding, indent=2) + "\n")
print(json.dumps(dict(status="CPU_CHECK_STARTING", root=str(destination),
                      command=command)), flush=True)
with (destination / "run.log").open("wb") as log:
    result = subprocess.run(command, cwd=str(destination), env=environment,
                            stdout=log, stderr=subprocess.STDOUT)
for name, digest in original_plan["sources"].items():
    assert sha(source / name) == digest, name
print(json.dumps(dict(exit_code=result.returncode,
                      log_tail=(destination / "run.log").read_text()[-4000:])), flush=True)
sys.exit(result.returncode)
'''


def main():
    review_raw = (ROOT / REVIEW).read_bytes()
    review = json.loads(review_raw)
    assert review["stage"] == "P2_C2_executor_cpu_check"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["isolated_CPU_payload_assessment"]["source_suitable_for_isolated_CPU_deployment"]
    source_map = review["source_sha256"]
    payload = {}
    for name in FILES:
        raw = (ROOT / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert source_map[name] == digest, name
        payload[name] = dict(raw=base64.b64encode(raw).decode(), sha256=digest)
    request = dict(
        snapshot="/home/gaob/Dual-RGBT/snapshots/p1b_dropmoe_p25_r1_20261102",
        root="/data2/gaob/Dual-RGBT/c2_cpu_check_r2_20261103",
        plan="reports/p1b_dropmoe_full_collection_r1_20261102/s0/plan.json",
        python="/home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python",
        files=payload, review=base64.b64encode(review_raw).decode(),
        reviewed_dependencies={
            name: digest for name, digest in source_map.items()
            if name not in FILES and not name.startswith("tests/")},
        review_sha256=hashlib.sha256(review_raw).hexdigest())
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         "2025", request["python"], "-c", shlex.quote(REMOTE)],
        input=json.dumps(request), encoding="utf-8")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
