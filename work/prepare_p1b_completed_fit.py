"""Accept four completed collectors and prepare a disabled local FIT draft."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_e0_policy_refresh_scaled_fit import prepare_scaled

REVIEW = "reports/E0_EXECUTOR_V2_P1B_COLLECTION_ACCEPTANCE_TOOL_SOURCE_REVIEW_R1_20261102.json"
VERIFIER = "tools/verify_e0_policy_refresh_collection.py"
REMOTE = r'''
import base64, hashlib, json, subprocess, sys
from pathlib import Path
request = json.loads(sys.stdin.read())
root = Path(request["root"])
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(root / request["verifier"]) == request["verifier_sha256"]
review_path = root / request["review"]
assert sha(review_path) == request["review_sha256"]
review = json.loads(review_path.read_text())
assert review["stage"] == "P1B_direct_E0_collection_acceptance_tool"
assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
assert review["sources"][request["verifier"]] == request["verifier_sha256"]
assert any(row["host"] == request["host"] and row["root"] == str(root)
           for row in review["target_snapshots"])
plans, statuses, archive = [], [], {}
for shard in range(4):
    path = root / ("reports/p1b_%s_full_collection_r1_20261102/s%d/plan.json" %
                   (request["branch"], shard))
    plan = json.loads(path.read_text())
    native_path = root / "runs/pipeline_tasks" / plan["task"] / "status.json"
    raw = native_path.read_bytes()
    native = json.loads(raw)
    assert native["task"] == plan["task"] and native["command"] == plan["command"]
    statuses.append(dict(task=plan["task"], state=native["state"],
                         exit_code=native.get("exit_code"),
                         wrapper_live=Path("/proc/%d" % native["pid"]).exists()))
    plans.append(str(path))
    archive["native_s%d.json" % shard] = base64.b64encode(raw).decode()
if not all(row["state"] == "COMPLETE" and row["exit_code"] == 0 for row in statuses):
    print(json.dumps(dict(status="NOT_READY", tasks=statuses)))
    sys.exit(0)
output = Path(request["acceptance"])
assert not output.exists()
assert not (root / "runs/pipeline_tasks" / request["fit_task"]).exists()
command = [request["python"], str(root / request["verifier"]),
           "--plans"] + plans + ["--output", str(output)]
result = subprocess.run(command, cwd=str(root), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
assert result.returncode == 0, result.stdout.decode()
raw = output.read_bytes()
accepted = json.loads(raw)
assert accepted["status"] == "PASS" and accepted["sequences"] == 882
assert accepted["records"] == 91091 and accepted["tracked_frames"] == 458553
assert accepted["cuda_initialized"] is False and accepted["full_fit_admitted"] is False
assert accepted["verifier_sha256"] == request["verifier_sha256"]
archive["acceptance.json"] = base64.b64encode(raw).decode()
archive["verifier.log"] = base64.b64encode(result.stdout).decode()
print(json.dumps(dict(status="ACCEPTED", acceptance_sha256=sha(output),
                     files=archive)))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    args = parser.parse_args()
    branch = args.branch
    host = "2025" if branch == "dropmoe" else "2026"
    parent = "/home/gaob/Dual-RGBT" if host == "2025" else "/data2/gb/Dual-RGBT"
    snapshot = ("p1b_dropmoe_p25_r1_20261102" if host == "2025"
                else "p1b_direct_r3_20261102")
    root = parent + "/snapshots/" + snapshot
    folder = ROOT / "reports/p1b_full_fit_r1_20261102" / branch
    assert not folder.exists()
    review_raw = (ROOT / REVIEW).read_bytes()
    review = json.loads(review_raw)
    verifier_sha = hashlib.sha256((ROOT / VERIFIER).read_bytes()).hexdigest()
    assert review["sources"][VERIFIER] == verifier_sha
    task = "fit_p1b_%s_full_b512_r1_p%s_20261102" % (branch, host[-2:])
    acceptance = root + "/reports/p1b_full_collection_acceptance_%s_r1_20261102.json" % branch
    request = dict(
        host=host, root=root, branch=branch, python=parent + "/conda-envs/dual_rgbt/bin/python",
        review=REVIEW, review_sha256=hashlib.sha256(review_raw).hexdigest(),
        verifier=VERIFIER, verifier_sha256=verifier_sha,
        acceptance=acceptance, fit_task=task)
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", host,
         request["python"], "-c", shlex.quote(REMOTE)],
        input=json.dumps(request), encoding="utf-8", stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True)
    report = json.loads(result.stdout)
    if report["status"] == "NOT_READY":
        print(json.dumps(dict(branch=branch, **report)))
        return
    assert report["status"] == "ACCEPTED"
    folder.mkdir(parents=True, exist_ok=False)
    for name, payload in report["files"].items():
        (folder / name).write_bytes(base64.b64decode(payload))
    local_acceptance = folder / "acceptance.json"
    assert hashlib.sha256(local_acceptance.read_bytes()).hexdigest() == report["acceptance_sha256"]
    plan, binding = prepare_scaled(
        str(local_acceptance), acceptance, task, "3", 512, 4)
    draft = folder / "draft"
    draft.mkdir()
    (draft / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (draft / "binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    print(json.dumps(dict(branch=branch, status="PREPARED_DISABLED",
                          acceptance_sha256=report["acceptance_sha256"], **binding)))


if __name__ == "__main__":
    main()
