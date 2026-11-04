"""Read-only remote P1B terminal audit before fixed epoch005 evaluation."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]

REMOTE = r'''
import base64
import hashlib
import json
from pathlib import Path
import socket
import sys
import torch

request = json.loads(sys.stdin.read())
draft = request["draft"]
root = Path(draft["root"])
branch = draft["branch"]
task = "fit_p1b_%s_full_b512_r1_p%s_20261102" % (
    branch, draft["host"][-2:])
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_bytes())
plan_path = root / "reports" / task / "plan.json"
plan = read(plan_path)
run = Path(plan["run"])
native_path = root / "runs/pipeline_tasks" / task / "status.json"
native = read(native_path)
complete = read(run / "COMPLETE.json")
assert plan["stage"] == "P1B_direct_E0" and plan["mode"] == "fit"
assert plan["host"] == draft["host"] and plan["branch"] == branch
assert plan["root"] == str(root) and plan["task"] == task
assert plan["round"] == 1 and plan["smoke"] is False
assert plan["launch_allowed"] is True
assert (plan["epochs"], plan["batch_size"], plan["workers"],
        plan["seed"]) == (5, 512, 4, 2027)
assert plan["preview_lr"] == 4e-5 and plan["quality_gain_lr"] == 4e-4
assert plan == read(run / "config.json")
assert native["state"] == "COMPLETE" and native["exit_code"] == 0
assert native["task"] == task and native["physical_gpu"] == plan["gpu"]
assert native["hostname"] == socket.gethostname()
assert native["command"] == plan["command"]
assert native["command"][-2:] == ["--plan", str(plan_path)]
review_path = Path(plan["review"]["path"])
assert sha(review_path) == plan["review"]["sha256"]
review = read(review_path)
assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
assert review["stage"] == plan["stage"] and review["sources"] == plan["sources"]
binding = hashlib.sha256(json.dumps(
    {key: value for key, value in plan.items()
     if key not in ("launch_allowed", "review")},
    sort_keys=True, separators=(",", ":")).encode()).hexdigest()
assert binding in review["admitted_plan_bindings"]
for name, digest in dict(plan["sources"], **plan["supplemental_sources"]).items():
    assert sha(root / name) == digest, name
for item in plan["inputs"].values():
    assert sha(item["path"]) == item["sha256"], item["path"]
assert read(run / "precision.json") == dict(
    matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)
assert complete["stage"] == "P1B_direct_E0_fit"
assert complete["status"] == "COMPLETE" and complete["smoke"] is False
assert complete["branch"] == branch and complete["closed_loop_round"] == 1
assert complete["plan_sha256"] == sha(plan_path)
assert complete["sources"] == plan["sources"]
assert (complete["epochs"], complete["steps"], complete["records"]) == (5, 890, 91091)
assert complete["executor_preserved"] is True
assert complete["all_policy_modules_changed"] is True
assert complete["strict_reload"] is True
assert complete["benchmark_evidence"] is False
assert complete["executor_sha256"] == plan["inputs"]["executor"]["sha256"]
assert set(complete["artifacts"]) == {
    "epoch_%03d.pth" % epoch for epoch in range(1, 6)}
for name, digest in complete["artifacts"].items():
    assert sha(run / name) == digest, name
torch.set_num_threads(1)
saved = torch.load(run / "epoch_005.pth", map_location="cpu")
original = torch.load(plan["inputs"]["executor"]["path"], map_location="cpu")
assert saved["config"] == plan
assert saved["epoch"] == 5 and saved["step"] == 890
assert saved["executor_sha256"] == plan["inputs"]["executor"]["sha256"]
assert set(saved["net"]) == set(original["net"])
prefixes = ("preview.", "quality.", "gain.")
for name, tensor in saved["net"].items():
    assert torch.isfinite(tensor).all(), name
    if not name.startswith(prefixes):
        assert torch.equal(tensor, original["net"][name]), name
for prefix in prefixes:
    assert any(not torch.equal(tensor, saved["net"][name])
               for name, tensor in original["net"].items()
               if name.startswith(prefix)), prefix
for name, digest in draft["source_sha256"].items():
    assert sha(root / name) == digest, name
files = {}
for name, path in dict(
        plan=plan_path, config=run / "config.json",
        native=native_path, complete=run / "COMPLETE.json",
        precision=run / "precision.json", fit_review=review_path).items():
    raw = path.read_bytes()
    files[name + ".json"] = dict(
        raw=base64.b64encode(raw).decode(),
        sha256=hashlib.sha256(raw).hexdigest(), remote_path=str(path))
print(json.dumps(dict(
    status="P1B_FIXED_EPOCH005_CPU_AUDIT_PASS",
    host=draft["host"], branch=branch, task=task, root=str(root),
    checkpoint=dict(path=str(run / "epoch_005.pth"),
                    sha256=complete["artifacts"]["epoch_005.pth"]),
    factory=plan["inputs"]["factory"],
    source_sha256=draft["source_sha256"], fit_plan_binding=binding,
    cpu_state_finite=True, executor_preserved=True,
    all_policy_modules_changed=True, cuda_initialized=torch.cuda.is_initialized(),
    benchmark_evidence=False, evaluation_admitted=False, files=files)))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    assert not output.exists()
    draft = json.loads((ROOT / "reports/p1a_evaluation_r1_20261102" /
                        args.branch / "draft.json").read_bytes())
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         draft["host"], draft["smoke"]["command"][0], "-B", "-c",
         shlex.quote(REMOTE)],
        input=json.dumps(dict(draft=draft)), text=True, capture_output=True,
        check=True)
    audit = json.loads(result.stdout)
    assert audit["status"] == "P1B_FIXED_EPOCH005_CPU_AUDIT_PASS"
    assert audit["cuda_initialized"] is False
    files = audit.pop("files")
    output.mkdir(parents=True, exist_ok=False)
    for name, item in files.items():
        raw = base64.b64decode(item["raw"])
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        (output / name).write_bytes(raw)
    audit["archived_evidence"] = {
        name: {key: value for key, value in item.items() if key != "raw"}
        for name, item in files.items()}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
