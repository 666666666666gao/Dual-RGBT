"""Run the reviewed C2 CPU scorer and retrieve a single completed archive."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRY = "work/score_c2_policy_completed_remote.py"
SCORER = "tools/score_c2_policy_completed.py"
CPU = {
    "2025": "/home/gaob/Dual-RGBT/conda-envs/cpu-scoring/bin/python",
    "2026": "/data2/gb/Dual-RGBT/conda-envs/cpu-scoring/bin/python",
}
FIT_NAMES = {
    "plan", "config", "native", "complete", "precision",
    "review", "acceptance", "replay_native",
}
NAMES = {
    "score.json", "curves.npz", "scoring_log.log", "plan.json", "review.json",
    "native.json", "config.json", "complete.json", "precision.json", "protocol.json",
} | {"fit_" + name + ".json" for name in FIT_NAMES}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def binding(plan):
    return hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items()
         if key not in ("launch_allowed", "review")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def authenticate(plan, review):
    assert plan["stage"] == review["stage"] == "P2_C2_policy_evaluation"
    assert plan["launch_allowed"] is True
    assert plan["host"] in CPU and plan["scoring_interpreter"] == CPU[plan["host"]]
    assert plan["smoke"] is False and plan["smoke_frames"] == 0
    assert plan["cost_weight"] == .05
    assert plan["exploratory_test"] is True
    assert plan["test_eligible"] is plan["dev_enabled"] is False
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == plan["sources"]
    assert {ENTRY, SCORER}.issubset(plan["sources"])
    assert binding(plan) in review["admitted_plan_bindings"]
    assert set(plan["fit_evidence"]) == FIT_NAMES
    output, run = Path(plan["score_output"]).resolve(), Path(plan["run"]).resolve()
    assert output != run and run not in output.parents
    assert output.suffix == ".json"


def remote_score(request):
    plan_path = Path(request["plan_path"])
    assert sha(plan_path) == request["plan_sha256"]
    plan = read(plan_path)
    assert plan["host"] == request["host"]
    assert sha(plan["review"]["path"]) == plan["review"]["sha256"] == request["review_sha256"]
    review = read(plan["review"]["path"])
    authenticate(plan, review)
    root, run = Path(plan["root"]), Path(plan["run"])
    assert root.resolve() == ROOT.resolve()
    assert Path(sys.executable).resolve() == Path(plan["scoring_interpreter"]).resolve()
    for name, digest in plan["sources"].items():
        assert sha(root / name) == digest, name
    native_path = root / "runs/pipeline_tasks" / plan["task"] / "status.json"
    native = read(native_path)
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == plan["task"]
    assert native["command"] == plan["command"]
    assert native["physical_gpu"] == plan["gpu"]
    assert not Path("/proc", str(native["pid"])).exists()
    output = Path(plan["score_output"]).resolve()
    curves = output.with_suffix(".npz")
    log = output.with_suffix(".scoring.log")
    existed = output.exists()
    if existed:
        assert curves.is_file() and log.is_file()
    else:
        assert not curves.exists() and not log.exists()
    result = subprocess.run([
        plan["scoring_interpreter"], "-B", str(root / SCORER),
        "--plan", str(plan_path), "--run", str(run),
        "--toolkit", plan["toolkit"], "--protocol", plan["protocol"]["path"],
        "--protocol-sha256", plan["protocol"]["sha256"], "--output", str(output),
    ], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not existed:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("xb") as handle:
            handle.write(result.stdout)
    assert result.returncode == 0, result.stdout.decode(errors="replace")[-4000:]
    score = read(output)
    assert score["status"] == "SCORED_COMPLETE" and score["task"] == plan["task"]
    assert score["evaluation_plan_sha256"] == request["plan_sha256"]
    assert score["source_review_sha256"] == request["review_sha256"]
    assert score["native_status_sha256"] == sha(native_path)
    assert score["curves_sha256"] == sha(curves)
    paths = {
        "score.json": output, "curves.npz": curves, "scoring_log.log": log,
        "plan.json": plan_path, "review.json": Path(plan["review"]["path"]),
        "native.json": native_path, "config.json": run / "config.json",
        "complete.json": run / "COMPLETE.json", "precision.json": run / "precision.json",
        "protocol.json": Path(plan["protocol"]["path"]),
    }
    paths.update({"fit_" + name + ".json": Path(item["path"])
                  for name, item in plan["fit_evidence"].items()})
    files = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        files[name] = dict(raw=base64.b64encode(raw).decode(),
                           sha256=hashlib.sha256(raw).hexdigest())
    return dict(
        status="ARCHIVED_C2_POLICY_SCORED_COMPLETE", task=plan["task"],
        host=plan["host"], variant=plan["variant"], benchmark=plan["benchmark"],
        evaluation_plan_sha256=request["plan_sha256"], scoring_reused=existed,
        helper_sha256=sha(root / ENTRY), files=files,
    )


def prepare_request(plan_path, review_path):
    plan, review = read(plan_path), read(review_path)
    assert sha(review_path) == plan["review"]["sha256"]
    authenticate(plan, review)
    for name in (ENTRY, SCORER):
        assert sha(ROOT / name) == plan["sources"][name], name
    assert plan["command"][-1] == plan["plan_path"]
    return plan, dict(
        host=plan["host"], plan_path=plan["plan_path"],
        plan_sha256=sha(plan_path), review_sha256=sha(review_path),
    )


def archive_result(response, request, destination):
    assert not destination.exists()
    assert response["status"] == "ARCHIVED_C2_POLICY_SCORED_COMPLETE"
    assert response["host"] == request["host"]
    assert response["evaluation_plan_sha256"] == request["plan_sha256"]
    assert response["helper_sha256"] == sha(ROOT / ENTRY)
    assert set(response["files"]) == NAMES
    files = {}
    for name, item in response["files"].items():
        raw = base64.b64decode(item["raw"], validate=True)
        assert hashlib.sha256(raw).hexdigest() == item["sha256"], name
        files[name] = raw
    assert hashlib.sha256(files["plan.json"]).hexdigest() == request["plan_sha256"]
    assert hashlib.sha256(files["review.json"]).hexdigest() == request["review_sha256"]
    score, plan = json.loads(files["score.json"]), json.loads(files["plan.json"])
    assert score["task"] == response["task"] == plan["task"]
    assert score["benchmark"] == response["benchmark"] == plan["benchmark"]
    assert score["variant"] == response["variant"] == plan["variant"]
    audit = {key: value for key, value in response.items() if key != "files"}
    metrics = ("PR", "NPR", "SR") if plan["benchmark"] == "LasHeR" else (
        ("PR", "SR") if plan["benchmark"] == "RGBT210" else ("MPR", "MSR"))
    audit.update(
        metrics={name: score[name] for name in metrics},
        archived_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
    )
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        with (destination / name).open("xb") as handle:
            handle.write(raw)
    with (destination / "audit.json").open("x") as handle:
        handle.write(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--plan")
    parser.add_argument("--review")
    parser.add_argument("--archive")
    args = parser.parse_args()
    if args.remote:
        print(json.dumps(remote_score(json.load(sys.stdin))))
        return
    assert args.plan and args.review and args.archive
    destination = Path(args.archive).resolve()
    assert not destination.exists()
    plan, request = prepare_request(Path(args.plan), Path(args.review))
    command = [
        "ssh", "-T", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
        plan["host"], shlex.quote(plan["scoring_interpreter"]), "-B",
        shlex.quote(str(Path(plan["root"]) / ENTRY)), "--remote",
    ]
    result = subprocess.run(command, input=json.dumps(request), text=True,
                            capture_output=True, check=True)
    print(json.dumps(archive_result(json.loads(result.stdout), request, destination), indent=2))


if __name__ == "__main__":
    main()
