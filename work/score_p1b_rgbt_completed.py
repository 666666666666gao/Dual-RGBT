"""Score one exact completed fixed-epoch P1B RGBT run and archive it once."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]

REMOTE = r'''
import base64
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

request = json.loads(sys.stdin.read())
d = request["draft"]
root = Path(d["root"])
job = d["jobs"][0]
run = Path(job["run"])
expected = {"rgbt210": (210, 104706), "rgbt234": (234, 116650)}[d["benchmark"]]
assert (job["sequences"], job["frames"]) == expected
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_bytes())

def validate_score_provenance(score):
    assert score["source_sha256"] == d["source_sha256"]
    assert score["source_review_sha256"] == request["review_sha256"]
    assert score["manifest_sha256"] == job["manifest"]["sha256"]
    assert score["toolkit_protocol_sha256"] == d["protocol"]["sha256"]
    assert score["toolkit_code_sha256"] == read(d["protocol"]["path"])["toolkit_code_sha256"]
    assert score["scoring_script_sha256"] == d["source_sha256"]["tools/score_e0_finetuned.py"]
    assert score["metric_implementation_sha256"] == {
        name: d["source_sha256"][name] for name in (
            "tools/joint_candidate_results.py", "scripts/score_tracking.py",
            "scripts/joint_evaluation_contract.py", "scripts/benchmark_inputs.py")}
    assert score["native_status_sha256"] == sha(native_path)
    assert score["artifacts_sha256"] == {
        path.relative_to(run).as_posix(): sha(path)
        for path in run.rglob("*") if path.is_file()}

assert sha(root / "reports/draft.json") == request["draft_sha256"]
assert sha(d["review"]["path"]) == request["review_sha256"]
review = read(d["review"]["path"])
assert review["stage"] == "E0_finetuned_evaluation"
assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
assert review["reviewed_draft"]["sha256"] == request["draft_sha256"]
assert review["source_sha256"] == d["source_sha256"]
assert review["checkpoint"] == d["checkpoint"] and review["factory"] == d["factory"]
assert review["target_root"] == str(root) and review["target_host"] == d["host"]
for name, digest in d["source_sha256"].items():
    assert sha(root / name) == digest, name
native_path = root / "runs/pipeline_tasks" / job["task"] / "status.json"
native = read(native_path)
if native["state"] != "COMPLETE" or native.get("exit_code") != 0:
    print(json.dumps(dict(status="NOT_READY", native=native)))
    sys.exit(0)
config = read(run / "config.json")
complete = read(run / "COMPLETE.json")
command = [request["review_sha256"] if word == "PENDING_REVIEW_SHA256" else word
           for word in job["command"]]
assert native["command"] == config["native_command"] == command
assert native["task"] == config["task"] == complete["task"] == job["task"]
assert native["physical_gpu"] == config["physical_gpu"] == job["gpu"]
assert complete["sequences"] == config["split_sequences"] == expected[0]
assert complete["frames"] == config["split_frames"] == expected[1]
assert complete["state_unchanged"] is True and complete["smoke_only"] is False
assert complete["optimizer_updates"] == complete["backward_calls"] == 0
assert config["checkpoint_sha256"] == complete["checkpoint_sha256"] == d["checkpoint"]["sha256"]
assert config["source_sha256"] == d["source_sha256"]
assert config["source_review_sha256"] == request["review_sha256"]
assert read(run / "precision.json") == dict(
    matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)
manifest = read(job["manifest"]["path"])
assert sha(job["manifest"]["path"]) == job["manifest"]["sha256"]
assert {path.stem for path in (run / "traces").glob("*.csv")} == {
    row["name"] for row in manifest["sequences"]}
for row in manifest["sequences"]:
    with (run / "traces" / (row["name"] + ".csv")).open() as handle:
        frames = [int(record["frame"]) for record in csv.DictReader(handle)]
    assert frames == list(range(1, len(row["rgb"])))
assert sha(d["protocol"]["path"]) == d["protocol"]["sha256"]
score_path = Path(job["score_output"])
log = score_path.with_suffix(".scoring.log")
if not score_path.exists():
    assert not score_path.with_suffix(".npz").exists() and not log.exists()
    result = subprocess.run([
        d["scoring_interpreter"], "-B", str(root / "tools/score_e0_finetuned.py"),
        "--run", str(run), "--toolkit", d["toolkit"],
        "--protocol", d["protocol"]["path"], "--output", str(score_path)],
        cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_bytes(result.stdout)
    assert result.returncode == 0, result.stdout.decode()[-4000:]
score = read(score_path)
assert score["status"] == "SCORED_COMPLETE"
assert (score["sequences"], score["frames"]) == expected
assert score["exploratory_test"] is True and score["test_eligible"] is False
assert score["checkpoint_sha256"] == d["checkpoint"]["sha256"]
assert score["curves_sha256"] == sha(score_path.with_suffix(".npz"))
validate_score_provenance(score)
files = {}
for name, path in dict(
        score=score_path, curves=score_path.with_suffix(".npz"),
        native=native_path, config=run / "config.json",
        complete=run / "COMPLETE.json", precision=run / "precision.json",
        review=Path(d["review"]["path"]), scoring_log=log).items():
    raw = path.read_bytes()
    suffix = ".npz" if name == "curves" else ".log" if name == "scoring_log" else ".json"
    files[name + suffix] = dict(
        raw=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
metric_names = ("PR", "SR") if d["benchmark"] == "rgbt210" else ("MPR", "MSR")
print(json.dumps(dict(
    status="ARCHIVED_P1B_RGBT_SCORED_COMPLETE", task=job["task"],
    benchmark=d["benchmark"], metrics={key: score[key] for key in metric_names},
    fps=score["fps"], mean_ms=score["mean_ms"], p95_ms=score["p95_ms"],
    total_parameters=score["total_parameters"], files=files)))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("benchmark", choices=("rgbt210", "rgbt234"))
    args = parser.parse_args()
    folder = ROOT / "reports/p1b_rgbt_completed_scores_r1_20261104" / args.branch / args.benchmark
    assert not folder.exists()
    plans = ("reports/p1b_rgbt_evaluation_r2_20261104"
             if (args.branch, args.benchmark) == ("sot", "rgbt234")
             else "reports/p1b_rgbt_evaluation_r1_20261104")
    draft_raw = (ROOT / plans /
                 args.branch / args.benchmark / "draft.json").read_bytes()
    draft = json.loads(draft_raw)
    review_raw = (ROOT / "reports" / PurePosixPath(draft["review"]["path"]).name).read_bytes()
    request = dict(
        draft=draft, draft_sha256=hashlib.sha256(draft_raw).hexdigest(),
        review_sha256=hashlib.sha256(review_raw).hexdigest())
    result = subprocess.run([
        "ssh", "-T", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
        draft["host"], draft["scoring_interpreter"], "-B", "-c", shlex.quote(REMOTE)],
        input=json.dumps(request), text=True, capture_output=True, check=True)
    audit = json.loads(result.stdout)
    if audit["status"] == "NOT_READY":
        print(json.dumps(audit))
        return
    assert audit["status"] == "ARCHIVED_P1B_RGBT_SCORED_COMPLETE"
    files = audit.pop("files")
    folder.mkdir(parents=True, exist_ok=False)
    for name, item in files.items():
        raw = base64.b64decode(item["raw"])
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        (folder / name).write_bytes(raw)
    audit["archived_sha256"] = {name: item["sha256"] for name, item in files.items()}
    (folder / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
