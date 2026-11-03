"""Score an exact completed fixed P1A control once and archive its compute."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import base64,hashlib,json,subprocess,sys
from pathlib import Path
request=json.loads(sys.stdin.read())
root=Path(request["root"])
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
review_path=root/"reports"/request["review_name"]
assert sha(review_path)==request["review_sha256"]
review=json.loads(review_path.read_text())
assert review["stage"]=="E0_finetuned_evaluation"
assert review["status"]=="SOURCE_PASS" and review["blocking_issues"]==0
for name,digest in review["source_sha256"].items():
    assert sha(root/name)==digest,name
task=request["task"]
native_path=root/"runs/pipeline_tasks"/task/"status.json"
native=json.loads(native_path.read_text())
assert native["task"]==task
if native["state"]!="COMPLETE" or native.get("exit_code")!=0:
    print(json.dumps(dict(status="NOT_READY",native=native)))
    sys.exit(0)
command=native["command"]
run=Path(command[command.index("--run")+1])
config=json.loads((run/"config.json").read_text())
complete=json.loads((run/"COMPLETE.json").read_text())
expected=request["expected"]
assert complete["sequences"]==expected[0] and complete["frames"]==expected[1]
assert native["command"]==config["native_command"] and native["physical_gpu"]==config["physical_gpu"]
assert complete["state_unchanged"] and complete["optimizer_updates"]==complete["backward_calls"]==0
scoring=review["scoring"]
score_path=Path(scoring["outputs"][task])
log_files={}
if not score_path.exists():
    assert not score_path.with_suffix(".npz").exists()
    score_command=[scoring["interpreter"],"-B",str(root/"tools/score_e0_finetuned.py"),
                   "--run",str(run),"--toolkit",scoring["toolkit"],
                   "--protocol",scoring["protocol"]["path"],"--output",str(score_path)]
    result=subprocess.run(score_command,cwd=str(root),stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    log=score_path.with_suffix(".scoring.log")
    assert not log.exists()
    log.write_bytes(result.stdout)
    assert result.returncode==0,result.stdout.decode()[-4000:]
    log_files["scoring.log"]=log
score=json.loads(score_path.read_text())
assert score["status"]=="SCORED_COMPLETE" and score["sequences"]==expected[0]
assert score["frames"]==expected[1] and score["exploratory_test"] is True
assert score["test_eligible"] is False and score["checkpoint_sha256"]==config["checkpoint_sha256"]
assert sha(score_path.with_suffix(".npz"))==score["curves_sha256"]
profile=root/"reports/P1B_E0_CURRENT_ROUTE_MAC_CPU_CROSSCHECK_P25_20261102.json"
profile_sha="9d63719a1d16d83310b7100939a64183dcf5c010c96b7d4522c1602651e8542a"
assert sha(profile)==profile_sha
summary_tool=root/"tools/summarize_e0_full_compute.py"
assert sha(summary_tool)=="efd27f1d4ab4de69e77c91e2033eb644a7b52904c319428c108dce2cc54aa77e"
compute_path=score_path.with_name(score_path.stem+"_compute.json")
if not compute_path.exists():
    compute_command=[scoring["interpreter"],"-B",str(summary_tool),
                     "--root",str(root),"--run",str(run),
                     "--score",str(score_path),"--score-sha256",sha(score_path),
                     "--profile",str(profile),"--profile-sha256",profile_sha,
                     "--output",str(compute_path)]
    result=subprocess.run(compute_command,cwd=str(root),stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    log=score_path.with_suffix(".compute.log")
    assert not log.exists()
    log.write_bytes(result.stdout)
    assert result.returncode==0,result.stdout.decode()[-4000:]
    log_files["compute.log"]=log
compute=json.loads(compute_path.read_text())
assert compute["status"]=="COMPLETE_FULL_RUN_COMPUTE_SUMMARY"
assert compute["score_sha256"]==sha(score_path) and compute["frames"]==expected[1]
assert compute["profile_sha256"]==profile_sha and compute["task"]==task
files={}
for name,path in dict(
        score=score_path,curves=score_path.with_suffix(".npz"),compute=compute_path,
        native=native_path,config=run/"config.json",complete=run/"COMPLETE.json",
        progress=run/"progress.json",source_review=review_path,**log_files).items():
    raw=path.read_bytes()
    suffix=".npz" if name=="curves" else ".log" if name.endswith(".log") else ".json"
    filename=name if name.endswith(".log") else name+suffix
    files[filename]=dict(raw=base64.b64encode(raw).decode(),
                         sha256=hashlib.sha256(raw).hexdigest())
print(json.dumps(dict(status="ARCHIVED_SCORED_COMPLETE",task=task,
    metrics={key:score[key] for key in ("PR","NPR","SR","MPR","MSR") if key in score},
    fps=score["fps"],total_parameters=score["total_parameters"],
    mean_tracked_frame_gmacs=compute["mean_tracked_frame_gmacs"],files=files)))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("benchmark", choices=("lasher", "rgbt210", "rgbt234"))
    args = parser.parse_args()
    folder = ROOT / "reports/p1a_completed_scores_r1_20261103" / args.branch / args.benchmark
    assert not folder.exists()
    title = "DROPMOE" if args.branch == "dropmoe" else "SOT"
    review_name = "P1A_EPOCH20_%s_EVALUATION_SOURCE_REVIEW_R1_20261102.json" % title
    raw = (ROOT / "reports" / review_name).read_bytes()
    review = json.loads(raw)
    host = "2025" if args.branch == "dropmoe" else "2026"
    task = "eval_p1a_e020_%s_%s_p%s_r1_20261102" % (
        args.benchmark, args.branch, host[-2:])
    expected = dict(lasher=(245, 220703), rgbt210=(210, 104706), rgbt234=(234, 116650))
    parent = "/home/gaob/Dual-RGBT" if host == "2025" else "/data2/gb/Dual-RGBT"
    request = dict(root=review["target_root"], task=task, review_name=review_name,
                   review_sha256=hashlib.sha256(raw).hexdigest(),
                   expected=expected[args.benchmark])
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         host, parent + "/conda-envs/cpu-scoring/bin/python", "-c", shlex.quote(REMOTE)],
        input=json.dumps(request), encoding="utf-8", stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True)
    report = json.loads(result.stdout)
    if report["status"] == "NOT_READY":
        print(json.dumps(report))
        return
    folder.mkdir(parents=True, exist_ok=False)
    files = report.pop("files")
    for name, item in files.items():
        raw = base64.b64decode(item["raw"])
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        (folder / name).write_bytes(raw)
    report["archived_sha256"] = {name: item["sha256"] for name, item in files.items()}
    (folder / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
