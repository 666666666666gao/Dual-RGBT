"""Bind SOT234 evaluation to successful SOT210 completion before any launch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

from operate_p1b_rgbt_evaluation import OPERATIONAL as BASE_OPERATIONAL
from operate_p1b_evaluation import REMOTE as BASE_REMOTE

ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL = BASE_OPERATIONAL + ("work/operate_p1b_sot_rgbt234_followup.py",)
FOLDER = "reports/p1b_rgbt_evaluation_r2_20261104/sot/rgbt234"
GATE = r'''
import io
import hashlib
import json
from pathlib import Path
import sys

raw_request = sys.stdin.read()
followup = json.loads(raw_request)
if followup["action"] in ("deploy", "smoke", "full"):
    preceding = followup["draft"]["predecessor"]
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    load = lambda path: json.loads(Path(path).read_bytes())
    assert digest(preceding["draft_path"]) == preceding["draft_sha256"]
    assert digest(preceding["review_path"]) == preceding["review_sha256"]
    predecessor_draft = load(preceding["draft_path"])
    assert predecessor_draft["host"] == followup["draft"]["host"] == "2026"
    assert predecessor_draft["branch"] == followup["draft"]["branch"] == "sot"
    assert predecessor_draft["benchmark"] == "rgbt210"
    job = preceding["job"]
    assert predecessor_draft["jobs"] == [job]
    assert job["gpu"] == followup["draft"]["smoke"]["gpu"] == "1"
    native = load(preceding["native_path"])
    config = load(Path(job["run"]) / "config.json")
    complete = load(Path(job["run"]) / "COMPLETE.json")
    command = [preceding["review_sha256"] if word == "PENDING_REVIEW_SHA256" else word
               for word in job["command"]]
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == config["task"] == complete["task"] == job["task"]
    assert native["physical_gpu"] == config["physical_gpu"] == job["gpu"]
    assert native["command"] == config["native_command"] == command
    assert config["source_review_sha256"] == preceding["review_sha256"]
    assert complete["sequences"] == config["split_sequences"] == 210
    assert complete["frames"] == config["split_frames"] == 104706
    assert complete["state_unchanged"] is True and complete["smoke_only"] is False
    assert complete["optimizer_updates"] == complete["backward_calls"] == 0
    assert not Path("/proc/%s" % native["pid"]).exists()
sys.stdin = io.StringIO(raw_request)
'''
REMOTE = GATE + "\n" + BASE_REMOTE


def prepare(old, predecessor, predecessor_sha256, review_sha256):
    result = copy.deepcopy(old)
    assert result["branch"] == "sot" and result["benchmark"] == "rgbt234"
    assert predecessor["branch"] == "sot" and predecessor["benchmark"] == "rgbt210"
    root = old["root"].replace("_r1_20261104", "_r2_20261104")
    review = root + "/reports/P1B_SOT_RGBT234_EVALUATION_REVIEW_R2_20261104.json"
    result["root"] = root
    result["review"] = dict(path=review, sha256="PENDING")
    for job in [result["smoke"]] + result["jobs"]:
        old_task, old_run = job["task"], job["run"]
        job["task"] = old_task.replace("_r1_20261104", "_r2_20261104")
        job["run"] = old_run.replace(old_task, job["task"])
        job["command"] = [
            word.replace(old["root"], root).replace(old_task, job["task"])
            for word in job["command"]]
        command = job["command"]
        command[command.index("--source-review") + 1] = review
        job["score_output"] = root + "/reports/" + job["task"] + "_SCORES.json"
    previous_job = predecessor["jobs"][0]
    result["predecessor"] = dict(
        draft_path=predecessor["root"] + "/reports/draft.json",
        draft_sha256=predecessor_sha256,
        review_path=predecessor["review"]["path"], review_sha256=review_sha256,
        job=copy.deepcopy(previous_job),
        native_path=predecessor["root"] + "/runs/pipeline_tasks/" +
        previous_job["task"] + "/status.json")
    result["scheduling"] = (
        "Deploy/smoke/full require pinned SOT210 native COMPLETE0, exact command/"
        "config/full receipt and exited wrapper, then unchanged fresh empty GPU gate.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "deploy", "smoke", "full", "inspect"))
    args = parser.parse_args()
    folder = ROOT / FOLDER
    if args.action == "prepare":
        assert not folder.exists()
        old = json.loads((ROOT / "reports/p1b_rgbt_evaluation_r1_20261104/"
                          "sot/rgbt234/draft.json").read_bytes())
        preceding_raw = (ROOT / "reports/p1b_rgbt_evaluation_r1_20261104/"
                         "sot/rgbt210/draft.json").read_bytes()
        preceding = json.loads(preceding_raw)
        review = ROOT / "reports" / PurePosixPath(preceding["review"]["path"]).name
        draft = prepare(old, preceding, hashlib.sha256(preceding_raw).hexdigest(),
                        hashlib.sha256(review.read_bytes()).hexdigest())
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "draft.json").write_text(json.dumps(draft, indent=2) + "\n")
        print(json.dumps(dict(status="DISABLED_DRAFT", root=draft["root"])))
        return
    draft_raw = (folder / "draft.json").read_bytes()
    draft = json.loads(draft_raw)
    review_raw = (ROOT / "reports" / PurePosixPath(draft["review"]["path"]).name).read_bytes()
    receipt = folder / (args.action + "_receipt.json")
    if args.action != "inspect":
        assert not receipt.exists()
    request = dict(
        draft=draft, draft_hex=draft_raw.hex(),
        draft_sha256=hashlib.sha256(draft_raw).hexdigest(),
        review_hex=review_raw.hex(), review_sha256=hashlib.sha256(review_raw).hexdigest(),
        operational_sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in OPERATIONAL},
        entry_hex=(ROOT / "scripts/eval_p1b_completed.py").read_bytes().hex(),
        action=args.action)
    result = subprocess.run([
        "ssh", "-T", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
        draft["host"], draft["smoke"]["command"][0], "-B", "-c", shlex.quote(REMOTE)],
        input=json.dumps(request), text=True, capture_output=True, check=True)
    output = json.loads(result.stdout)
    if args.action != "inspect":
        receipt.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
