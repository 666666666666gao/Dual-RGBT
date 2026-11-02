"""Execute the unchanged R2 scorer with the independently reviewed dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory_files(paths):
    files = set()
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.update(item for item in path.rglob("*") if item.is_file())
        else:
            files.add(path)
    return {str(path): sha(path) for path in sorted(files)}


parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", required=True, choices=("LasHeR", "RGBT210", "RGBT234"))
parser.add_argument("--review-sha256", required=True)
parser.add_argument("--driver-sha256", required=True)
args = parser.parse_args()
root = Path("/home/gaob/Dual-RGBT/snapshots/cadtrack_ep20_original_r2_20261102")
folder = root / "reports/cadtrack_ep20_original_r3_scoring_dependency"
review_path = root / "reports/CADTRACK_EP20_ORIGINAL_SCORING_DEPENDENCY_REVIEW_R3_20261102.json"
draft_path = folder / "draft.json"
assert sha(review_path) == args.review_sha256
review = json.loads(review_path.read_bytes())
draft = json.loads(draft_path.read_bytes())
assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
assert review["stage"] == draft["stage"]
assert review["reviewed_drafts"]["reports/cadtrack_ep20_original_r3_scoring_dependency/draft.json"] == sha(draft_path)
assert draft["target_root"] == str(root)
base_draft = root / draft["base_draft"]
assert sha(base_draft) == draft["base_draft_sha256"]
base = json.loads(base_draft.read_bytes())
assert sha(root / draft["base_source_review"]) == draft["base_source_review_sha256"]
assert sha(root / draft["base_scoring_script"]) == draft["base_scoring_script_sha256"]
dependency = draft["additional_dependency"]
assert sha(root / dependency["path"]) == dependency["sha256"]
protocol_path = Path(base["protocol"])
protocol_sha = "899c14feddceaead8973067e5fe34fdc84284804e6ea389c92bf84d77ad932d2"
assert sha(protocol_path) == protocol_sha
protocol = json.loads(protocol_path.read_bytes())
toolkit = Path(base["toolkit"])
toolkit_paths = [toolkit / name.replace("\\", "/")
                 for name in protocol["toolkit_code_sha256"]]
toolkit_hashes = dict(zip(map(str, toolkit_paths),
                          protocol["toolkit_code_sha256"].values()))
assert all(sha(path) == digest for path, digest in toolkit_hashes.items())
job = base["jobs"][args.benchmark]
manifest_path = Path(job["manifest"])
manifest = json.loads(manifest_path.read_bytes())
assert sha(manifest_path) == job["manifest_sha256"]
bound = {str(root / key): value for key, value in base["source_sha256"].items()}
bound.update({
    str(root / dependency["path"]): dependency["sha256"],
    str(review_path): args.review_sha256,
    str(draft_path): sha(draft_path),
    str(base_draft): draft["base_draft_sha256"],
    str(root / draft["base_source_review"]): draft["base_source_review_sha256"],
    str(protocol_path): protocol_sha,
    **toolkit_hashes,
    str(manifest_path): job["manifest_sha256"],
})
assert all(sha(key) == value for key, value in bound.items())
run = Path(job["run"])
config = json.loads((run / "config.json").read_bytes())
receipt = json.loads((run / "COMPLETE.json").read_bytes())
native = json.loads(Path(config["native_status"]).read_bytes())
assert native["state"] == "COMPLETE" and native["exit_code"] == 0
assert config["task"] == receipt["task"] == native["task"] == job["task"]
assert receipt["config_sha256"] == sha(run / "config.json")
assert receipt["sequences"] == job["expected_sequences"]
assert receipt["frames"] == job["expected_frames"]
consumed = [
    run / "config.json", run / "COMPLETE.json", Path(config["native_status"]),
    Path(config["checkpoint"]), manifest_path, run / "predictions",
    run / "groundtruth", run / "frame_seconds.npy",
]
if args.benchmark != "LasHeR":
    consumed.extend(toolkit / name.replace("\\", "/")
                    for name in manifest["annotation_file_sha256"])
input_inventory = inventory_files(consumed)
bound.update(input_inventory)
output = folder / (args.benchmark.lower() + ".json")
provenance = folder / (args.benchmark.lower() + "_scoring_receipt.json")
assert not output.exists() and not output.with_suffix(".npz").exists()
assert not provenance.exists()
command = [
    base["scoring_interpreter"], "-B", str(root / draft["base_scoring_script"]),
    "--run", str(run), "--output", str(output),
    "--toolkit", base["toolkit"], "--protocol", base["protocol"],
]
started = datetime.now(timezone.utc).isoformat()
result = subprocess.run(command, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
post = inventory_files([Path(key) for key in bound])
record = dict(
    stage=draft["stage"], benchmark=args.benchmark, command=command,
    started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
    exit_code=result.returncode, pre_sha256=bound, post_sha256=post,
    unchanged=bound == post, stdout=result.stdout.decode("utf-8"),
    stderr=result.stderr.decode("utf-8"), exploratory_test=True, test_eligible=False,
    driver_sha256=args.driver_sha256, consumed_input_inventory=input_inventory,
)
if output.exists():
    record["score_sha256"] = sha(output)
if output.with_suffix(".npz").exists():
    record["curves_sha256"] = sha(output.with_suffix(".npz"))
provenance.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({key: value for key, value in record.items()
                  if key not in ("pre_sha256", "post_sha256", "stdout")}), flush=True)
assert bound == post
if result.returncode != 0:
    raise SystemExit(result.returncode)
score = json.loads(output.read_bytes())
print(json.dumps({key: value for key, value in score.items()
                  if key not in ("per_sequence", "source_sha256")}), flush=True)
