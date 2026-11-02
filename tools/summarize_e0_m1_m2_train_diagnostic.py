"""Verify and summarize the predeclared frozen TRAIN diagnostic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4194304), b""):
            h.update(block)
    return h.hexdigest()


def state(iou):
    return 0 if iou >= .5 else 2 if iou < .1 else 1


def aggregate(rows):
    valid = [r for r in rows if int(r["valid"])]
    matrix = [[0] * 3 for _ in range(3)]
    for row in valid:
        matrix[state(float(row["e0_iou"]))][int(row["m1_state"])] += 1
    replacements = [r for r in valid if int(r["m2_class_index"]) > 0]
    deltas = [float(r["output_iou"]) - float(r["e0_iou"]) for r in replacements]
    return dict(
        tracked_frames=len(rows), valid_frames=len(valid),
        confusion_gt_rows_predicted_columns=matrix,
        m2_invocations=sum(int(r["m2_invoked"]) for r in rows),
        no_match=sum(int(r["m2_invoked"]) and int(r["m2_class_index"]) == 0 for r in rows),
        replacements=sum(int(r["m2_class_index"]) > 0 for r in rows),
        valid_replacements=len(replacements),
        template_proposed=sum(int(r["template_write_proposed"]) for r in rows),
        template_blocked=sum(int(r["template_write_blocked"]) for r in rows),
        replacement_better=sum(d > 0 for d in deltas),
        replacement_equal=sum(d == 0 for d in deltas),
        replacement_worse=sum(d < 0 for d in deltas),
        replacement_mean_iou_delta=float(np.mean(deltas)) if deltas else None,
        reliable_to_below_half=sum(float(r["e0_iou"]) >= .5 and float(r["output_iou"]) < .5
                                   for r in replacements),
        lost_to_at_least_half=sum(float(r["e0_iou"]) < .1 and float(r["output_iou"]) >= .5
                                  for r in replacements))


def persistence(rows):
    frames = {int(r["frame"]): r for r in rows}
    recovered = [r for r in rows if int(r["valid"]) and int(r["m2_class_index"]) > 0
                 and float(r["e0_iou"]) < .1 and float(r["output_iou"]) >= .5]
    result = {}
    for window in (1, 5, 10):
        eligible = success = 0
        for row in recovered:
            following = [frames.get(int(row["frame"]) + i) for i in range(1, window + 1)]
            if all(r is not None and int(r["valid"]) for r in following):
                eligible += 1
                success += int(all(float(r["output_iou"]) >= .5 for r in following))
        result[str(window)] = dict(eligible=eligible, sustained=success)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan_path = root / args.plan
    plan = json.loads(plan_path.read_bytes())
    assert sha(plan_path) == "b4bcae6ce38bbe2e645aea21a886a1d16080832fb25cf3fb5854e8819856c393"
    run = root / plan["run"]
    receipt = json.loads((run / "COMPLETE.json").read_bytes())
    native_path = root / "runs/pipeline_tasks" / plan["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == receipt["task"] == plan["task"]
    assert native["physical_gpu"] == "1" and native["hostname"] == plan["target_hostname"]
    assert receipt["plan_sha256"] == sha(plan_path)
    assert receipt["state_unchanged"] and receipt["optimizer_updates"] == receipt["backward_calls"] == 0
    assert receipt["frames"] == plan["frames"] == 3076 and receipt["sequences"] == 8
    assert receipt["split"] == "train" and receipt["test_eligible"] is False
    assert receipt["artifacts_sha256"] == {
        p.relative_to(run).as_posix(): sha(p) for p in run.rglob("*")
        if p.is_file() and p.name != "COMPLETE.json"}
    config = json.loads((run / "config.json").read_bytes())
    assert config["plan"] == plan and config["native_command"] == native["command"]
    review_path = root / "reports/E0_M1_M2_FIXED_TRAIN_DIAGNOSTIC_SOURCE_REVIEW_R1_20261101.json"
    assert sha(review_path) == config["source_review_sha256"] == (
        "8f45f27f121e4091f87311d6d9f027c3b4df20c74421ef7aac4e521573da80de")
    review = json.loads(review_path.read_bytes())
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["stage"] == plan["stage"]
    draft_path = root / plan["reviewed_draft"]
    draft = json.loads(draft_path.read_bytes())
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(draft_path)
    assert set(draft) == set(plan) and draft["launch_allowed"] is False
    assert all(plan[k] == v for k, v in draft.items() if k != "launch_allowed")
    for kind in ("source_sha256", "input_sha256"):
        assert review[kind] == plan[kind]
        assert all(sha(root / name) == value for name, value in plan[kind].items())
    seconds = np.load(run / "frame_seconds.npy", allow_pickle=False)
    assert seconds.shape == (3076,) and np.isfinite(seconds).all() and (seconds > 0).all()
    assert np.isclose(seconds.sum(), receipt["seconds"], rtol=0, atol=1e-9)
    manifest = json.loads((root / plan["manifest"]).read_bytes())
    selected = {r["name"]: r for r in manifest["train"] if r["name"] in plan["sequence_names"]}
    collection = json.loads((root / plan["collection_plan"]).read_bytes())
    assert plan["sequence_names"] == collection["fit_sequences"][:4] + collection["holdout_sequences"][:4]
    assert {p.stem for p in (run / "traces").glob("*.csv")} == set(selected)
    partitions = {"fit": [], "holdout": []}
    sampled = {"fit": [], "holdout": []}
    sequences = {}
    progress = json.loads((run / "progress.json").read_bytes())
    assert [r["sequence"] for r in progress] == plan["sequence_names"]
    for name in plan["sequence_names"]:
        row = selected[name]
        count = len(row["rgb"])
        part = "fit" if name in collection["fit_sequences"] else "holdout"
        with (run / "traces" / (name + ".csv")).open() as handle:
            rows = list(csv.DictReader(handle))
        assert [int(r["frame"]) for r in rows] == list(range(1, count))
        valid = set(row["valid_frames"])
        cadence = set([f for f in row["valid_frames"] if 0 < f < count][::25])
        for r in rows:
            assert r["partition"] == part and int(r["valid"]) == int(int(r["frame"]) in valid)
            assert int(r["m2_invoked"]) == int(int(r["m1_state"]) == 2 and int(r["frame"]) in cadence)
            if int(r["valid"]):
                assert all(np.isfinite(float(r[k])) and 0 <= float(r[k]) <= 1
                           for k in ("e0_iou", "output_iou"))
            if int(r["m2_class_index"]) <= 0:
                assert r["e0_iou"] == r["output_iou"]
        partitions[part].extend(rows)
        sampled[part].extend(r for r in rows if int(r["frame"]) in cadence)
        sequences[name] = dict(partition=part, **aggregate(rows), recovery_persistence=persistence(rows))
    all_rows = partitions["fit"] + partitions["holdout"]
    total = aggregate(all_rows)
    assert total["m2_invocations"] == receipt["m2_invocations"]
    assert total["replacements"] == receipt["m2_candidate_selections"]
    summary = dict(
        task=plan["task"], native_sha256=sha(native_path), receipt_sha256=sha(run / "COMPLETE.json"),
        verification="PASS", total=total, sequences=sequences,
        partitions={k: aggregate(v) for k, v in partitions.items()},
        candidate_cadence={k: aggregate(v) for k, v in sampled.items()},
        synchronized_seconds=float(seconds.sum()), concurrent_fps=float(len(seconds) / seconds.sum()),
        parameters=config["total_parameters"], test_eligible=False,
        scope="Small frozen TRAIN diagnostic; no independent E0 baseline, optimizer or TEST admission. "
              "GT rows/columns reliable,uncertain,lost use >=.5/<.1. "
              "E0 proposals share intervened history; holdout was seen by upstream E0.")
    with args.output.open("x") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
