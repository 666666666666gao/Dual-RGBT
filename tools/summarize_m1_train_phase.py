"""Describe frame-phase errors in the already completed frozen TRAIN diagnostic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.archive / "runtime"
    receipt_path = run / "COMPLETE.json"
    receipt = json.loads(receipt_path.read_bytes())
    summary = json.loads((args.archive / "summary.json").read_bytes())
    native_path = args.archive / "native_status.json"
    native = json.loads(native_path.read_bytes())
    assert summary["verification"] == "PASS"
    assert sha(receipt_path) == summary["receipt_sha256"]
    assert sha(native_path) == summary["native_sha256"]
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert receipt["split"] == "train" and receipt["test_eligible"] is False
    assert receipt["state_unchanged"]
    assert receipt["optimizer_updates"] == receipt["backward_calls"] == 0
    files = sorted((run / "traces").glob("*.csv"))
    assert {p.stem for p in files} == set(summary["sequences"])
    rows = []
    bindings = {}
    for path in files:
        relative = path.relative_to(run).as_posix()
        bindings[relative] = sha(path)
        assert bindings[relative] == receipt["artifacts_sha256"][relative]
        with path.open(newline="") as handle:
            sequence_rows = list(csv.DictReader(handle))
        assert [int(r["frame"]) for r in sequence_rows] == list(
            range(1, len(sequence_rows) + 1))
        assert all(r["partition"] == summary["sequences"][path.stem]["partition"]
                   for r in sequence_rows)
        rows.extend(sequence_rows)
    assert len(rows) == summary["total"]["tracked_frames"]
    result = []
    for partition in ("fit", "holdout"):
        valid = [r for r in rows if r["partition"] == partition and int(r["valid"])]
        assert len(valid) == summary["partitions"][partition]["valid_frames"]
        for phase in range(5):
            selected = [r for r in valid if int(r["frame"]) % 5 == phase]
            lost = [r for r in selected if float(r["e0_iou"]) < .1]
            reliable = [r for r in selected if float(r["e0_iou"]) >= .5]
            result.append(dict(
                partition=partition, frame_mod5=phase, valid=len(selected),
                lost=len(lost),
                lost_recognized=sum(int(r["m1_state"]) == 2 for r in lost),
                reliable=len(reliable),
                reliable_interrupted=sum(int(r["m1_state"]) != 0 for r in reliable)))
    report = dict(
        status="VERIFIED_FIXED_TRAIN_PHASE_DESCRIPTION",
        receipt_sha256=sha(receipt_path), native_sha256=sha(native_path),
        summary_sha256=sha(args.archive / "summary.json"),
        trace_sha256=bindings, tracked_rows=len(rows), strata=result,
        script_sha256=sha(Path(__file__)), test_eligible=False,
        training_or_inference_performed=False,
        scope="Post-hoc description of eight fixed TRAIN sequences, not a causal "
              "phase experiment or population estimate. Upstream E0 saw both "
              "partitions; local proposals share intervened history. Phase1 "
              "includes more frames than the actual 25-valid-frame M2 cadence. "
              "No threshold, checkpoint, seed or policy is selected.")
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
