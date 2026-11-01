"""Closed-loop diagnostic on fixed TRAIN sequences; GT is read after decisions."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from benchmark_inputs import sha
from dual_rgbt.data import read_pair
from dual_rgbt.e0_m1_guard import load_reliability
from dual_rgbt.e0_m1_m2_online import E0M1M2Tracker, recovery_due
from lib.utils.box_ops import clip_box

STAGE = "E0_M1_M2_fixed_train_diagnostic"
PARENT = "ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c"


def overlap(box, target):
    box, target = np.asarray(box, np.float64), np.asarray(target, np.float64)
    intersection = np.maximum(0, np.minimum(box[:2] + box[2:], target[:2] + target[2:]) -
                              np.maximum(box[:2], target[:2])).prod()
    return float(intersection / (box[2:].prod() + target[2:].prod() - intersection))


def original_e0_box(previous, normalized, height, width):
    """Reconstruct exactly the base tracker crop geometry and clipping."""
    centre = previous[:2] + previous[2:] / 2
    side = math.sqrt(float(previous[2] * previous[3])) * 4.
    left, top = centre - side / 2
    left, top, side = np.asarray([left, top, side], np.float32)
    box = np.asarray(normalized, np.float32)
    xywh = np.concatenate([box[:2] * side + [left, top] - box[2:] * side / 2,
                           box[2:] * side])
    return np.asarray(clip_box(xywh.tolist(), height, width, margin=10), np.float32)


def admission(path, review_path, review_sha):
    plan = json.loads(Path(path).read_bytes())
    review = json.loads(Path(review_path).read_bytes())
    assert sha(review_path) == review_sha
    assert plan["stage"] == review["stage"] == STAGE
    assert plan["split"] == "train" and plan["phase"] == "diagnostic"
    assert plan["launch_allowed"] is True
    assert plan["test_eligible"] is plan["exploratory_test"] is plan["training_allowed"] is False
    assert plan["target_root"] == review["target_root"] == ROOT.as_posix()
    assert plan["target_host"] == "2026" and plan["assigned_gpu"] == "1"
    assert plan["target_hostname"] == socket.gethostname()
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["source_sha256"] == plan["source_sha256"]
    assert review["input_sha256"] == plan["input_sha256"]
    draft = json.loads((ROOT / plan["reviewed_draft"]).read_bytes())
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(ROOT / plan["reviewed_draft"])
    assert draft["launch_allowed"] is False and set(draft) == set(plan)
    assert all(plan[k] == v for k, v in draft.items() if k != "launch_allowed")
    assert all(sha(ROOT / k) == v for k, v in plan["source_sha256"].items())
    assert all(sha(ROOT / k) == v for k, v in plan["input_sha256"].items())
    assert plan["source_sha256"]["scripts/diagnose_e0_m1_m2_train.py"] == sha(__file__)
    collection = json.loads((ROOT / plan["collection_plan"]).read_bytes())
    assert plan["collection_plan"] in plan["input_sha256"]
    assert collection["split"] == "train"
    assert plan["sequence_names"] == collection["fit_sequences"][:4] + collection["holdout_sequences"][:4]
    assert len(plan["sequence_names"]) == len(set(plan["sequence_names"])) == 8
    witness = json.loads((ROOT / plan["witness_receipt"]).read_bytes())
    witness_native = json.loads((ROOT / plan["witness_native"]).read_bytes())
    assert all(plan[k] in plan["input_sha256"]
               for k in ("witness_receipt", "witness_native", "witness_plan"))
    assert witness_native["state"] == "COMPLETE" and witness_native["exit_code"] == 0
    assert witness["task"] == witness_native["task"] == plan["witness_task"]
    assert witness["frames"] == 93 and witness["smoke_only"] and witness["state_unchanged"]
    assert witness["optimizer_updates"] == witness["backward_calls"] == 0
    witness_plan = json.loads((ROOT / plan["witness_plan"]).read_bytes())
    assert witness["plan_sha256"] == sha(ROOT / plan["witness_plan"])
    assert witness_plan["task"] == plan["witness_task"]
    assert witness_native["physical_gpu"] == "1"
    assert witness_native["hostname"] == plan["target_hostname"]
    witness_config = json.loads(((ROOT / plan["witness_receipt"]).parent / "config.json").read_bytes())
    assert witness_config["plan"] == witness_plan
    assert witness_config["plan_sha256"] == witness["plan_sha256"]
    assert witness_config["native_command"] == witness_native["command"]
    for key in ("parent_checkpoint", "m1_checkpoint", "m2_checkpoint", "factory_checkpoint", "manifest"):
        assert plan[key] == witness_plan[key]
        assert plan["input_sha256"][plan[key]] == witness_plan["input_sha256"][plan[key]]
    for key in witness_plan["source_sha256"]:
        assert plan["source_sha256"][key] == witness_plan["source_sha256"][key]
    witness_run = (ROOT / plan["witness_receipt"]).parent
    assert witness["artifacts_sha256"] == {
        p.relative_to(witness_run).as_posix(): sha(p)
        for p in witness_run.rglob("*") if p.is_file() and p.name != "SMOKE.json"}
    return plan, collection


@torch.no_grad()
def main():
    from dual_rgbt.model import DualRGBT

    parser = argparse.ArgumentParser()
    for name in ("plan", "source-review", "source-review-sha256", "task", "run"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    plan, collection = admission(args.plan, args.source_review, args.source_review_sha256)
    assert args.task == plan["task"] and args.run == plan["run"]
    native_path = ROOT / "runs/pipeline_tasks" / args.task / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["state"] == "RUNNING" and native["pid"] == os.getppid()
    assert native["task"] == args.task and native["hostname"] == socket.gethostname()
    assert native["physical_gpu"] == os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert native["command"] == [sys.executable] + sys.argv == [
        str(ROOT / "conda-envs/dual_rgbt/bin/python"), str(ROOT / "scripts/diagnose_e0_m1_m2_train.py"),
        "--plan", str(ROOT / args.plan), "--source-review", str(ROOT / args.source_review),
        "--source-review-sha256", args.source_review_sha256, "--task", args.task, "--run", args.run]
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["OPENCV_FOR_THREADS_NUM"] == "1"
    manifest = json.loads((ROOT / plan["manifest"]).read_bytes())
    rows = {r["name"]: r for r in manifest["train"]}
    assert not set(plan["sequence_names"]) & (
        set(manifest["test_sequences"]) | {r["name"] for r in manifest["dev"]})
    selected = [rows[name] for name in plan["sequence_names"]]
    assert sum(len(r["rgb"]) for r in selected) == plan["frames"]
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = DualRGBT(str(ROOT / plan["factory_checkpoint"]))
    model.load_state_dict(torch.load(ROOT / plan["parent_checkpoint"], map_location="cpu")["net"],
                          strict=True)
    model.cuda().eval().requires_grad_(False)
    head, mean, scale = load_reliability(ROOT / plan["m1_checkpoint"], PARENT, "cuda")
    tracker = E0M1M2Tracker(model, 0., ROOT / plan["m2_checkpoint"], head, mean, scale)
    modules = dict(e0=model, m1=head, m2=tracker.head)
    before = {n: {k: v.cpu().clone() for k, v in m.state_dict().items()} for n, m in modules.items()}
    run = ROOT / args.run
    run.mkdir(parents=True, exist_ok=False)
    (run / "traces").mkdir()
    config = dict(plan=plan, plan_sha256=sha(args.plan), native_command=native["command"],
                  source_review_sha256=args.source_review_sha256,
                  total_parameters=sum(p.numel() for m in modules.values() for p in m.parameters()),
                  timing_scope="Decode/init/E0/M1/actual M2 synchronized; excludes GT diagnostics/model load.")
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    fields = ["frame", "valid", "partition", "m1_state", "m1_reliable", "m1_uncertain", "m1_lost",
              "template_write_proposed", "template_write_blocked", "m2_invoked", "m2_class_index",
              "m2_no_match_score", "m2_selected_score", "e0_iou", "output_iou", "seconds"]
    progress, timings, calls, replacements = [], [], 0, 0
    for row in selected:
        count = len(row["rgb"])
        valid = set(row["valid_frames"])
        part = "fit" if row["name"] in collection["fit_sequences"] else "holdout"
        tracker.set_sequence(row["valid_frames"], count)
        started = time.perf_counter()
        tracker.initialize(read_pair(row, 0), row["boxes"][0])
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        assert tracker.guard.record is None and tracker.guard.observer.record is None
        with (run / "traces" / (row["name"] + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            for frame in range(1, count):
                previous = tracker.base.state.copy()
                started = time.perf_counter()
                image = read_pair(row, frame)
                box, route = tracker.track(image)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                assert tracker.guard.observer.calls == 1 and tracker.base.frame == frame
                assert np.isfinite(box).all() and (box[2:] > 0).all()
                record, decision = tracker.guard.record, tracker.last_m2
                assert decision["invoked"] == recovery_due(frame, tracker.valid_frames, record["m1_state"])
                e0 = original_e0_box(previous, tracker.guard.observer.record["final_normalized_box"],
                                     image.shape[0], image.shape[1])
                if decision["class_index"] <= 0:
                    assert np.array_equal(e0, box)
                # Diagnostics read this frame's annotation only after the causal decision.
                target = row["boxes"][frame] if frame in valid else None
                writer.writerow(dict(frame=frame, valid=int(frame in valid), partition=part,
                                     **record, m2_invoked=int(decision["invoked"]),
                                     m2_class_index=decision["class_index"],
                                     m2_no_match_score=decision["no_match_score"],
                                     m2_selected_score=decision["selected_score"],
                                     e0_iou=overlap(e0, target) if target is not None else "",
                                     output_iou=overlap(box, target) if target is not None else "",
                                     seconds=elapsed))
                calls += int(decision["invoked"])
                replacements += int(decision["class_index"] > 0)
                timings.append(elapsed)
        progress.append(dict(sequence=row["name"], frames=count, partition=part))
        (run / "progress.json").write_text(json.dumps(progress) + "\n")
        print(json.dumps(progress[-1]), flush=True)
    assert all(torch.equal(v.cpu(), before[n][k]) for n, m in modules.items()
               for k, v in m.state_dict().items())
    assert all(p.grad is None for m in modules.values() for p in m.parameters())
    tracker.close()
    assert model.predict == tracker.guard.observer.original
    admission(args.plan, args.source_review, args.source_review_sha256)
    timing = np.asarray(timings, np.float64)
    assert timing.shape == (plan["frames"],) and np.isfinite(timing).all() and (timing > 0).all()
    np.save(run / "frame_seconds.npy", timing)
    receipt = dict(task=args.task, split="train", sequences=8, frames=plan["frames"],
                   plan_sha256=sha(args.plan), state_unchanged=True, test_eligible=False,
                   optimizer_updates=0, backward_calls=0, m2_invocations=calls,
                   m2_candidate_selections=replacements, seconds=float(timing.sum()),
                   scope="Fixed TRAIN closed loop; original E0 proposal is on intervened history, "
                         "not an independent matched E0 control. No optimizer or TEST admission.",
                   artifacts_sha256={p.relative_to(run).as_posix(): sha(p)
                                     for p in run.rglob("*") if p.is_file()})
    (run / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
