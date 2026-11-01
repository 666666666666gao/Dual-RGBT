"""Bound TRAIN-only execution witness for M1-gated M2; no TEST support."""
import argparse
import csv
import json
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

STAGE = "E0_M1_M2_train_witness"
PARENT = "ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c"
M1 = "658c84275ad1e1624d4b453fee89dd312273633664024bf0b42ae78dcc189ebf"
M2 = "7e7df5d25d3e8667fe38a4ca8f6ec3662308ffcb52a09bd16aecaab59a9cb4be"


def admission(plan_path, review_path, review_sha):
    plan = json.loads(Path(plan_path).read_bytes())
    review = json.loads(Path(review_path).read_bytes())
    assert sha(review_path) == review_sha
    assert plan["stage"] == review["stage"] == STAGE
    assert plan["split"] == "train" and plan["phase"] == "smoke"
    assert plan["launch_allowed"] is True and plan["test_eligible"] is False
    assert plan["training_allowed"] is plan["exploratory_test"] is False
    assert plan["target_root"] == ROOT.as_posix()
    assert plan["target_host"] == "2026"
    assert plan["target_hostname"] == socket.gethostname()
    assert plan["assigned_gpu"] == "1"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["target_root"] == plan["target_root"]
    assert review["source_sha256"] == plan["source_sha256"]
    assert review["input_sha256"] == plan["input_sha256"]
    draft_path = ROOT / plan["reviewed_draft"]
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(draft_path)
    draft = json.loads(draft_path.read_bytes())
    assert draft["launch_allowed"] is False
    assert set(plan) == set(draft)
    assert all(plan[k] == v for k, v in draft.items() if k != "launch_allowed")
    assert all(sha(ROOT / k) == v for k, v in plan["source_sha256"].items())
    assert all(sha(ROOT / k) == v for k, v in plan["input_sha256"].items())
    assert plan["source_sha256"]["scripts/check_e0_m1_m2_train.py"] == sha(__file__)
    assert plan["input_sha256"][plan["parent_checkpoint"]] == PARENT
    assert plan["input_sha256"][plan["m1_checkpoint"]] == M1
    assert plan["input_sha256"][plan["m2_checkpoint"]] == M2
    assert plan["manifest"] in plan["input_sha256"]
    assert plan["factory_checkpoint"] in plan["input_sha256"]
    assert plan["frames_per_sequence"] == 31
    assert plan["sequence_names"] == ["2up", "motocross", "carturn"]
    return plan


@torch.no_grad()
def main():
    from dual_rgbt.model import DualRGBT

    parser = argparse.ArgumentParser()
    for name in ("plan", "source-review", "source-review-sha256", "task", "run"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    plan = admission(args.plan, args.source_review, args.source_review_sha256)
    assert args.task == plan["task"] and args.run == plan["run"]
    native_path = ROOT / "runs/pipeline_tasks" / args.task / "status.json"
    native = json.loads(native_path.read_bytes())
    expected = [str(ROOT / "conda-envs/dual_rgbt/bin/python"),
                str(ROOT / "scripts/check_e0_m1_m2_train.py"),
                "--plan", str(ROOT / args.plan),
                "--source-review", str(ROOT / args.source_review),
                "--source-review-sha256", args.source_review_sha256,
                "--task", args.task, "--run", args.run]
    assert native["state"] == "RUNNING" and native["pid"] == os.getppid()
    assert native["command"] == [sys.executable] + sys.argv == expected
    assert native["task"] == args.task
    assert native["hostname"] == socket.gethostname()
    assert native["physical_gpu"] == os.environ["CUDA_VISIBLE_DEVICES"] == plan["assigned_gpu"]
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["OPENCV_FOR_THREADS_NUM"] == "1"
    manifest = json.loads((ROOT / plan["manifest"]).read_bytes())
    rows = {row["name"]: row for row in manifest["train"]}
    excluded = set(manifest["test_sequences"]) | {r["name"] for r in manifest["dev"]}
    assert not excluded.intersection(plan["sequence_names"])
    selected = [rows[name] for name in plan["sequence_names"]]
    assert all(len(row["rgb"]) >= 31 and len(row["tir"]) >= 31 for row in selected)
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
    before = {name: {k: v.cpu().clone() for k, v in module.state_dict().items()}
              for name, module in modules.items()}
    run = ROOT / args.run
    run.mkdir(parents=True, exist_ok=False)
    (run / "traces").mkdir()
    config = dict(plan=plan, plan_sha256=sha(args.plan), native_command=native["command"],
                  native_status=str(native_path), source_review_sha256=args.source_review_sha256,
                  timing_scope="Synchronized decode, E0, M1, actual M2 calls and initialization; "
                               "model load excluded. Concurrent load.",
                  total_parameters=sum(p.numel() for m in modules.values() for p in m.parameters()))
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    timings, progress, calls, replacements = [], [], 0, 0
    fields = ["frame", "m1_state", "m1_reliable", "m1_uncertain", "m1_lost",
              "template_write_proposed", "template_write_blocked",
              "m2_invoked", "m2_class_index", "seconds"]
    for row in selected:
        tracker.set_sequence(row["valid_frames"], 31)
        started = time.perf_counter()
        tracker.initialize(read_pair(row, 0), row["boxes"][0])
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        assert tracker.guard.record is None and tracker.guard.observer.record is None
        with (run / "traces" / (row["name"] + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            for frame in range(1, 31):
                started = time.perf_counter()
                box, route = tracker.track(read_pair(row, frame))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                assert tracker.guard.observer.calls == 1 and tracker.base.frame == frame
                assert np.isfinite(box).all() and (box[2:] > 0).all()
                record, decision = tracker.guard.record, tracker.last_m2
                assert decision["invoked"] == recovery_due(frame, tracker.valid_frames,
                                                           record["m1_state"])
                calls += int(decision["invoked"])
                replacements += int(decision["class_index"] > 0)
                writer.writerow(dict(frame=frame, **record,
                                     m2_invoked=int(decision["invoked"]),
                                     m2_class_index=decision["class_index"], seconds=elapsed))
                timings.append(elapsed)
        progress.append(dict(sequence=row["name"], frames=31))
        (run / "progress.json").write_text(json.dumps(progress) + "\n")
        print(json.dumps(progress[-1]), flush=True)
    assert all(torch.equal(v.cpu(), before[name][k])
               for name, module in modules.items() for k, v in module.state_dict().items())
    assert all(p.grad is None for m in modules.values() for p in m.parameters())
    tracker.close()
    assert model.predict == tracker.guard.observer.original
    admission(args.plan, args.source_review, args.source_review_sha256)
    timing = np.asarray(timings, np.float64)
    assert timing.shape == (93,) and np.isfinite(timing).all() and (timing > 0).all()
    np.save(run / "frame_seconds.npy", timing)
    receipt = dict(task=args.task, frames=93, sequences=3, smoke_only=True,
                   split="train", test_eligible=False, exploratory_test=False,
                   plan_sha256=sha(args.plan), state_unchanged=True,
                   optimizer_updates=0, backward_calls=0, m2_invocations=calls,
                   m2_candidate_selections=replacements, seconds=float(timing.sum()),
                   positive_recovery_exercised=replacements > 0,
                   scope="Execution witness only; zero M2 calls is valid under M1 gating. "
                         "No recovery gain, TRAIN policy acceptance or TEST admission.",
                   artifacts_sha256={p.relative_to(run).as_posix(): sha(p)
                                     for p in run.rglob("*") if p.is_file()})
    (run / "SMOKE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
