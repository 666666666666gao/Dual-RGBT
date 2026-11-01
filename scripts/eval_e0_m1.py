"""Matched protected E0 / M1 template-commit experiment; no test selection."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_inputs import groundtruth, load_manifest, sha
from dual_rgbt.data import read_pair
from dual_rgbt.e0_m1_guard import M1TemplateGuard, load_reliability
from dual_rgbt.model import DualRGBT
from dual_rgbt.tracker import Tracker

STAGE = "E0_M1_template_guard_evaluation"
PARENT_SHA = "ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c"
HEAD_SHA = "658c84275ad1e1624d4b453fee89dd312273633664024bf0b42ae78dcc189ebf"


def check_admission(plan_path, review_path, review_sha):
    assert sha(review_path) == review_sha
    plan = json.loads(Path(plan_path).read_bytes())
    review = json.loads(Path(review_path).read_bytes())
    assert plan["stage"] == review["stage"] == STAGE
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["target_root"] == plan["target_root"] == ROOT.as_posix()
    assert review["source_sha256"] == plan["source_sha256"]
    assert review["reviewed_drafts"][Path(plan_path).name] == sha(plan_path)
    assert all(sha(ROOT / name) == digest for name, digest in plan["source_sha256"].items())
    assert all(sha(ROOT / name) == digest for name, digest in plan["input_sha256"].items())
    assert plan["checkpoint_sha256"] == PARENT_SHA
    assert plan["m1_sha256"] == HEAD_SHA
    assert plan["input_sha256"][plan["checkpoint"]] == PARENT_SHA
    assert plan["input_sha256"][plan["m1"]] == HEAD_SHA
    assert plan["factory_sha256"] == plan["input_sha256"][plan["factory"]]
    assert plan["manifest_sha256"] == plan["input_sha256"][plan["manifest"]]
    assert plan["cost_weight"] == 0.0
    assert plan["matmul_tf32"] is plan["cudnn_tf32"] is False
    assert plan["training_allowed"] is plan["test_eligible"] is False
    assert plan["mode"] in ("control", "observe", "guard")
    assert plan["smoke_frames"] in (0, 31)
    return plan, review


def shadow_step(tracker, image):
    """One unchanged E0 step from the same pre-step state, TRAIN smoke only."""
    direct = Tracker(tracker.model, tracker.cost_weight)
    direct.state = tracker.state.copy()
    direct.templates = tracker.templates.clone()
    direct.history = tracker.history.clone()
    direct.output_window = tracker.output_window
    direct.frame, direct.since_query = tracker.frame, tracker.since_query
    box, route = direct.track(image)
    return direct, box, route


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "source-review", "source-review-sha256", "task", "run"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    plan, review = check_admission(args.plan, args.source_review, args.source_review_sha256)
    smoke = plan["smoke_frames"] == 31
    native_path = ROOT / "runs/pipeline_tasks" / args.task / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["state"] == "RUNNING" and native["pid"] == os.getppid()
    assert native["command"] == [sys.executable] + sys.argv
    assert native["physical_gpu"] == os.environ["CUDA_VISIBLE_DEVICES"]
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["OPENCV_FOR_THREADS_NUM"] == "1"
    if smoke:
        raw = json.loads((ROOT / plan["manifest"]).read_bytes())
        by_name = {row["name"]: row for row in raw["train"]}
        names = ["2up", "motocross", "carturn"]
        assert plan["sequence_names"] == names
        assert not set(names) & ({row["name"] for row in raw["dev"]} | set(raw["test_sequences"]))
        selected = [by_name[name] for name in names]
        benchmark, split = "LasHeR", "train_smoke"
        split_frames = 93
        assert not plan["exploratory_test"]
    else:
        assert len(plan["smoke_dependencies"]) == 2
        for dependency in plan["smoke_dependencies"]:
            witness_run = ROOT / dependency["run"]
            witness = json.loads((witness_run / "SMOKE.json").read_bytes())
            witness_config = json.loads((witness_run / "config.json").read_bytes())
            witness_native = json.loads((ROOT / "runs/pipeline_tasks" /
                                         dependency["task"] / "status.json").read_bytes())
            assert witness_native["state"] == "COMPLETE" and witness_native["exit_code"] == 0
            assert witness_native["task"] == witness["task"] == witness_config["task"] == dependency["task"]
            assert witness_native["command"] == witness_config["native_command"]
            assert witness["config_sha256"] == sha(witness_run / "config.json")
            assert witness["smoke_only"] and witness["train_parity_steps"] == 90
            assert witness["state_unchanged"] and witness["frames"] == 93
            assert witness["optimizer_updates"] == witness["backward_calls"] == 0
            assert witness_config["mode"] == dependency["mode"]
            assert witness_config["source_review_sha256"] == args.source_review_sha256
            assert witness_config["source_sha256"] == plan["source_sha256"]
            assert witness["checkpoint_sha256"] == PARENT_SHA and witness["m1_sha256"] == HEAD_SHA
            assert witness["artifacts_sha256"] == {
                p.relative_to(witness_run).as_posix(): sha(p)
                for p in witness_run.rglob("*") if p.is_file() and p.name != "SMOKE.json"}
            witness_plan, _ = check_admission(witness_config["plan"], args.source_review,
                                               args.source_review_sha256)
            assert witness_plan["smoke_frames"] == 31
            assert sha(witness_config["plan"]) == witness_config["plan_sha256"]
        manifest = load_manifest(ROOT / plan["manifest"])
        benchmark, split = manifest["benchmark"], manifest["split"]
        assert split == "test" and plan["exploratory_test"] is True
        selected = manifest["sequences"]
        assert len(selected) == {"LasHeR": 245, "RGBT210": 210, "RGBT234": 234}[benchmark]
        split_frames = manifest["frame_count"]
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    np.random.seed(2027)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = DualRGBT(str(ROOT / plan["factory"]))
    model.load_state_dict(torch.load(ROOT / plan["checkpoint"], map_location="cpu")["net"], strict=True)
    model.cuda().eval().requires_grad_(False)
    before = {key: value.cpu().clone() for key, value in model.state_dict().items()}
    head, mean, scale = load_reliability(ROOT / plan["m1"], PARENT_SHA, "cuda")
    head_before = {key: value.cpu().clone() for key, value in head.state_dict().items()}
    direct = Tracker(model, 0.0)
    tracker = direct if plan["mode"] == "control" else M1TemplateGuard(
        direct, head, mean, scale, commit_guard=plan["mode"] == "guard")
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    for name in ("predictions", "groundtruth", "traces"):
        (run / name).mkdir()
    config = dict(vars(args), **plan, benchmark=benchmark, split=split,
                  plan_sha256=sha(args.plan), split_sequences=len(selected),
                  split_frames=split_frames, sequences=[row["name"] for row in selected],
                  shards=1, shard=0, native_status=str(native_path),
                  native_command=native["command"], physical_gpu=native["physical_gpu"],
                  total_parameters=sum(p.numel() for p in model.parameters()) +
                  (sum(p.numel() for p in head.parameters()) if plan["mode"] != "control" else 0),
                  gpu_name=torch.cuda.get_device_name(0),
                  input_sizes=dict(search=256, template=128, templates=2),
                  timing_scope="Decode, initialization, M1/observer and synchronized rollback included. "
                               "Model load excluded. Concurrent load, not isolated throughput.")
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    timings, progress, proposed, blocked, parity_steps = [], [], 0, 0, 0
    for sequence in selected:
        name = sequence["name"]
        count = 31 if smoke else len(sequence["rgb"])
        initial_box = sequence["boxes"][0] if smoke else sequence["initial_box"]
        prediction = [np.asarray(initial_box, dtype=np.float32)]
        started = time.perf_counter()
        tracker.initialize(read_pair(sequence, 0), initial_box)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        fields = ["frame", "modality", "action", "confidence", "seconds", "primary_tokens",
                  "auxiliary_tokens", "seconds_including_decode", "m1_state", "m1_reliable",
                  "m1_uncertain", "m1_lost", "template_write_proposed", "template_write_blocked"]
        with (run / "traces" / (name + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            for frame in range(1, count):
                decode_started = time.perf_counter()
                image = read_pair(sequence, frame) if smoke else None
                decode_seconds = time.perf_counter() - decode_started if smoke else 0.0
                if smoke and plan["mode"] != "control":
                    tracker.observer.begin(False)
                    shadow, expected_box, expected_route = shadow_step(direct, image)
                    previous_template = direct.templates[:, 1].clone()
                started = time.perf_counter()
                box, route = tracker.track(image if smoke else read_pair(sequence, frame))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started + decode_seconds
                assert np.isfinite(box).all() and (box[2:] > 0).all()
                record = (tracker.record if plan["mode"] != "control" else
                          dict(m1_state=-1, m1_reliable="", m1_uncertain="", m1_lost="",
                               template_write_proposed=int(direct.frame % 5 == 0 and
                                                           route["confidence"] >= .65),
                               template_write_blocked=0))
                if smoke and plan["mode"] != "control":
                    assert np.array_equal(box, expected_box)
                    assert {k: v for k, v in route.items() if k != "seconds"} == {
                        k: v for k, v in expected_route.items() if k != "seconds"}
                    assert torch.equal(direct.history, shadow.history)
                    assert (direct.frame, direct.since_query) == (shadow.frame, shadow.since_query)
                    if record["template_write_blocked"]:
                        shadow.templates[:, 1].copy_(previous_template)
                    assert torch.equal(direct.templates, shadow.templates)
                    parity_steps += 1
                proposed += record["template_write_proposed"]
                blocked += record["template_write_blocked"]
                writer.writerow(dict(frame=frame, seconds_including_decode=elapsed, **route, **record))
                prediction.append(box.copy())
                timings.append(elapsed)
        np.savetxt(run / "predictions" / (name + ".txt"), prediction, fmt="%.6f", delimiter=",")
        visible, infrared = groundtruth(sequence, benchmark)
        if benchmark == "LasHeR":
            np.savetxt(run / "groundtruth" / (name + ".txt"), visible[:count], fmt="%.9g", delimiter=",")
        else:
            folder = run / "groundtruth" / name
            folder.mkdir()
            for modality, boxes in (("visible", visible), ("infrared", infrared)):
                np.savetxt(folder / (modality + ".txt"), boxes[:count], fmt="%.9g", delimiter=",")
        progress.append(dict(sequence=name, frames=count))
        (run / "progress.json").write_text(json.dumps(progress) + "\n")
        print(json.dumps(progress[-1]), flush=True)
    if plan["mode"] != "control":
        tracker.close()
    assert all(torch.equal(v.cpu(), before[k]) for k, v in model.state_dict().items())
    assert all(torch.equal(v.cpu(), head_before[k]) for k, v in head.state_dict().items())
    assert all(p.grad is None for p in list(model.parameters()) + list(head.parameters()))
    check_admission(args.plan, args.source_review, args.source_review_sha256)
    assert sha(args.plan) == config["plan_sha256"]
    frames = sum(row["frames"] for row in progress)
    assert frames == split_frames
    timing = np.asarray(timings, dtype=np.float64)
    assert timing.shape == (frames,) and np.isfinite(timing).all() and (timing > 0).all()
    np.save(run / "frame_seconds.npy", timing)
    complete = dict(task=args.task, sequences=len(progress), frames=frames, seconds=float(timing.sum()),
                    checkpoint_sha256=PARENT_SHA, m1_sha256=HEAD_SHA,
                    config_sha256=sha(run / "config.json"), smoke_only=smoke,
                    optimizer_updates=0, backward_calls=0, state_unchanged=True,
                    template_write_proposed=proposed, template_write_blocked=blocked,
                    train_parity_steps=parity_steps, exploratory_test=not smoke, test_eligible=False,
                    artifacts_sha256={p.relative_to(run).as_posix(): sha(p)
                                      for p in run.rglob("*") if p.is_file()})
    (run / ("SMOKE.json" if smoke else "COMPLETE.json")).write_text(json.dumps(complete, indent=2) + "\n")
    print(json.dumps({k: v for k, v in complete.items() if k != "artifacts_sha256"}), flush=True)


if __name__ == "__main__":
    main()
