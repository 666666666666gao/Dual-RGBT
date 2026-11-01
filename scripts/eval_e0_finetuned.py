"""Evaluate a strictly identified E0 checkpoint without training or selection."""
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
from dual_rgbt.model import DualRGBT
from dual_rgbt.tracker import Tracker


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "checkpoint-sha256", "factory", "factory-sha256",
                 "manifest", "manifest-sha256", "source-review", "source-review-sha256",
                 "task", "run"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cost-weight", type=float, default=0.05)
    parser.add_argument("--smoke-frames", type=int, choices=(0, 11), default=0)
    args = parser.parse_args()
    assert sha(args.source_review) == args.source_review_sha256
    review = json.loads(Path(args.source_review).read_bytes())
    assert review["stage"] == "E0_finetuned_evaluation"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert all(sha(ROOT / name) == digest for name, digest in review["source_sha256"].items())
    assert args.cost_weight >= 0 and np.isfinite(args.cost_weight)
    assert sha(args.checkpoint) == args.checkpoint_sha256
    assert sha(args.factory) == args.factory_sha256
    assert sha(args.manifest) == args.manifest_sha256
    manifest = load_manifest(args.manifest)
    smoke = args.smoke_frames == 11
    assert manifest["split"] == ("train_smoke" if smoke else "test")
    if smoke:
        assert manifest["benchmark"] == "LasHeR"
        assert len(manifest["sequences"][0]["rgb"]) >= 11
    else:
        assert manifest["sequence_count"] == {
            "LasHeR": 245, "RGBT210": 210, "RGBT234": 234}[manifest["benchmark"]]
    native_path = ROOT / "runs/pipeline_tasks" / args.task / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["state"] == "RUNNING" and native["pid"] == os.getppid()
    assert native["command"] == [sys.executable] + sys.argv
    assert native["physical_gpu"] == os.environ["CUDA_VISIBLE_DEVICES"]
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = DualRGBT(args.factory)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["net"], strict=True)
    model.cuda().eval().requires_grad_(False)
    before = {key: tensor.cpu().clone() for key, tensor in model.state_dict().items()}
    tracker = Tracker(model, args.cost_weight)
    selected = manifest["sequences"][:1] if smoke else manifest["sequences"]
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    for folder in ("predictions", "groundtruth", "traces"):
        (run / folder).mkdir()
    config = dict(vars(args), stage="E0_finetuned_evaluation",
                  benchmark=manifest["benchmark"], split=manifest["split"],
                  split_sequences=manifest["sequence_count"],
                  split_frames=manifest["frame_count"],
                  sequences=[row["name"] for row in selected],
                  shards=1, shard=0, native_status=str(native_path),
                  native_command=native["command"], physical_gpu=native["physical_gpu"],
                  total_parameters=sum(value.numel() for value in model.parameters()),
                  source_sha256=review["source_sha256"],
                  exploratory_test=not smoke, test_eligible=False,
                  gpu_name=torch.cuda.get_device_name(0),
                  input_sizes=dict(search=256, template=128, templates=2),
                  timing_scope="Decode and initialization included; model load excluded; "
                               "concurrent system load, not isolated throughput.")
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    timings, progress = [], []
    for sequence in selected:
        name = sequence["name"]
        count = args.smoke_frames or len(sequence["rgb"])
        prediction = [np.asarray(sequence["initial_box"], dtype=np.float32)]
        started = time.perf_counter()
        tracker.initialize(read_pair(sequence, 0), sequence["initial_box"])
        timings.append(time.perf_counter() - started)
        with (run / "traces" / (name + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, [
                "frame", "modality", "action", "confidence", "seconds",
                "primary_tokens", "auxiliary_tokens", "seconds_including_decode"])
            writer.writeheader()
            for frame in range(1, count):
                started = time.perf_counter()
                box, route = tracker.track(read_pair(sequence, frame))
                elapsed = time.perf_counter() - started
                assert np.isfinite(box).all() and (box[2:] > 0).all()
                prediction.append(box.copy())
                timings.append(elapsed)
                writer.writerow(dict(frame=frame, seconds_including_decode=elapsed, **route))
        np.savetxt(run / "predictions" / (name + ".txt"), prediction,
                   fmt="%.6f", delimiter=",")
        visible, infrared = groundtruth(sequence, manifest["benchmark"])
        if manifest["benchmark"] == "LasHeR":
            np.savetxt(run / "groundtruth" / (name + ".txt"), visible[:count],
                       fmt="%.9g", delimiter=",")
        else:
            folder = run / "groundtruth" / name
            folder.mkdir()
            for modality, boxes in (("visible", visible), ("infrared", infrared)):
                np.savetxt(folder / (modality + ".txt"), boxes[:count],
                           fmt="%.9g", delimiter=",")
        progress.append(dict(sequence=name, frames=count))
        (run / "progress.json").write_text(json.dumps(progress) + "\n")
        print(json.dumps(progress[-1]), flush=True)
    assert all(torch.equal(value.cpu(), before[key])
               for key, value in model.state_dict().items())
    assert all(value.grad is None for value in model.parameters())
    frames = sum(row["frames"] for row in progress)
    assert frames == (11 if smoke else manifest["frame_count"])
    timing = np.asarray(timings, dtype=np.float64)
    assert timing.shape == (frames,) and np.isfinite(timing).all() and (timing > 0).all()
    np.save(run / "frame_seconds.npy", timing)
    complete = dict(task=args.task, sequences=len(progress), frames=frames,
                    seconds=float(timing.sum()), checkpoint_sha256=args.checkpoint_sha256,
                    config_sha256=sha(run / "config.json"), smoke_only=smoke,
                    optimizer_updates=0, backward_calls=0, state_unchanged=True,
                    exploratory_test=not smoke, test_eligible=False)
    (run / ("SMOKE.json" if smoke else "COMPLETE.json")).write_text(
        json.dumps(complete, indent=2) + "\n")
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
