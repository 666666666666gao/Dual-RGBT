"""Evaluate an admitted fixed-final C2 policy FIT without training or selection."""
import argparse
import csv
import hashlib
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
from dual_rgbt.c2_policy_evaluation import load_completed_policy

COUNTS = {"LasHeR": (245, 220703), "RGBT210": (210, 104706),
          "RGBT234": (234, 116650)}
ENTRY = "scripts/eval_c2_policy_completed.py"


def validate_plan(plan, plan_path, root):
    assert plan["stage"] == "P2_C2_policy_evaluation"
    assert plan["launch_allowed"] is True
    assert plan["host"] in ("2025", "2026") and plan["variant"] in ("zero", "film")
    assert plan["root"] == str(root) and Path.cwd().resolve() == root.resolve()
    assert plan["task"] and Path(plan["task"]).name == plan["task"]
    assert plan["gpu"] in ("0", "1", "2", "3")
    assert Path(plan["run"]) == Path(plan["run_root"]) / plan["task"]
    assert plan["smoke_frames"] in (0, 11) and plan["cost_weight"] == .05
    smoke = plan["smoke_frames"] == 11
    assert plan["smoke"] is smoke and plan["exploratory_test"] is (not smoke)
    assert plan["test_eligible"] is plan["dev_enabled"] is False
    assert plan["command"][-4:] == ["-B", str(root / ENTRY), "--plan", str(plan_path)]
    assert sha(plan["review"]["path"]) == plan["review"]["sha256"]
    review = json.loads(Path(plan["review"]["path"]).read_bytes())
    assert review["stage"] == plan["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == plan["sources"]
    binding = hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items()
         if key not in ("launch_allowed", "review")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert binding in review["admitted_plan_bindings"]
    assert {
        ENTRY, "dual_rgbt/c2_policy_evaluation.py", "dual_rgbt/c2_policy_refresh.py",
        "dual_rgbt/c2_executor.py", "dual_rgbt/model.py", "dual_rgbt/data.py",
        "dual_rgbt/tracker.py", "scripts/benchmark_inputs.py",
    }.issubset(plan["sources"])
    for name, digest in plan["sources"].items():
        assert sha(root / name) == digest, name
    fit_item = plan["fit_evidence"]["plan"]
    assert sha(fit_item["path"]) == fit_item["sha256"]
    fit = json.loads(Path(fit_item["path"]).read_bytes())
    for name in ("host", "variant", "c2_task_id", "cost_weight"):
        assert fit[name] == plan[name], name
    assert plan["checkpoint"]["path"] == str(Path(fit["run"]) / "epoch_005.pth")
    assert sha(plan["manifest"]["path"]) == plan["manifest"]["sha256"]
    manifest = load_manifest(plan["manifest"]["path"])
    assert manifest["benchmark"] == plan["benchmark"]
    assert manifest["split"] == ("train_smoke" if smoke else "test")
    if smoke:
        assert manifest["benchmark"] == "LasHeR"
        assert len(manifest["sequences"][0]["rgb"]) >= 11
    else:
        assert (manifest["sequence_count"], manifest["frame_count"]) == COUNTS[plan["benchmark"]]
    return manifest


def track_sequences(tracker, manifest, run, smoke_frames, read_pair):
    """Use the established online protocol and export full scoring artifacts."""
    selected = manifest["sequences"][:1] if smoke_frames else manifest["sequences"]
    timings, progress = [], []
    for sequence in selected:
        name = sequence["name"]
        count = smoke_frames or len(sequence["rgb"])
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
    frames = sum(row["frames"] for row in progress)
    assert frames == (11 if smoke_frames else manifest["frame_count"])
    timing = np.asarray(timings, dtype=np.float64)
    assert timing.shape == (frames,) and np.isfinite(timing).all() and (timing > 0).all()
    np.save(run / "frame_seconds.npy", timing)
    return progress, timing


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_bytes())
    manifest = validate_plan(plan, plan_path, ROOT)
    native_path = ROOT / "runs/pipeline_tasks" / plan["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["state"] == "RUNNING" and native["pid"] == os.getppid()
    assert native["task"] == plan["task"]
    assert native["command"] == plan["command"] == [sys.executable, "-B", *sys.argv]
    assert native["physical_gpu"] == plan["gpu"] == os.environ["CUDA_VISIBLE_DEVICES"]
    torch.set_num_threads(1)
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = load_completed_policy(
        plan["fit_evidence"], plan["checkpoint"]["path"],
        plan["checkpoint"]["sha256"], device="cuda")
    from dual_rgbt.data import read_pair
    from dual_rgbt.tracker import Tracker

    before = {key: value.cpu().clone() for key, value in model.state_dict().items()}
    tracker = Tracker(model, plan["cost_weight"])
    run = Path(plan["run"])
    run.mkdir(parents=True, exist_ok=False)
    for folder in ("predictions", "groundtruth", "traces"):
        (run / folder).mkdir()
    selected = manifest["sequences"][:1] if plan["smoke"] else manifest["sequences"]
    config = dict(
        plan=plan, plan_sha256=sha(plan_path), benchmark=manifest["benchmark"],
        split=manifest["split"], split_sequences=manifest["sequence_count"],
        split_frames=manifest["frame_count"], sequences=[row["name"] for row in selected],
        native_status=str(native_path), native_command=native["command"],
        physical_gpu=native["physical_gpu"],
        total_parameters=sum(value.numel() for value in model.parameters()),
        source_sha256=plan["sources"], exploratory_test=plan["exploratory_test"],
        test_eligible=False, gpu_name=torch.cuda.get_device_name(0),
        input_sizes=dict(search=256, template=128, templates=2),
        timing_scope="Decode and initialization included; model load excluded; "
                     "concurrent system load, not isolated throughput.")
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    progress, timing = track_sequences(
        tracker, manifest, run, plan["smoke_frames"], read_pair)
    assert all(torch.equal(value.cpu(), before[key])
               for key, value in model.state_dict().items())
    assert all(value.grad is None for value in model.parameters())
    precision = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                     cudnn_benchmark=torch.backends.cudnn.benchmark)
    assert not any(precision.values())
    (run / "precision.json").write_text(json.dumps(precision) + "\n")
    complete = dict(
        stage=plan["stage"], task=plan["task"], sequences=len(progress),
        frames=len(timing), seconds=float(timing.sum()),
        checkpoint_sha256=plan["checkpoint"]["sha256"], plan_sha256=sha(plan_path),
        config_sha256=sha(run / "config.json"), smoke_only=plan["smoke"],
        optimizer_updates=0, backward_calls=0, state_unchanged=True,
        exploratory_test=plan["exploratory_test"], test_eligible=False)
    (run / ("SMOKE.json" if plan["smoke"] else "COMPLETE.json")).write_text(
        json.dumps(complete, indent=2) + "\n")
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
