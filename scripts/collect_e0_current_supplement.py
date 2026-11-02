"""Collect only missing current-E0 evidence beside immutable M2 TRAIN caches."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tools")]
from current_e0_supplement_contract import admission, sha, verify_arrays, verify_native
from dual_rgbt.data import read_pair
from dual_rgbt.e0_current_evidence import current_proposal_evidence
from dual_rgbt.e0_observation import E0Observation
from check_e0_observation_train import equal_state, snapshot
from profile_crop_api import six_channel_crop
from run_benchmark_tracking import build_model, build_tracker, freeze
from train_global_evidence_policy import cpu_state


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_bytes())
    sequences, caches = admission(ROOT, plan)
    native = json.loads((ROOT / "runs/pipeline_tasks" / plan["task"] / "status.json").read_bytes())
    verify_native(ROOT, plan, native, args.plan, "RUNNING")
    assert native["pid"] == os.getppid()
    assert native["command"] == [sys.executable] + sys.argv
    assert native["physical_gpu"] == plan["assigned_gpu"] == os.environ["CUDA_VISIBLE_DEVICES"]
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["OPENCV_FOR_THREADS_NUM"] == "1"
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Preserve the original collection's crop binding before importing Tracker.
    assert "dual_rgbt.tracker" not in sys.modules
    model = build_model("e0", None)
    model.load_state_dict(torch.load(ROOT / plan["checkpoint"], map_location="cpu")["net"], strict=True)
    freeze(model)
    before = cpu_state(model)
    assert len(before) == 265 and sum(p.numel() for p in model.parameters()) == 103374215
    control, observed = [build_tracker("e0", model, 0., 0) for _ in range(2)]
    import dual_rgbt.tracker as tracker_module
    assert tracker_module.crop_pair is six_channel_crop
    original_predict = model.predict
    observer = E0Observation(model)
    directory = ROOT / plan["run"]
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(plan=plan, plan_path=args.plan, plan_sha256=sha(plan_path),
                  native_command=native["command"],
                  timing_scope="Paired E0 init/tracking, decoding, two extra fixed-template "
                               "encoders/ROI and observations; excludes model loading.")
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    progress, sets, started = [], 0, time.time()
    for name in plan["sequence_names"]:
        sequence = sequences[name]
        relative, digest = caches[name]
        assert sha(ROOT / relative) == digest
        with np.load(ROOT / relative, allow_pickle=False) as original:
            count = plan["frames_per_sequence"]
            assert len(sequence["rgb"]) >= count
            sample_ids = [f for f in sequence["valid_frames"] if 0 < f < count][::25]
            assert original["frame"][original["frame"] < count].tolist() == sample_ids
            timings, predictions, samples = [], [], []
            torch.cuda.synchronize()
            initial_started = time.perf_counter()
            image = read_pair(sequence, 0)
            for tracker in (control, observed):
                tracker.initialize(image, sequence["boxes"][0])
            equal_state(snapshot(control), snapshot(observed))
            assert np.array_equal(observed.state, original["predictions"][0])
            fixed_templates = observed.templates.clone()
            fixed_before = fixed_templates.cpu().clone()
            predictions.append(observed.state.copy())
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - initial_started)
            for frame in range(1, count):
                torch.cuda.synchronize()
                frame_started = time.perf_counter()
                image = read_pair(sequence, frame)
                previous = observed.state.copy()
                assert np.array_equal(previous, original["predictions"][frame - 1])
                observer.begin(False)
                control_box, control_route = control.track(image)
                assert observer.calls == 1 and observer.record is None
                observer.begin(frame in sample_ids)
                box, route = observed.track(image)
                assert observer.calls == 1
                assert np.array_equal(box, control_box)
                assert np.array_equal(box, original["predictions"][frame])
                assert {k: v for k, v in route.items() if k != "seconds"} == {
                    k: v for k, v in control_route.items() if k != "seconds"}
                equal_state(snapshot(control), snapshot(observed))
                predictions.append(box.copy())
                if frame in sample_ids:
                    state = snapshot(observed)
                    observation = observer.record["features"].copy()
                    search, _, geometry = six_channel_crop(image, previous, 4., 256)
                    evidence = current_proposal_evidence(
                        model, fixed_templates, search[None].cuda(), geometry, box)
                    equal_state(snapshot(observed), state)
                    assert torch.equal(fixed_templates.cpu(), fixed_before)
                    samples.append(dict(
                        frame=frame, e0_features=evidence["features"].cpu().numpy(),
                        e0_coverage=evidence["coverage"].cpu().numpy(),
                        normalized_box=evidence["normalized_box"], geometry=geometry,
                        current_box=box.copy(), previous_box=previous,
                        observation=observation,
                        route=np.asarray([route["modality"], route["action"], route["confidence"]], np.float64)))
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - frame_started)
            assert [row["frame"] for row in samples] == sample_ids
            arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
            arrays["frame"] = arrays["frame"].astype(np.int64)
            arrays["predictions"] = np.asarray(predictions, np.float32)
            arrays["frame_seconds"] = np.asarray(timings, np.float64)
            result = verify_arrays(arrays, sequence, original, count)
            np.savez(directory / (name + ".npz"), **arrays)
            progress.append(result)
            sets += result["candidate_sets"]
            (directory / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    observer.close()
    assert model.predict == original_predict
    assert all(torch.equal(before[key], value) for key, value in cpu_state(model).items())
    assert all(p.grad is None for p in model.parameters())
    admission(ROOT, plan)
    assert sha(plan_path) == config["plan_sha256"]
    receipt = dict(
        status="CURRENT_E0_SUPPLEMENT_SMOKE_COMPLETE", task=plan["task"],
        plan_sha256=sha(plan_path), split="train", test_eligible=False,
        sequences=3, frames=93, candidate_sets=sets, supplement_encoder_calls=2 * sets,
        anchor_encoder_calls=0, candidate_encoder_calls=0, frozen_state_tensors=len(before),
        original_predictions_bit_equal=True, all_online_states_bit_equal=True,
        model_unchanged=True, predict_restored=True, no_gradients=True,
        optimizer_updates=0, backward_calls=0, wall_seconds=time.time() - started,
        artifacts_sha256={p.relative_to(directory).as_posix(): sha(p)
                          for p in directory.rglob("*") if p.is_file()},
        scope="Only missing current E0 dual-modal ROI and914-D observations on "
              "paired original TRAIN states. Original24 candidates/anchors unchanged. "
              "No GT labels, policy commits, training, full collection or benchmark.")
    (directory / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
