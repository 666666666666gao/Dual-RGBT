"""Collect compact pre-decision TRAIN states for direct-E0 policy refresh."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_rgbt.data import crop_pair, read_pair
from dual_rgbt.e0_policy_refresh import (
    file_sha, load_completed_executor, load_plan, policy_refresh_loss,
    rebuild_state, template_provenance, train_sequences, update_provenance)
from dual_rgbt.tracker import Tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan, "collect")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["gpu"]
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.backends.cudnn.benchmark = False
    sequences = train_sequences(plan["inputs"]["manifest"]["path"])
    selected = sequences[plan["shard"]::plan["shards"]]
    assert plan["label_stride"] >= 1
    if plan["smoke"]:
        selected = selected[:1]
    run = Path(plan["run"])
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(plan, indent=2) + "\n")
    model = load_completed_executor(plan)
    before = {key: tensor.cpu().clone() for key, tensor in model.state_dict().items()}
    tracker = Tracker(model, plan["cost_weight"])
    artifacts, total_frames, total_records = {}, 0, 0
    for sequence in selected:
        started = time.time()
        assert 0 in sequence["valid_frames"]
        initial = np.asarray(sequence["boxes"][0], dtype=np.float32)
        tracker.initialize(read_pair(sequence, 0), initial)
        provenance = template_provenance(initial)
        valid = set(sequence["valid_frames"])
        records, frame_count = [], 0
        last_frame = min(len(sequence["boxes"]), 11) if plan["smoke"] else len(sequence["boxes"])
        for frame in range(1, last_frame):
            image = read_pair(sequence, frame)
            previous = tracker.state.copy()
            history = tracker.history[0].cpu().numpy().copy()
            sources = {key: value.copy() for key, value in provenance.items()}
            templates = tracker.templates.clone()
            predicted, route = tracker.track(image)
            frame_count += 1
            if frame in valid and frame % plan["label_stride"] == 0:
                search, _, geometry = crop_pair(image, previous, 4., 256)
                # Read annotation only after the actual causal decision.
                truth = np.asarray(sequence["boxes"][frame], dtype=np.float32)
                left, top, side = geometry
                target = np.concatenate([
                    (truth[:2] - [left, top]) / side, truth[2:] / side]).astype(np.float32)
                record = dict(
                    frame=frame, previous=previous, history=history,
                    geometry=geometry, template_frames=sources["frames"],
                    template_boxes=sources["boxes"], target=target,
                    prediction=predicted.copy(), route=route)
                post_templates, post_history = tracker.templates.clone(), tracker.history.clone()
                post_frame, post_since = tracker.frame, tracker.since_query
                with torch.no_grad():
                    labels = policy_refresh_loss(
                        model, templates, search[None].cuda(),
                        torch.from_numpy(history)[None].cuda(),
                        torch.from_numpy(target)[None].cuda(), tracker.output_window)
                for key in ("quality_targets", "gain_targets", "primary_boxes",
                            "candidate_indices", "candidate_boxes", "query_boxes"):
                    record[key] = labels[key].cpu()
                assert np.array_equal(tracker.state, predicted)
                assert torch.equal(tracker.templates, post_templates)
                assert torch.equal(tracker.history, post_history)
                assert tracker.frame == post_frame and tracker.since_query == post_since
                record["post_history"] = post_history[0].cpu().numpy().copy()
                record["post_since_query"] = post_since
                if plan["smoke"]:
                    rebuilt = rebuild_state(sequence, record)
                    assert torch.equal(rebuilt[0], templates[0].cpu())
                    assert torch.equal(rebuilt[1], search)
                    assert torch.equal(rebuilt[2], torch.from_numpy(history))
                records.append(record)
            update_provenance(provenance, frame, route, predicted)
        expected = sum(frame in valid and frame % plan["label_stride"] == 0
                       for frame in range(1, last_frame))
        assert len(records) == expected
        path = run / (sequence["name"] + ".pth")
        torch.save(dict(sequence=sequence["name"], records=records,
                        tracked_frames=frame_count,
                        plan_sha256=file_sha(args.plan)), path)
        artifacts[path.name] = file_sha(path)
        total_frames += frame_count
        total_records += len(records)
        (run / "progress.json").write_text(json.dumps(dict(
            sequences=len(artifacts), tracked_frames=total_frames,
            records=total_records, current=sequence["name"],
            last_sequence_seconds=time.time() - started)) + "\n")
        print(sequence["name"], frame_count, len(records), flush=True)
    assert total_records > 0
    assert all(torch.equal(before[key], tensor.cpu())
               for key, tensor in model.state_dict().items())
    (run / "COMPLETE.json").write_text(json.dumps(dict(
        status="COMPLETE", stage="P1B_direct_E0_collection",
        smoke=plan["smoke"], plan_sha256=file_sha(args.plan),
        executor_sha256=plan["inputs"]["executor"]["sha256"],
        sequence_names=[s["name"] for s in selected],
        tracked_frames=total_frames, records=total_records,
        artifacts=artifacts, benchmark_evidence=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
