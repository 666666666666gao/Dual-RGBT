"""Replay a frozen M2 head on accepted TRAIN caches, stratified by E0 IoU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4194304), b""):
            digest.update(block)
    return digest.hexdigest()


def box_iou(boxes, groundtruth):
    import numpy as np
    boxes = np.asarray(boxes, dtype=np.float64)
    groundtruth = np.asarray(groundtruth, dtype=np.float64)
    assert boxes.shape == groundtruth.shape and boxes.shape[1] == 4
    assert np.isfinite(boxes).all() and np.isfinite(groundtruth).all()
    assert (boxes[:, 2:] > 0).all() and (groundtruth[:, 2:] > 0).all()
    intersection = np.maximum(
        0, np.minimum(boxes[:, :2] + boxes[:, 2:], groundtruth[:, :2] + groundtruth[:, 2:])
        - np.maximum(boxes[:, :2], groundtruth[:, :2])).prod(1)
    return intersection / (boxes[:, 2:].prod(1) + groundtruth[:, 2:].prod(1) - intersection)


def new_counts():
    return dict(sets=0, positive_available=0, no_match_truth=0,
                selections=0, no_match_selected=0, selected_positive=0,
                reliable_to_below_half=0, lost_to_at_least_half=0,
                better=0, worse=0, equal=0, iou_delta_sum=0.)


def collected_iou(boxes, groundtruth):
    """Preserve the collector's float32 box arithmetic before GT promotion."""
    import numpy as np
    result = []
    for candidates, target in zip(boxes, groundtruth):
        gt = np.asarray(target, dtype=np.float64)
        left = np.maximum(candidates[:, :2], gt[:2])
        right = np.minimum(candidates[:, :2] + candidates[:, 2:], gt[:2] + gt[2:])
        intersection = np.maximum(0, right - left).prod(1)
        result.append(intersection / (
            candidates[:, 2:].prod(1) + gt[2:].prod() - intersection))
    return np.stack(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    import numpy as np
    import torch
    root = args.root
    sys.path.insert(0, str(root))
    module = root / "dual_rgbt/e0_identity.py"
    assert sha(module) == "89b15dea17514e5516dfe43e76295662ae47da8f98ea590e3fc5565925278669"
    from dual_rgbt.e0_identity import E0CandidateIdentity

    acceptance_path = root / "reports/E0_M2_IDENTITY_FULL_COLLECTION_ACCEPTANCE_P25_20261101.json"
    acceptance_sha = sha(acceptance_path)
    assert acceptance_sha == "74eb5104ee223c8fa4eec4992c47f1e466efc6f27a9679a95d372fdd521f2522"
    acceptance = json.loads(acceptance_path.read_bytes())
    assert acceptance["split"] == "train" and acceptance["test_eligible"] is False
    plan_path = root / acceptance["collection_plan"]
    assert sha(plan_path) == acceptance["collection_plan_sha256"]
    plan = json.loads(plan_path.read_bytes())
    assert plan["split"] == "train" and plan["cost_weight"] == 0.
    manifest_path = root / plan["manifest"]
    assert sha(manifest_path) == plan["input_sha256"][plan["manifest"]]
    manifest = json.loads(manifest_path.read_bytes())
    sequences = {row["name"]: row for row in manifest["train"]}
    fit, held = set(plan["fit_sequences"]), set(plan["holdout_sequences"])
    assert len(fit) == 706 and len(held) == 176 and fit.isdisjoint(held)
    assert fit | held == set(sequences)
    checkpoint_path = root / args.checkpoint
    assert sha(checkpoint_path) == "7e7df5d25d3e8667fe38a4ca8f6ec3662308ffcb52a09bd16aecaab59a9cb4be"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["epoch"] == 9
    assert checkpoint["acceptance_sha256"] == acceptance_sha
    assert checkpoint["protected_e0_checkpoint_sha256"] == plan["checkpoint_sha256"]
    assert checkpoint["selection_partition"] == "train_component_holdout"
    torch.set_num_threads(1)
    head = E0CandidateIdentity().eval().requires_grad_(False)
    head.load_state_dict(checkpoint["net"], strict=True)
    before = {name: value.clone() for name, value in head.state_dict().items()}
    totals = {part: {state: new_counts() for state in ("reliable", "uncertain", "lost")}
              for part in ("fit", "holdout")}
    classification = {part: dict(supervised=0, correct=0, candidate_truth=0,
                                candidate_correct=0, no_match_truth=0, no_match_correct=0)
                      for part in totals}
    seen = set()
    input_bindings = {}
    for relative, digest in acceptance["inputs_sha256"].items():
        if not relative.endswith(".npz"):
            continue
        path = root / relative
        assert sha(path) == digest, relative
        name = path.stem
        assert name in sequences and name not in seen
        seen.add(name)
        part = "fit" if name in fit else "holdout"
        sequence = sequences[name]
        with np.load(path, allow_pickle=False) as arrays:
            frames = arrays["frame"]
            labels = arrays["labels"]
            count = len(frames)
            expected_frames = [frame for frame in sequence["valid_frames"]
                               if 0 < frame < len(sequence["rgb"])][::25]
            assert frames.tolist() == expected_frames
            assert arrays["predictions"].shape == (len(sequence["rgb"]), 4)
            assert labels.shape == (count, 25)
            anchor = torch.from_numpy(arrays["anchor"])[None].expand(count, -1, -1)
            with torch.no_grad():
                logits = head(anchor, torch.from_numpy(arrays["features"]),
                              torch.from_numpy(arrays["coverage"]))
            chosen = logits.argmax(1).numpy()
            groundtruth = np.asarray(sequence["boxes"])[frames]
            assert np.array_equal(arrays["iou"], collected_iou(arrays["box"], groundtruth))
            e0_iou = box_iou(arrays["predictions"][frames],
                             groundtruth)
            output_iou = e0_iou.copy()
            selected = chosen > 0
            selected_boxes = arrays["box"][np.arange(count), np.maximum(chosen - 1, 0)]
            selected_iou = box_iou(selected_boxes, groundtruth)
            output_iou[selected] = selected_iou[selected]
            positive = (labels[:, 1:] == 1).any(1)
            no_match = labels[:, 0] == 1
            supervised = (labels == 1).any(1)
            correct = labels[np.arange(count), chosen] == 1
            c = classification[part]
            for key, mask in (
                    ("supervised", supervised), ("correct", supervised & correct),
                    ("candidate_truth", positive), ("candidate_correct", positive & selected & correct),
                    ("no_match_truth", no_match), ("no_match_correct", no_match & ~selected)):
                c[key] += int(mask.sum())
            state_masks = dict(reliable=e0_iou >= .5, lost=e0_iou < .1,
                               uncertain=(e0_iou >= .1) & (e0_iou < .5))
            delta = output_iou - e0_iou
            for state, mask in state_masks.items():
                row = totals[part][state]
                for key, values in (
                        ("sets", mask), ("positive_available", mask & positive),
                        ("no_match_truth", mask & no_match), ("selections", mask & selected),
                        ("no_match_selected", mask & ~selected),
                        ("selected_positive", mask & selected & correct),
                        ("reliable_to_below_half", mask & (e0_iou >= .5) & (output_iou < .5)),
                        ("lost_to_at_least_half", mask & (e0_iou < .1) & (output_iou >= .5)),
                        ("better", mask & (delta > 0)), ("worse", mask & (delta < 0)),
                        ("equal", mask & (delta == 0))):
                    row[key] += int(values.sum())
                row["iou_delta_sum"] += float(delta[mask].sum())
            input_bindings[relative] = digest
    assert seen == fit | held
    assert sum(v["sets"] for part in totals.values() for v in part.values()) == 18711
    assert classification["holdout"]["correct"] == 2747
    assert classification["holdout"]["supervised"] == 3450
    assert classification["holdout"]["candidate_correct"] == 2685
    assert classification["holdout"]["no_match_correct"] == 62
    for part, counts in classification.items():
        saved = checkpoint["metrics"][part]
        assert counts["supervised"] == saved["supervised_sets"]
        assert counts["candidate_truth"] == saved["sets_with_positive_candidate"]
        assert counts["candidate_correct"] == saved["correct_candidate_picks"]
        assert counts["no_match_truth"] == saved["no_match_sets"]
        assert counts["no_match_correct"] == saved["correct_no_match_picks"]
        assert abs(counts["correct"] / counts["supervised"]
                   - saved["supervised_top1_positive_fraction"]) < 1e-12
    assert all(torch.equal(value, before[name]) for name, value in head.state_dict().items())
    assert not torch.cuda.is_initialized()
    result = dict(
        status="FROZEN_M2_FULL_TRAIN_STATE_REPLAY_COMPLETE", sequences=len(seen),
        candidate_sets=18711, classification=classification, states=totals,
        acceptance_sha256=acceptance_sha, checkpoint_sha256=sha(checkpoint_path),
        manifest_sha256=sha(manifest_path), collection_plan_sha256=sha(plan_path),
        cache_sha256=input_bindings, identity_source_sha256=sha(module),
        replay_source_sha256=sha(Path(__file__)),
        overlap_check="All stored candidate IoUs exactly match original collector "
                      "arithmetic. Proposal comparisons use float64 for both boxes; "
                      "training labels and model classification remain unchanged.",
        cuda_initialized=False, model_unchanged=True, optimizer_updates=0,
        test_eligible=False, scientific_result_audit_accepted=False,
        scope="Frozen head on original paired E0 cost0 TRAIN trajectories. No M1 "
              "predictions, online interventions or independent benchmark. "
              "Upstream E0 saw component holdout. Selected overlaps describe "
              "instantaneous proposals, not closed-loop recovery or identity proof. "
              "No threshold, seed or checkpoint selected; cache bytes and saved "
              "holdout classification counts verified, not a new source admission.")
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cache_sha256"}), flush=True)


if __name__ == "__main__":
    main()
