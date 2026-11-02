"""TRAIN-only current-E0/candidate joins, relative loss and protection metrics."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from dual_rgbt.e0_relative_identity import relative_action_targets
from full_current_e0_supplement_contract import verify_collection
from replay_m2_train_states import box_iou, collected_iou


def targets_and_loss(logits, current_iou, candidate_iou, current_coverage, coverage, labels):
    """Keep ambiguous original candidate labels outside the loss denominator."""
    assert labels.shape == logits.shape and labels.shape[1] == candidate_iou.shape[1] + 1
    assert ((labels == -1) | (labels == 0) | (labels == 1)).all()
    known_coverage = coverage.masked_fill((labels[:, 1:] == -1)[:, :, None], 0)
    targets, trainable = relative_action_targets(
        current_iou, candidate_iou, current_coverage, known_coverage)
    assert trainable.any()
    known = torch.cat([torch.ones_like(labels[:, :1], dtype=torch.bool),
                       labels[:, 1:] != -1], 1)
    supervised_logits = logits.masked_fill(~known, float("-inf"))[trainable]
    selected = targets[trainable]
    assert torch.isfinite(supervised_logits.gather(1, selected[:, None])).all()
    return torch.nn.functional.cross_entropy(supervised_logits, selected), targets, trainable


def action_metrics(chosen, current_iou, candidate_iou, sequence_index, labels):
    """Measure proposal changes on fixed original TRAIN states, never online SR."""
    assert chosen.dtype == torch.long and chosen.shape == current_iou.shape == sequence_index.shape
    assert candidate_iou.shape[0] == len(chosen)
    assert current_iou.dtype == candidate_iou.dtype == torch.float64
    assert ((chosen >= 0) & (chosen <= candidate_iou.shape[1])).all()
    assert torch.isfinite(current_iou).all() and torch.isfinite(candidate_iou).all()
    selected_iou = candidate_iou.gather(1, (chosen - 1).clamp_min(0)[:, None]).squeeze(1)
    output_iou = torch.where(chosen == 0, current_iou, selected_iou)
    delta = output_iou - current_iou
    reliable, lost = current_iou >= .5, current_iou < .1
    macro, reliable_macro = [], []
    for sequence in sequence_index.unique():
        mask = sequence_index == sequence
        macro.append(float(delta[mask].mean()))
        if (mask & reliable).any():
            reliable_macro.append(float(delta[mask & reliable].mean()))
    assert macro and reliable_macro
    return dict(
        sets=len(chosen), sequences=len(macro),
        reliable_sets=int(reliable.sum()), lost_sets=int(lost.sum()),
        keep=int((chosen == 0).sum()), replacements=int((chosen > 0).sum()),
        reliable_to_below_half=int((reliable & (output_iou < .5)).sum()),
        lost_to_at_least_half=int((lost & (output_iou >= .5)).sum()),
        selected_original_unknown=int(((chosen > 0) & (
            labels.gather(1, chosen[:, None]).squeeze(1) == -1)).sum()),
        mean_iou_delta=float(delta.mean()),
        sequence_macro_iou_delta=sum(macro) / len(macro),
        sequence_macro_reliable_iou_delta=sum(reliable_macro) / len(reliable_macro))


def qualifies_for_train_acceptance(metrics):
    """Predeclared held TRAIN baseline counts and fixed protection/recovery gates."""
    assert metrics["reliable_sets"] == 2899 and metrics["lost_sets"] == 800
    return (metrics["reliable_to_below_half"] == 0
            and metrics["sequence_macro_reliable_iou_delta"] >= 0
            and metrics["lost_to_at_least_half"] > 179)


def load_verified_data(root, supplement_plan, supplement_acceptance):
    """Require full acceptance before reading any TRAIN targets for head updates."""
    actual = verify_collection(root, supplement_plan)
    assert json.loads((root / supplement_acceptance).read_bytes()) == actual
    plan = json.loads((root / supplement_plan).read_bytes())
    manifest = json.loads((root / plan["manifest"]).read_bytes())
    sequences = {row["name"]: row for row in manifest["train"]}
    parents = {Path(name).stem: name for name in plan["cache_sha256"]}
    fit = set(plan["fit_sequences"])
    fields = ("anchor", "current", "current_coverage", "candidates", "coverage",
              "labels", "current_iou", "candidate_iou", "sequence_index")
    parts = {part: {key: [] for key in fields} for part in ("fit", "holdout")}
    for job in plan["jobs"]:
        for name in job["sequence_names"]:
            sequence = sequences[name]
            part = parts["fit" if name in fit else "holdout"]
            with np.load(root / parents[name], allow_pickle=False) as original, \
                    np.load(root / job["run"] / (name + ".npz"), allow_pickle=False) as current:
                frames = original["frame"]
                assert np.array_equal(frames, current["frame"])
                assert np.array_equal(original["predictions"], current["predictions"])
                count = len(frames)
                groundtruth = np.asarray(sequence["boxes"])[frames]
                assert np.array_equal(original["iou"], collected_iou(original["box"], groundtruth))
                e0_iou = box_iou(original["predictions"][frames], groundtruth)
                candidate_iou = box_iou(original["box"].reshape(-1, 4),
                                       np.repeat(groundtruth, 24, axis=0)).reshape(count, 24)
                values = dict(
                    anchor=np.repeat(original["anchor"][None], count, axis=0),
                    current=current["e0_features"].copy(),
                    current_coverage=current["e0_coverage"].copy(),
                    candidates=original["features"].copy(), coverage=original["coverage"].copy(),
                    labels=original["labels"].copy(), current_iou=e0_iou, candidate_iou=candidate_iou,
                    sequence_index=np.full(count, plan["sequence_names"].index(name), np.int64))
                for key, value in values.items():
                    part[key].append(value)
    data = {part: {key: torch.from_numpy(np.concatenate(values)) for key, values in fields.items()}
            for part, fields in parts.items()}
    assert len(data["fit"]["labels"]) == 14762
    assert len(data["holdout"]["labels"]) == 3949
    assert int((data["holdout"]["current_iou"] >= .5).sum()) == 2899
    assert int((data["holdout"]["current_iou"] < .1).sum()) == 800
    return data
