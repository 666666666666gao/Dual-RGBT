"""Build causal context tensors from already accepted TRAIN caches."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from dual_rgbt.e0_context_advantage import context_features
from dual_rgbt.e0_relative_identity import E0RelativeIdentity
from current_e0_supplement_contract import sha
from relative_e0_train_data import load_verified_data


def sequence_weights(sequence_index, eligible):
    """Fixed full-partition weights, mean-reduced over uniformly sampled sets."""
    assert sequence_index.shape == eligible.shape
    assert eligible.dtype == torch.bool
    result = torch.zeros(len(sequence_index), dtype=torch.float32)
    sequences, counts = sequence_index[eligible].unique(return_counts=True)
    for sequence, count in zip(sequences, counts):
        result[eligible & sequence_index.eq(sequence)] = (
            len(sequence_index) / (len(sequences) * int(count)))
    return result


def load_context_data(root, relative_plan):
    """Verify collection and reuse its exact ordering; never load TEST."""
    root = Path(root)
    plan = json.loads((root / relative_plan).read_bytes())
    assert plan["split"] == "train" and plan["test_eligible"] is False
    data = load_verified_data(root, plan["collection_plan"], plan["collection_acceptance"])
    collection = json.loads((root / plan["collection_plan"]).read_bytes())
    parents = {Path(name).stem: name for name in collection["cache_sha256"]}
    fit = set(collection["fit_sequences"])
    fields = ("detector", "current_box", "candidate_boxes", "previous_box", "history", "route")
    metadata = {part: {name: [] for name in fields} for part in data}
    for job in collection["jobs"]:
        for name in job["sequence_names"]:
            part = metadata["fit" if name in fit else "holdout"]
            with np.load(root / parents[name], allow_pickle=False) as original, \
                    np.load(root / job["run"] / (name + ".npz"), allow_pickle=False) as current:
                frames = original["frame"]
                assert np.array_equal(frames, current["frame"])
                assert np.array_equal(original["predictions"], current["predictions"])
                assert np.array_equal(current["previous_box"], original["predictions"][frames - 1])
                assert np.array_equal(current["current_box"], original["predictions"][frames])
                assert current["observation"].shape == (len(frames), 914)
                values = dict(
                    detector=original["score"].copy(), candidate_boxes=original["box"].copy(),
                    current_box=current["current_box"].copy(),
                    previous_box=current["previous_box"].copy(),
                    # Four 32-D preview vectors precede the actual 8-D input history.
                    history=current["observation"][:, 128:136].copy(),
                    route=current["route"].copy())
                for key, value in values.items():
                    part[key].append(value)
    head = E0RelativeIdentity().cpu().eval().requires_grad_(False)
    assert sha(root / plan["initialization"]) == plan["initialization_sha256"]
    head.load_state_dict(torch.load(root / plan["initialization"], map_location="cpu")["net"],
                         strict=True)
    result = {}
    with torch.no_grad():
        for part, original in data.items():
            meta = {key: torch.from_numpy(np.concatenate(values))
                    for key, values in metadata[part].items()}
            scores = torch.cat([
                head(*(original[key][start:start + 512] for key in (
                    "anchor", "current", "current_coverage", "candidates", "coverage")))
                for start in range(0, len(original["labels"]), 512)])
            features, observed = context_features(
                scores, meta["detector"], original["current_coverage"], original["coverage"],
                meta["current_box"], meta["candidate_boxes"], meta["previous_box"],
                meta["history"], meta["route"])
            degrading = (observed & original["current_iou"][:, None].ge(.5)
                         & original["candidate_iou"].lt(original["current_iou"][:, None]))
            result[part] = dict(
                features=features, observed=observed,
                **{key: original[key] for key in
                   ("current_iou", "candidate_iou", "sequence_index", "labels")},
                advantage_weight=sequence_weights(original["sequence_index"], observed.any(1)),
                protection_weight=sequence_weights(original["sequence_index"], degrading.any(1)))
    assert not torch.cuda.is_initialized()
    assert all(not value.requires_grad for fields in result.values() for value in fields.values())
    return result
