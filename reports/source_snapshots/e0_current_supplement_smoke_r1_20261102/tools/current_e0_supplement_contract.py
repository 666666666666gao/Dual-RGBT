"""Bindings and CPU verification for current-E0 TRAIN evidence supplements."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ACCEPTANCE_SHA = "74eb5104ee223c8fa4eec4992c47f1e466efc6f27a9679a95d372fdd521f2522"
PARENT_SHA = "ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4194304), b""):
            digest.update(block)
    return digest.hexdigest()


def inputs(root, plan):
    assert plan["stage"] == "M2_current_E0_evidence_supplement"
    assert plan["target_host"] == "2026" and plan["target_root"] == str(root)
    assert plan["split"] == "train" and plan["test_eligible"] is False
    assert plan["training_allowed"] is False and plan["cost_weight"] == 0.
    assert plan["phase"] == "smoke" and plan["frames_per_sequence"] == 31
    assert len(plan["sequence_names"]) == len(set(plan["sequence_names"])) == 3
    assert plan["assigned_gpu"] in ("0", "1")
    assert plan["checkpoint_sha256"] == PARENT_SHA
    assert plan["acceptance_sha256"] == ACCEPTANCE_SHA
    assert sha(root / plan["acceptance"]) == ACCEPTANCE_SHA
    for name, digest in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        assert sha(root / name) == digest, name
    assert plan["input_sha256"][plan["checkpoint"]] == PARENT_SHA
    acceptance = json.loads((root / plan["acceptance"]).read_bytes())
    assert acceptance["split"] == "train" and acceptance["test_eligible"] is False
    collection_path = root / acceptance["collection_plan"]
    assert sha(collection_path) == acceptance["collection_plan_sha256"]
    collection = json.loads(collection_path.read_bytes())
    assert collection["checkpoint_sha256"] == PARENT_SHA
    assert plan["manifest"] == collection["manifest"]
    assert plan["input_sha256"][plan["manifest"]] == collection["input_sha256"][plan["manifest"]]
    manifest = json.loads((root / plan["manifest"]).read_bytes())
    sequences = {row["name"]: row for row in manifest["train"]}
    forbidden = {row["name"] for row in manifest["dev"]} | set(manifest["test_sequences"])
    assert len(sequences) == 882 and not set(plan["sequence_names"]) & forbidden
    assert set(plan["sequence_names"]) <= set(sequences)
    caches = {Path(name).stem: (name, digest)
              for name, digest in acceptance["inputs_sha256"].items()
              if name.endswith(".npz")}
    assert len(caches) == 882
    assert set(plan["cache_sha256"]) == {caches[name][0] for name in plan["sequence_names"]}
    for name in plan["sequence_names"]:
        relative, digest = caches[name]
        assert plan["cache_sha256"][relative] == digest == sha(root / relative)
    return sequences, caches


def admission(root, plan):
    sequences, caches = inputs(root, plan)
    assert plan["launch_allowed"] is True
    review_path = root / plan["source_review"]
    assert sha(review_path) == plan["source_review_sha256"]
    review = json.loads(review_path.read_bytes())
    assert review["stage"] == plan["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["target_host"] == plan["target_host"]
    assert review["target_root"] == str(root)
    assert review["source_sha256"] == plan["source_sha256"]
    draft_path = root / plan["reviewed_draft"]
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(draft_path)
    draft = json.loads(draft_path.read_bytes())
    assert draft["launch_allowed"] is False
    omit = {"launch_allowed", "source_review_sha256"}
    assert {k: v for k, v in plan.items() if k not in omit} == {
        k: v for k, v in draft.items() if k not in omit}
    return sequences, caches


def verify_arrays(arrays, sequence, original, count=31):
    frames = [f for f in sequence["valid_frames"] if 0 < f < count][::25]
    assert frames and len(sequence["rgb"]) >= count
    n = len(frames)
    shapes = dict(frame=(n,), e0_features=(n, 2, 768), e0_coverage=(n, 2),
                  normalized_box=(n, 4), geometry=(n, 3), current_box=(n, 4),
                  previous_box=(n, 4), observation=(n, 914), route=(n, 3),
                  predictions=(count, 4), frame_seconds=(count,))
    assert set(arrays) == set(shapes)
    for key, shape in shapes.items():
        assert arrays[key].shape == shape and np.isfinite(arrays[key]).all(), key
        assert arrays[key].dtype == (np.int64 if key == "frame" else
                                     np.float64 if key in ("frame_seconds", "route") else np.float32), key
    assert arrays["frame"].tolist() == frames
    assert original["frame"][original["frame"] < count].tolist() == frames
    assert np.array_equal(arrays["predictions"], original["predictions"][:count])
    assert np.array_equal(arrays["current_box"], original["predictions"][frames])
    assert np.array_equal(arrays["previous_box"], original["predictions"][np.asarray(frames) - 1])
    assert ((arrays["e0_coverage"] >= 0) & (arrays["e0_coverage"] <= 1)).all()
    assert (arrays["frame_seconds"] > 0).all()
    assert np.isin(arrays["route"][:, 0], (0, 1)).all()
    assert np.isin(arrays["route"][:, 1], (0, 1, 2, 3, 4)).all()
    for i, box in enumerate(arrays["previous_box"]):
        centre = box[:2] + box[2:] / 2
        side = math.sqrt(float(box[2] * box[3])) * 4.
        left, top = centre - side / 2
        expected = np.asarray([left, top, side], np.float32)
        assert np.array_equal(arrays["geometry"][i], expected)
        left, top, side = expected
        current = arrays["current_box"][i]
        normalized = np.concatenate([
            (current[:2] + current[2:] / 2 - [left, top]) / side,
            current[2:] / side]).astype(np.float32)
        assert np.array_equal(arrays["normalized_box"][i], normalized)
    return dict(sequence=sequence["name"], frames=count, candidate_sets=n,
                no_fully_covered_modality=int((arrays["e0_coverage"] != 1).all(1).sum()))


def verify_run(root, directory):
    config = json.loads((directory / "config.json").read_bytes())
    receipt = json.loads((directory / "COMPLETE.json").read_bytes())
    plan = config["plan"]
    sequences, caches = admission(root, plan)
    assert config["plan_sha256"] == receipt["plan_sha256"] == sha(root / config["plan_path"])
    assert json.loads((root / config["plan_path"]).read_bytes()) == plan
    native_path = root / "runs/pipeline_tasks" / plan["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == receipt["task"] == plan["task"]
    command = [str(root / "conda-envs/dual_rgbt/bin/python"),
               str(root / "scripts/collect_e0_current_supplement.py"),
               "--plan", config["plan_path"]]
    assert native["command"] == config["native_command"] == command
    assert native["physical_gpu"] == plan["assigned_gpu"]
    assert receipt["status"] == "CURRENT_E0_SUPPLEMENT_SMOKE_COMPLETE"
    assert receipt["split"] == "train" and receipt["test_eligible"] is False
    assert receipt["all_online_states_bit_equal"] and receipt["original_predictions_bit_equal"]
    assert receipt["model_unchanged"] and receipt["predict_restored"] and receipt["no_gradients"]
    assert receipt["optimizer_updates"] == receipt["backward_calls"] == 0
    assert receipt["frozen_state_tensors"] == 265
    assert receipt["anchor_encoder_calls"] == receipt["candidate_encoder_calls"] == 0
    expected = {p.relative_to(directory).as_posix(): sha(p)
                for p in directory.rglob("*") if p.is_file() and p.name != "COMPLETE.json"}
    assert receipt["artifacts_sha256"] == expected
    assert set(expected) == {"config.json", "progress.json"} | {
        name + ".npz" for name in plan["sequence_names"]}
    rows = []
    for name in plan["sequence_names"]:
        relative, _ = caches[name]
        with np.load(directory / (name + ".npz"), allow_pickle=False) as values, \
                np.load(root / relative, allow_pickle=False) as original:
            rows.append(verify_arrays(values, sequences[name], original))
    assert json.loads((directory / "progress.json").read_bytes()) == rows
    assert receipt["sequences"] == len(rows) == 3
    assert receipt["frames"] == sum(row["frames"] for row in rows) == 93
    assert receipt["candidate_sets"] == sum(row["candidate_sets"] for row in rows)
    assert receipt["supplement_encoder_calls"] == 2 * receipt["candidate_sets"]
    return dict(status="CURRENT_E0_SUPPLEMENT_SMOKE_VERIFIED", rows=rows,
                receipt_sha256=sha(directory / "COMPLETE.json"),
                native_sha256=sha(native_path), test_eligible=False,
                full_collection_allowed=False,
                scope="Paired smoke and original-cache alignment only, not full "
                      "supplement acceptance, training or benchmark gain.")
