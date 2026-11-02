"""Direct-E0 policy refresh primitives and compact causal state records."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dual_rgbt.losses import localization_loss


POLICY_PREFIXES = ("preview.", "quality.", "gain.")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plan(path, mode):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    assert plan["stage"] == "P1B_direct_E0"
    assert plan["mode"] == mode and plan["host"] == "2026"
    assert Path.cwd().resolve() == Path(plan["root"]).resolve()
    assert plan["launch_allowed"] is True
    review = json.loads(Path(plan["review"]["path"]).read_text(encoding="utf-8"))
    assert file_sha(plan["review"]["path"]) == plan["review"]["sha256"]
    assert review["stage"] == "P1B_direct_E0"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == plan["sources"]
    for name, digest in plan["sources"].items():
        assert file_sha(name) == digest, name
    for item in plan["inputs"].values():
        assert file_sha(item["path"]) == item["sha256"], item["path"]
    if not plan["smoke"]:
        acceptance = plan["smoke_acceptance"]
        assert file_sha(acceptance["path"]) == acceptance["sha256"]
        smoke = json.loads(Path(acceptance["path"]).read_text())
        assert smoke["status"] == "COMPLETE" and smoke["smoke"] is True
        assert smoke["stage"] == "P1B_direct_E0_fit" and smoke["steps"] == 2
        assert smoke["executor_preserved"] and smoke["all_policy_modules_changed"]
        assert smoke["strict_reload"] and smoke["sources"] == plan["sources"]
        assert smoke["executor_sha256"] == plan["inputs"]["executor"]["sha256"]
        assert smoke["closed_loop_round"] == plan["round"]
        assert smoke["branch"] == plan["branch"]
    # Source admission must also bind the exact execution recipe and input map.
    binding = {key: value for key, value in plan.items()
               if key not in ("review", "launch_allowed")}
    digest = hashlib.sha256(json.dumps(
        binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest in review["admitted_plan_bindings"]
    return plan


def train_sequences(manifest_path):
    document = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    sequences = document["train"]
    names = [row["name"] for row in sequences]
    assert len(names) == len(set(names)) == 882
    assert set(names).isdisjoint(row["name"] for row in document["dev"])
    assert set(names).isdisjoint(document["test_sequences"])
    return sequences


def load_completed_executor(plan):
    from dual_rgbt.model import DualRGBT

    inputs = plan["inputs"]
    terminal = json.loads(Path(inputs["p1a_complete"]["path"]).read_text())
    assert terminal["status"] == "COMPLETE"
    assert terminal["epochs"] == 20 and terminal["steps"] == 50000
    assert terminal["retained_epochs"] == [16, 17, 18, 19, 20]
    native = json.loads(Path(inputs["p1a_native"]["path"]).read_text())
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == plan["p1a_task_id"]
    saved = torch.load(inputs["executor"]["path"], map_location="cpu")
    assert saved["epoch"] == 20 and saved["step"] == 50000
    config = json.loads(Path(inputs["p1a_config"]["path"]).read_text())
    assert saved["config"] == config
    assert config["hashes"]["cadtrack"] == inputs["factory"]["sha256"]
    assert config["hashes"]["manifest"] == inputs["p1a_manifest"]["sha256"]
    assert config["policy_frozen"] is True
    original = train_sequences(inputs["p1a_manifest"]["path"])
    relocated = train_sequences(inputs["manifest"]["path"])
    for old, new in zip(original, relocated):
        assert old["name"] == new["name"]
        assert old["boxes"] == new["boxes"]
        assert old["valid_frames"] == new["valid_frames"]
        for modality in ("rgb", "tir"):
            assert [Path(p).name for p in old[modality]] == [
                Path(p).name for p in new[modality]]
    command = native["command"]
    assert command[command.index("--run") + 1] == config["run"]
    assert command[command.index("--cadtrack") + 1] == config["cadtrack"]
    assert command[command.index("--manifest") + 1] == config["manifest"]
    assert command[command.index("--parent") + 1] == config["parent"]
    model = DualRGBT(inputs["factory"]["path"])
    model.load_state_dict(saved["net"], strict=True)
    if "behavior" in inputs:
        behavior = torch.load(inputs["behavior"]["path"], map_location="cpu")
        assert behavior["executor_sha256"] == inputs["executor"]["sha256"]
        for key, tensor in saved["net"].items():
            if not key.startswith(POLICY_PREFIXES):
                assert torch.equal(tensor, behavior["net"][key]), key
        model.load_state_dict(behavior["net"], strict=True)
    return model.cuda().eval()


def template_provenance(initial_box):
    return {"frames": np.zeros((2, 2), dtype=np.int64),
            "boxes": np.tile(np.asarray(initial_box, dtype=np.float32), (2, 2, 1))}


def update_provenance(provenance, frame, route, predicted_box):
    if frame % 5 == 0 and route["confidence"] >= .65:
        modalities = [route["modality"]]
        if route["action"]:
            modalities.append(1 - route["modality"])
        for modality in modalities:
            provenance["frames"][1, modality] = frame
            provenance["boxes"][1, modality] = predicted_box


def rebuild_state(sequence, record):
    from dual_rgbt.data import crop_pair, read_pair

    templates = torch.empty(2, 6, 128, 128)
    for slot in range(2):
        for modality in range(2):
            image = read_pair(sequence, int(record["template_frames"][slot, modality]))
            crop = crop_pair(image, record["template_boxes"][slot, modality], 2., 128)[0]
            templates[slot, modality * 3:(modality + 1) * 3] = crop[
                modality * 3:(modality + 1) * 3]
    image = read_pair(sequence, int(record["frame"]))
    search, _, geometry = crop_pair(image, record["previous"], 4., 256)
    assert np.array_equal(geometry, record["geometry"])
    return templates, search, torch.from_numpy(record["history"]), torch.from_numpy(record["target"])


def configure_policy_refresh(model):
    """Keep executor parameters and evaluation buffers fixed during refresh."""
    model.requires_grad_(False)
    for module in (model.preview, model.quality, model.gain):
        module.requires_grad_(True)
    model.eval()
    return [
        {"params": list(model.preview.parameters()), "lr": 1e-5},
        {"params": list(model.quality.parameters()) + list(model.gain.parameters()),
         "lr": 1e-4},
    ]


def policy_refresh_loss(model, templates, search, history, target, window):
    """Recompute four-action targets from this update's current preview.

    Inputs describe a saved pre-decision TRAIN state. This function never
    updates the tracker. Targets use the existing direct-E0 localization-loss
    utility, not candidate-E1 labels or a detached cached policy feature.
    """
    assert window is not None
    preview = model.preview_state(templates, search, history)
    primaries, candidates, qualities, gains, outcomes = [], [], [], [], []
    with torch.no_grad():
        for modality in range(2):
            primary = model.primary(templates, search, modality, window)
            indices, boxes = model.candidates(primary, preview, modality)
            main_loss, _ = localization_loss(primary["box"], target)
            action_gains, action_boxes = [], []
            for action in range(1, 5):
                refined = model.correct(
                    templates, search, primary, modality, action,
                    indices[:, action - 1], boxes[:, action - 1], window)
                refined_loss, _ = localization_loss(refined["box"], target)
                action_gains.append(main_loss - refined_loss)
                action_boxes.append(refined["box"])
            primaries.append(primary)
            candidates.append((indices, boxes))
            qualities.append(-main_loss)
            gains.append(torch.stack(action_gains, 1))
            outcomes.append(torch.stack(action_boxes, 1))
    quality_targets = torch.stack(qualities, 1)
    quality_loss = F.smooth_l1_loss(preview["quality"], quality_targets)
    gain_loss = search.new_zeros(())
    for modality in range(2):
        indices, boxes = candidates[modality]
        predicted = model.gain_predictions(
            primaries[modality], preview, modality, indices, boxes)[:, 1:]
        gain_loss = gain_loss + F.smooth_l1_loss(predicted, gains[modality]) / 2
    centre = target[:, :2] + target[:, 2:] / 2
    inside = torch.isfinite(centre).all(1) & ((centre >= 0) & (centre < 1)).all(1)
    spatial_loss = search.new_zeros(())
    if inside.any():
        cell = (centre[inside] * 8).long()
        target_cell = cell[:, 1] * 8 + cell[:, 0]
        spatial_loss = sum(
            F.cross_entropy(scores[inside].flatten(1), target_cell)
            for scores in preview["maps"]) / 2
    loss = quality_loss + gain_loss + .1 * spatial_loss
    assert torch.isfinite(loss)
    return {
        "loss": loss, "quality_loss": quality_loss, "gain_loss": gain_loss,
        "spatial_loss": spatial_loss,
        "quality_targets": quality_targets,
        "gain_targets": torch.stack(gains, 1),
        "primary_boxes": torch.stack([p["box"] for p in primaries], 1),
        "candidate_indices": torch.stack([p[0] for p in candidates], 1),
        "candidate_boxes": torch.stack([p[1] for p in candidates], 1),
        "query_boxes": torch.stack(outcomes, 1),
    }
