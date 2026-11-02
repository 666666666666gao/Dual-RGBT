import json

import numpy as np
import pytest
import torch
from torch import nn

from dual_rgbt.e0_policy_refresh import (
    configure_policy_refresh, file_sha, load_plan, policy_refresh_loss,
    rebuild_state, template_provenance, update_provenance)


class TinyExecutor(nn.Module):
    """Exercise the policy/executor gradient boundary without a CADTrack load."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.decode = nn.BatchNorm1d(1)
        self.box_head = nn.Linear(1, 1)
        self.preview = nn.Linear(1, 1)
        self.quality = nn.Linear(1, 2)
        self.gain = nn.Linear(1, 4)
        self.calls = []

    def preview_state(self, templates, search, history):
        feature = self.preview(search + history)
        maps = feature[:, :, None] * torch.arange(
            64, device=feature.device).reshape(1, 8, 8) / 64
        return {"quality": self.quality(feature), "feature": feature,
                "maps": [maps, maps]}

    def primary(self, templates, search, modality, window):
        assert not torch.is_grad_enabled()
        self.calls.append(("primary", modality, window))
        box = search.new_tensor([[.5, .5, .3, .3]]).expand(len(search), -1)
        return {"box": box}

    def candidates(self, primary, preview, modality):
        # A changed preview must change the counterfactual target next update.
        indices = (preview["feature"] > 0).long().expand(-1, 4)
        return indices, primary["box"][:, None].expand(-1, 4, -1)

    def correct(self, templates, search, primary, modality, action,
                candidate, box, window):
        assert not torch.is_grad_enabled()
        self.calls.append(("correct", modality, action, window))
        refined = box.clone()
        refined[:, 0] += .01 * action + .05 * candidate
        return {"box": refined}

    def gain_predictions(self, primary, preview, modality, indices, boxes):
        predicted = self.gain(preview["feature"])
        return torch.cat((predicted.new_zeros(len(predicted), 1), predicted), 1)


def inputs():
    return (torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 1),
            torch.tensor([[.3, .3, .3, .3]]).expand(2, -1).clone(),
            torch.ones(1))


def test_refresh_trains_all_policy_modules_and_preserves_executor_and_inputs():
    torch.manual_seed(7)
    model = TinyExecutor()
    groups = configure_policy_refresh(model)
    assert [group["lr"] for group in groups] == [1e-5, 1e-4]
    assert not model.training and not model.decode.training
    before = {key: value.clone() for key, value in model.state_dict().items()}
    batch = inputs()
    saved = [value.clone() for value in batch]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    result = policy_refresh_loss(model, *batch)
    assert result["quality_targets"].shape == (2, 2)
    assert result["gain_targets"].shape == (2, 2, 4)
    result["loss"].backward()
    for module in (model.preview, model.quality, model.gain):
        assert any(value.grad is not None and value.grad.abs().sum() > 0
                   for value in module.parameters())
    for module in (model.backbone, model.decode, model.box_head):
        assert all(value.grad is None for value in module.parameters())
    optimizer.step()
    after = model.state_dict()
    for key, value in before.items():
        if key.startswith(("backbone.", "decode.", "box_head.")):
            assert torch.equal(value, after[key])
    assert all(torch.equal(a, b) for a, b in zip(batch, saved))
    assert [(call[1], call[2]) for call in model.calls if call[0] == "correct"] == [
        (modality, action) for modality in range(2) for action in range(1, 5)]
    assert all(call[-1] is batch[-1] for call in model.calls)


def test_refresh_recomputes_targets_after_current_preview_changes():
    model = TinyExecutor()
    configure_policy_refresh(model)
    with torch.no_grad():
        model.preview.weight.zero_()
        model.preview.bias.fill_(-1)
    first = policy_refresh_loss(model, *inputs())
    with torch.no_grad():
        model.preview.bias.fill_(1)
    second = policy_refresh_loss(model, *inputs())
    assert not first["gain_targets"].requires_grad
    assert not second["gain_targets"].requires_grad
    assert not torch.equal(first["gain_targets"], second["gain_targets"])


def test_off_crop_target_does_not_enter_spatial_ce_but_keeps_head_labels():
    model = TinyExecutor()
    configure_policy_refresh(model)
    batch = list(inputs())
    batch[3][:, 0] = 2.
    result = policy_refresh_loss(model, *batch)
    assert result["spatial_loss"] == 0
    assert result["quality_targets"].shape == (2, 2)
    assert result["gain_targets"].shape == (2, 2, 4)
    result["loss"].backward()
    assert model.preview.weight.grad.abs().sum() > 0


def test_template_sources_follow_tracker_cadence_modality_and_query():
    initial = np.array([10, 20, 30, 40], dtype=np.float32)
    sources = template_provenance(initial)
    saved = {key: value.copy() for key, value in sources.items()}
    route = dict(confidence=.7, modality=1, action=0)
    box = initial + 1
    update_provenance(sources, 4, route, box)
    assert np.array_equal(sources["frames"], saved["frames"])
    update_provenance(sources, 5, route, box)
    assert sources["frames"].tolist() == [[0, 0], [0, 5]]
    assert np.array_equal(saved["boxes"][1, 1], initial)
    route["action"] = 1
    update_provenance(sources, 10, route, box + 1)
    assert sources["frames"].tolist() == [[0, 0], [10, 10]]
    route["confidence"] = .64
    update_provenance(sources, 15, route, box + 2)
    assert sources["frames"].tolist() == [[0, 0], [10, 10]]


def test_rebuild_uses_each_saved_template_origin_and_original_search(monkeypatch):
    import dual_rgbt.data as data

    def read(sequence, frame):
        return frame

    def crop(frame, box, factor, size):
        tensor = torch.full((6, size, size), float(frame + box[0]))
        return tensor, None, np.array([1, 2, 3], dtype=np.float32)

    monkeypatch.setattr(data, "read_pair", read)
    monkeypatch.setattr(data, "crop_pair", crop)
    record = dict(
        frame=9, previous=np.array([4, 5, 6, 7]),
        template_frames=np.array([[0, 0], [5, 8]]),
        template_boxes=np.array([[[1, 2, 3, 4], [1, 2, 3, 4]],
                                 [[2, 3, 4, 5], [3, 4, 5, 6]]]),
        geometry=np.array([1, 2, 3], dtype=np.float32),
        history=np.arange(8, dtype=np.float32),
        target=np.array([.1, .2, .3, .4], dtype=np.float32))
    templates, search, history, target = rebuild_state({}, record)
    assert templates[0].eq(1).all()
    assert templates[1, :3].eq(7).all()
    assert templates[1, 3:].eq(11).all()
    assert search.eq(13).all()
    assert torch.equal(history, torch.from_numpy(record["history"]))
    assert torch.equal(target, torch.from_numpy(record["target"]))


def test_disabled_plan_rejected_before_review_or_gpu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(stage="P1B_direct_E0", mode="collect",
                                   host="2026", root=str(tmp_path),
                                   launch_allowed=False)))
    with pytest.raises(AssertionError):
        load_plan(path, "collect")


def test_mutated_execution_recipe_rejected_by_exact_review_binding(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "entry.py"
    source.write_text("pass\n")
    plan = dict(stage="P1B_direct_E0", mode="collect", host="2026",
                root=str(tmp_path), sources={str(source): file_sha(source)},
                inputs={}, seed=2027, smoke=True)
    binding = hashlib.sha256(json.dumps(
        plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(dict(
        stage="P1B_direct_E0", status="SOURCE_PASS", blocking_issues=0,
        sources=plan["sources"], admitted_plan_bindings=[binding])))
    plan.update(launch_allowed=True, review=dict(
        path=str(review_path), sha256=file_sha(review_path)))
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert load_plan(path, "collect")["seed"] == 2027
    plan["seed"] = 2028
    path.write_text(json.dumps(plan))
    with pytest.raises(AssertionError):
        load_plan(path, "collect")


def collection_fixture(tmp_path):
    import importlib.util
    from pathlib import Path

    entry = Path(__file__).resolve().parents[1] / "scripts/train_e0_policy_refresh.py"
    spec = importlib.util.spec_from_file_location("refresh_trainer_test", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sequence = dict(name="synthetic_train", boxes=[[0, 0, 1, 1]] * 4,
                    valid_frames=[0, 1, 2, 3])
    shared = dict(inputs={"executor": {"sha256": "synthetic_fixture"}},
                  sources={}, seed=2027, cost_weight=.05, label_stride=1,
                  round=1, branch="synthetic", smoke=True)
    collection_plan = dict(shared, mode="collect", shard=0, shards=4)
    plan_path = tmp_path / "config.json"
    plan_path.write_text(json.dumps(collection_plan))
    data_path = tmp_path / "synthetic_train.pth"
    torch.save(dict(sequence=sequence["name"], records=[
        dict(frame=f) for f in (1, 2, 3)], tracked_frames=3,
        plan_sha256="synthetic_plan"), data_path)
    receipt = dict(status="COMPLETE", smoke=True, executor_sha256="synthetic_fixture",
                   plan_sha256="synthetic_plan", sequence_names=[sequence["name"]],
                   artifacts={data_path.name: file_sha(data_path)},
                   records=3, tracked_frames=3)
    terminal = tmp_path / "COMPLETE.json"
    terminal.write_text(json.dumps(receipt))
    collection = dict(path=str(terminal), sha256=file_sha(terminal),
                      plan_sha256="synthetic_plan", plan=collection_plan)
    return module.StateDataset, dict(shared, collections=[collection]), sequence, data_path


def test_collection_checks_record_counts_and_artifact_bytes(tmp_path):
    dataset_type, plan, sequence, path = collection_fixture(tmp_path)
    assert len(dataset_type(plan, [sequence])) == 3
    saved = torch.load(path, map_location="cpu")
    saved["records"].pop()
    torch.save(saved, path)
    with pytest.raises(AssertionError):
        dataset_type(plan, [sequence])


def test_collection_rejects_duplicate_shard_and_mixed_behavior_branch(tmp_path):
    dataset_type, plan, sequence, path = collection_fixture(tmp_path)
    plan["collections"] *= 2
    with pytest.raises(AssertionError):
        dataset_type(plan, [sequence])
    plan["collections"] = plan["collections"][:1]
    plan["collections"][0]["plan"]["branch"] = "other_branch"
    with pytest.raises(AssertionError):
        dataset_type(plan, [sequence])
