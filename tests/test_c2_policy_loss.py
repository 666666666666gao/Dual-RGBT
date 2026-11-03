"""Tiny CPU witnesses for C2 preview flow and frozen executor boundaries."""
import pytest
import torch
from torch import nn

from dual_rgbt.c2_policy_loss import configure_policy_refresh, policy_refresh_loss


class TinyC2Executor(nn.Module):
    """Use C2's keyword-only preview interface without a production backbone."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.decode = nn.BatchNorm1d(1)
        self.box_head = nn.Linear(1, 1)
        self.missing = nn.Linear(1, 1)
        self.adapter = nn.Linear(1, 1)
        self.preview = nn.Linear(1, 1)
        self.quality = nn.Linear(1, 2)
        self.gain = nn.Linear(1, 4)
        self.calls = []
        self.preview_calls = 0

    def preview_state(self, templates, search, history):
        self.preview_calls += 1
        feature = self.preview(search + history)
        maps = feature[:, :, None] * torch.arange(64).reshape(1, 8, 8) / 64
        result = dict(feature=feature, maps=[maps, maps], quality=self.quality(feature))
        self.current_preview = result
        return result

    def primary(self, templates, search, modality, window, *, preview):
        assert not torch.is_grad_enabled()
        assert preview is self.current_preview
        self.calls.append(("primary", modality, window))
        feature = self.backbone(search) + self.missing(preview["feature"])
        centre = .5 + .03 * self.box_head(self.decode(feature)).tanh()
        box = torch.cat((centre, centre, centre * 0 + .3, centre * 0 + .3), 1)
        return dict(box=box)

    def candidates(self, primary, preview, modality):
        assert not torch.is_grad_enabled()
        assert preview is self.current_preview
        indices = (preview["feature"] > 0).long().expand(-1, 4)
        return indices, primary["box"][:, None].expand(-1, 4, -1)

    def correct(self, templates, search, primary, modality, action,
                candidate, box, window, *, preview):
        assert not torch.is_grad_enabled()
        assert preview is self.current_preview
        self.calls.append(("correct", modality, action, window))
        refined = box.clone()
        refined[:, 0] += (.01 * action + .05 * candidate +
                          .01 * self.adapter(preview["feature"]).tanh().flatten())
        return dict(box=refined)

    def gain_predictions(self, primary, preview, modality, indices, boxes):
        assert torch.is_grad_enabled()
        assert preview is self.current_preview
        predicted = self.gain(preview["feature"])
        return torch.cat((predicted.new_zeros(len(predicted), 1), predicted), 1)


def batch():
    return (torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 1),
            torch.tensor([[.3, .3, .3, .3]]).expand(2, -1).clone(), torch.ones(1))


def test_current_preview_and_all_ten_routes_preserve_executor_and_train_policy():
    torch.manual_seed(7)
    model = TinyC2Executor()
    groups = configure_policy_refresh(model)
    assert [group["lr"] for group in groups] == [1e-5, 1e-4]
    assert not model.training and not model.decode.training
    grouped = [id(param) for group in groups for param in group["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(param) for param in model.parameters()
                            if param.requires_grad}
    before = {name: value.clone() for name, value in model.state_dict().items()}
    inputs = batch()
    original = [tensor.clone() for tensor in inputs]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    result = policy_refresh_loss(model, *inputs)
    assert model.preview_calls == 1
    assert result["quality_targets"].shape == (2, 2)
    assert result["gain_targets"].shape == (2, 2, 4)
    assert result["query_boxes"].shape == (2, 2, 4, 4)
    assert not result["quality_targets"].requires_grad
    assert not result["gain_targets"].requires_grad
    assert not result["query_boxes"].requires_grad
    result["loss"].backward()
    for module in (model.preview, model.quality, model.gain):
        assert any(param.grad is not None and param.grad.abs().sum() > 0
                   for param in module.parameters())
    for name, param in model.named_parameters():
        if not name.startswith(("preview.", "quality.", "gain.")):
            assert param.grad is None
    optimizer.step()
    after = model.state_dict()
    for name, value in before.items():
        if not name.startswith(("preview.", "quality.", "gain.")):
            assert torch.equal(value, after[name]), name
    for prefix in ("preview.", "quality.", "gain."):
        assert any(not torch.equal(value, after[name])
                   for name, value in before.items() if name.startswith(prefix))
    assert all(torch.equal(a, b) for a, b in zip(inputs, original))
    assert [call[1] for call in model.calls if call[0] == "primary"] == [0, 1]
    assert [(call[1], call[2]) for call in model.calls if call[0] == "correct"] == [
        (modality, action) for modality in range(2) for action in range(1, 5)]
    assert all(call[-1] is inputs[-1] for call in model.calls)


def test_preview_change_recomputes_C2_missing_conditioned_targets():
    torch.manual_seed(11)
    model = TinyC2Executor()
    configure_policy_refresh(model)
    with torch.no_grad():
        model.preview.weight.zero_()
        model.preview.bias.fill_(-1)
    first = policy_refresh_loss(model, *batch())
    with torch.no_grad():
        model.preview.bias.fill_(1)
    second = policy_refresh_loss(model, *batch())
    assert model.preview_calls == 2
    assert not torch.equal(first["primary_boxes"], second["primary_boxes"])
    assert not torch.equal(first["gain_targets"], second["gain_targets"])
    assert not torch.equal(first["candidate_indices"], second["candidate_indices"])


def test_off_crop_labels_still_train_heads_without_spatial_cross_entropy():
    torch.manual_seed(13)
    model = TinyC2Executor()
    configure_policy_refresh(model)
    inputs = list(batch())
    inputs[3][:, 0] = 2.
    result = policy_refresh_loss(model, *inputs)
    assert result["spatial_loss"] == 0
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.preview.weight.grad.abs().sum() > 0
    assert result["quality_targets"].shape == (2, 2)


def test_requires_the_same_explicit_window_boundary():
    model = TinyC2Executor()
    configure_policy_refresh(model)
    inputs = list(batch())
    inputs[-1] = None
    with pytest.raises(AssertionError):
        policy_refresh_loss(model, *inputs)
