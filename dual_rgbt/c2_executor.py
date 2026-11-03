"""Versioned C2 executor; original E0 and its running snapshots stay separate."""
import torch
from torch import nn

from dual_rgbt.missing_modality import MissingModality
from dual_rgbt.model import DualRGBT
from dual_rgbt.route_lora import RouteLoRA


class ExecutorRouteLoRA(RouteLoRA):
    """Bind a route before the original attention/MLP invokes its linears."""

    def __init__(self, linear, rank, routes):
        super().__init__(linear, rank=rank, routes=routes)
        self.route = 0

    def forward(self, x):
        return super().forward(x, self.route)


class C2DualRGBT(DualRGBT):
    def __init__(self, factory, missing="zero", lora="none", rank=8):
        super().__init__(factory)
        self.c2_adapters = ()
        self.missing = MissingModality("zero")
        self.install_c2(missing, lora, rank)

    def install_c2(self, missing, lora, rank):
        assert not self.c2_adapters and self.missing.mode == "zero"
        assert lora in ("none", "shared", "routed")
        self.missing = MissingModality(missing)
        self.c2_config = dict(missing=missing, lora=lora, rank=rank)
        adapters = []
        if lora != "none":
            routes = 1 if lora == "shared" else 3
            for block in self.backbone.blocks:
                for owner, name in ((block.attn, "qkv"), (block.attn, "proj"),
                                    (block.mlp, "fc1"), (block.mlp, "fc2")):
                    adapter = ExecutorRouteLoRA(getattr(owner, name), rank, routes)
                    setattr(owner, name, adapter)
                    adapters.append(adapter)
        # Already registered under backbone; do not register a second state path.
        self.c2_adapters = tuple(adapters)

    @classmethod
    def from_e0(cls, factory, executor, missing, lora, rank=8):
        model = cls(factory)
        model.load_state_dict(torch.load(executor, map_location="cpu")["net"], strict=True)
        model.install_c2(missing, lora, rank)
        return model

    def encode(self, templates, image, indices=None, route=0):
        assert route in (0, 1, 2)
        bank = route if self.c2_config["lora"] == "routed" else 0
        for adapter in self.c2_adapters:
            adapter.route = bank
        return super().encode(templates, image, indices)

    def primary(self, templates, search, modality, window=None, *, preview):
        features, tokens = self.encode(
            templates[:, :, modality * 3:(modality + 1) * 3],
            search[:, modality * 3:(modality + 1) * 3], route=0)
        missing = self.missing(1 - modality, preview["descriptors"][1 - modality])
        pair = [features, missing] if modality == 0 else [missing, features]
        dense = torch.cat(pair, 2).transpose(1, 2).reshape(search.shape[0], 1536, 16, 16)
        score, box, size, offset = self.box_head(self.decode(dense))
        if window is not None:
            score = score * window
            box = self.box_head.cal_bbox(score, size, offset)
        return dict(features=features, score=score, box=box, size=size,
                    offset=offset, primary_tokens=tokens)

    def correct(self, templates, search, primary, modality, action, candidate,
                box, window=None, *, preview):
        indices = self.region_indices(candidate, action)
        auxiliary = 1 - modality
        evidence, tokens = self.encode(
            templates[:, :, auxiliary * 3:(auxiliary + 1) * 3],
            search[:, auxiliary * 3:(auxiliary + 1) * 3], indices,
            route=2 if action == 4 else 1)
        sparse = self.missing(auxiliary, preview["descriptors"][auxiliary]).scatter(
            1, indices[:, :, None].expand(-1, -1, 768), evidence)
        pair = [primary["features"], sparse] if modality == 0 else [sparse, primary["features"]]
        dense = torch.cat(pair, 2).transpose(1, 2).reshape(search.shape[0], 1536, 16, 16)
        score, refined, size, offset = self.box_head(self.decode(dense))
        if window is not None:
            score = score * window
            refined = self.box_head.cal_bbox(score, size, offset)
        return dict(box=refined, score=score, size=size, offset=offset,
                    auxiliary_tokens=tokens, region_indices=indices)

    def forward_action(self, templates, search, modality, action, history=None, window=None):
        if history is None:
            history = search.new_zeros(search.shape[0], 8)
        preview = self.preview_state(templates, search, history)
        main = self.primary(templates, search, modality, window, preview=preview)
        indices, boxes = self.candidates(main, preview, modality)
        result = dict(box=main["box"], primary=main, preview=preview, action=action,
                      modality=modality, auxiliary_tokens=0)
        if action:
            result.update(self.correct(
                templates, search, main, modality, action,
                indices[:, action - 1], boxes[:, action - 1], window, preview=preview))
        result["gains"] = self.gain_predictions(main, preview, modality, indices, boxes)
        return result

    @torch.no_grad()
    def predict(self, templates, search, history, cost_weight, window=None):
        assert search.shape[0] == 1
        preview = self.preview_state(templates, search, history)
        modality = int(preview["quality"].argmax(1).item())
        main = self.primary(templates, search, modality, window, preview=preview)
        indices, boxes = self.candidates(main, preview, modality)
        gains = self.gain_predictions(main, preview, modality, indices, boxes)
        action = int((gains - cost_weight * self.token_cost).argmax(1).item())
        result = dict(box=main["box"], primary=main, preview=preview, gains=gains,
                      modality=modality, action=action, auxiliary_tokens=0)
        if action:
            result.update(self.correct(
                templates, search, main, modality, action,
                indices[:, action - 1], boxes[:, action - 1], window, preview=preview))
        return result


def executor_parameter_groups(model, backbone_lr=1e-5, head_lr=1e-4):
    for module in (model.preview, model.quality, model.gain):
        module.requires_grad_(False)
    adapter_parameters = [
        parameter for adapter in model.c2_adapters for parameter in (adapter.a, adapter.b)]
    adapter_ids = {id(parameter) for parameter in adapter_parameters}
    backbone = [parameter for parameter in model.backbone.parameters()
                if parameter.requires_grad and id(parameter) not in adapter_ids]
    heads = [parameter for module in (model.decode, model.box_head, model.missing)
             for parameter in module.parameters() if parameter.requires_grad]
    heads += adapter_parameters
    groups = [dict(params=backbone, lr=backbone_lr), dict(params=heads, lr=head_lr)]
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert len(grouped) == len(set(grouped)) and set(grouped) == expected
    return groups
