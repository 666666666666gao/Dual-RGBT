"""Check actual C2 executor math and optimizer wiring on one TRAIN pair."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt.c2_executor import C2DualRGBT, executor_parameter_groups
from dual_rgbt.data import LasHeRPairsV2
from dual_rgbt.losses import executor_loss
from dual_rgbt.model import DualRGBT
from lib.test.utils.hann import hann2d


def assert_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_equal(first, second)
    else:
        assert left == right


def check(factory, executor, manifest):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert torch.cuda.is_initialized() is False
    torch.set_num_threads(1)
    torch.manual_seed(2027)
    torch.backends.cudnn.benchmark = False
    item = LasHeRPairsV2(manifest, "train", 1, 2027)[0]
    templates, search, target = [
        item[key].unsqueeze(0) for key in ("templates", "search", "target")]
    assert all(tensor.device.type == "cpu" for tensor in (templates, search, target))
    window = hann2d(torch.tensor([16, 16]), centered=True)
    baseline = DualRGBT(factory)
    baseline.load_state_dict(torch.load(executor, map_location="cpu")["net"], strict=True)
    baseline.eval()
    reference = {}
    with torch.no_grad():
        for modality in range(2):
            for action in range(5):
                reference[modality, action] = baseline.forward_action(
                    templates, search, modality, action, window=window)
        reference_predict = baseline.predict(
            templates, search, torch.zeros(1, 8), .05, window)
    base_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    assert base_parameters == 103374215
    variants = []
    for missing, lora, extra in (
            ("zero", "none", 0),
            ("learned_film", "none", 494592),
            ("learned_film", "shared", 494592 + 1179648),
            ("learned_film", "routed", 494592 + 3538944)):
        started = time.monotonic()
        model = C2DualRGBT.from_e0(factory, executor, missing, lora)
        model.eval()
        assert sum(parameter.numel() for parameter in model.parameters()) == base_parameters + extra
        routes = []
        with torch.no_grad():
            for modality in range(2):
                for action in range(5):
                    actual = model.forward_action(
                        templates, search, modality, action, window=window)
                    assert_equal(reference[modality, action], actual)
                    expected_bank = (0 if lora != "routed" or action == 0
                                     else 2 if action == 4 else 1)
                    assert all(adapter.route == expected_bank for adapter in model.c2_adapters)
                    routes.append(dict(modality=modality, action=action,
                                       selected_bank=expected_bank))
            assert_equal(reference_predict, model.predict(
                templates, search, torch.zeros(1, 8), .05, window))
        groups = executor_parameter_groups(model)
        assert [group["lr"] for group in groups] == [1e-5, 1e-4]
        assert all(not parameter.requires_grad for module in (
            model.preview, model.quality, model.gain) for parameter in module.parameters())
        variants.append(dict(missing=missing, lora=lora, added_parameters=extra,
                             tensor_equal_all_ten_routes=True, predict_equal=True,
                             optimizer_unique_complete=True, routes=routes,
                             cpu_seconds=time.monotonic() - started))
        if lora != "routed":
            del model
    # The routed model is the actual final variant from the loop above.
    frozen = {name: value.clone() for name, value in model.state_dict().items()
              if name.startswith(("preview.", "quality.", "gain."))}
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    model.train()
    # Single-pair connectivity requires every residual branch to participate.
    for block in model.backbone.blocks:
        block.drop_path.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    losses = []
    for step in range(2):
        output = model.forward_action(templates, search, 0, 4, window=window)
        loss = executor_loss(output, target)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for adapter in model.c2_adapters:
            assert adapter.b.grad is not None and torch.isfinite(adapter.b.grad).all()
            assert adapter.b.grad[0].abs().sum() > 0
            assert adapter.b.grad[2].abs().sum() > 0
            assert torch.count_nonzero(adapter.b.grad[1]) == 0
        assert model.missing.embedding.grad.abs().sum() > 0
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in model.missing.film.parameters())
        parameters = [parameter for group in groups for parameter in group["params"]]
        norm = torch.nn.utils.clip_grad_norm_(parameters, .1)
        assert torch.isfinite(norm)
        optimizer.step()
        losses.append(dict(step=step + 1, loss=float(loss), gradient_norm=float(norm)))
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in frozen.items())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        expected = model.forward_action(templates, search, 1, 1, window=window)
    restored = C2DualRGBT(factory, **model.c2_config)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    with torch.no_grad():
        assert_equal(expected, restored.forward_action(
            templates, search, 1, 1, window=window))
    assert torch.cuda.is_initialized() is False
    return dict(
        status="PASS", stage="P2_C2_executor_cpu_check",
        train_sequence=item["sequence"], train_frame=item["frame"],
        base_parameters=base_parameters, variants=variants,
        cpu_optimizer_updates=losses, frozen_policy_equal=True,
        diagnostic_drop_path_enabled=False,
        strict_state_dict_reload=True, cuda_initialized=False,
        gpu_execution=False, native_training=False, benchmark_evidence=False,
        scope_limits=[
            "One TRAIN pair; no convergence or benchmark claims.",
            "Route banks share a Parameter; AdamW decay/momentum may change inactive slices.",
            "Routing is set before each encode; concurrent/reentrant calls are unsupported.",
            "Merged inference and P2 policy-collection callers are not implemented.",
            "State-dict reload is tested; no serialized C2 checkpoint is written."])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True)
    parser.add_argument("--executor", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    report = check(args.factory, args.executor, args.manifest)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
