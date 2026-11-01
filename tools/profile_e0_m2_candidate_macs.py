"""Measure M2 candidate primary forwards on CPU against an admitted R15 plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "plan", "plan-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    root = Path(args.root).resolve()
    plan_path = root / args.plan
    output = root / args.output
    assert not output.exists()
    assert sha(plan_path) == args.plan_sha256
    plan = json.loads(plan_path.read_bytes())
    assert plan["stage"] == "E0_M2_online_integration"
    assert plan["target_root"] == root.as_posix()
    assert plan["launch_allowed"] is True
    assert plan["exploratory_test"] is True and plan["test_eligible"] is False
    review_path = root / plan["source_review"]
    assert sha(review_path) == plan["source_review_sha256"]
    review = json.loads(review_path.read_bytes())
    assert review["stage"] == plan["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["draft_map_verification"]["source_sha256"] == plan["source_sha256"]
    assert review["draft_map_verification"]["input_sha256"] == plan["input_sha256"]
    assert all(sha(root / name) == value for name, value in plan["source_sha256"].items())
    assert all(sha(root / name) == value for name, value in plan["input_sha256"].items())
    parent = root / plan["parent_checkpoint"]
    factory = root / plan["factory_checkpoint"]
    assert sha(parent) == plan["parent_checkpoint_sha256"]
    assert plan["input_sha256"][plan["factory_checkpoint"]] == sha(factory)

    sys.path.insert(0, str(root))
    import torch
    from torch import nn
    from torch.profiler import ProfilerActivity, profile
    from dual_rgbt.model import DualRGBT
    from lib.models.cadtrack.vit_care import Attention

    torch.set_num_threads(1)
    torch.manual_seed(2027)
    model = DualRGBT(factory).eval().requires_grad_(False)
    model.load_state_dict(torch.load(parent, map_location="cpu")["net"], strict=True)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    templates = torch.randn(1, 2, 6, 128, 128)
    search = torch.randn(1, 6, 256, 256)
    rows = []
    for modality in (0, 1):
        counts = dict(conv2d=0, linear=0, attention_products=0)

        def hook(layer, inputs, result):
            if isinstance(layer, nn.Conv2d):
                counts["conv2d"] += result.numel() * layer.weight[0].numel()
            elif isinstance(layer, nn.Linear):
                counts["linear"] += result.numel() * layer.in_features
            else:
                batch, tokens, width = inputs[0].shape
                counts["attention_products"] += 2 * batch * tokens * tokens * width

        handles = [layer.register_forward_hook(hook) for layer in model.modules()
                   if isinstance(layer, (nn.Conv2d, nn.Linear, Attention))]
        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU],
                                      with_flops=True, record_shapes=True) as measured:
            result = model.primary(templates, search, modality)
        for handle in handles:
            handle.remove()
        assert torch.isfinite(result["box"]).all()
        assert result["primary_tokens"] == 385
        operators = [dict(operator=event.key, shapes=event.input_shapes,
                          calls=event.count, flops=int(event.flops))
                     for event in measured.key_averages(group_by_input_shape=True)
                     if event.key in {"aten::conv2d", "aten::mm", "aten::bmm", "aten::addmm"}]
        assert operators and all(row["flops"] > 0 for row in operators)
        flops = sum(row["flops"] for row in operators)
        assert flops % 2 == 0
        assert flops // 2 == sum(counts.values())
        rows.append(dict(modality=modality, macs=flops // 2,
                         independent_formula_match=True,
                         formula_breakdown=counts, operators=operators))
        print(json.dumps(dict(modality=modality, macs=flops // 2)), flush=True)

    assert all(torch.equal(before[name], value)
               for name, value in model.state_dict().items())
    assert all(value.grad is None for value in model.parameters())
    assert not torch.cuda.is_initialized()
    assert all(sha(root / name) == value for name, value in plan["source_sha256"].items())
    assert all(sha(root / name) == value for name, value in plan["input_sha256"].items())
    report = dict(status="CPU_M2_PRIMARY_MAC_CROSSCHECK_COMPLETE",
                  plan=str(plan_path), plan_sha256=args.plan_sha256,
                  source_sha256=plan["source_sha256"],
                  input_sha256=plan["input_sha256"],
                  source_review_sha256=plan["source_review_sha256"],
                  parent_checkpoint_sha256=plan["parent_checkpoint_sha256"],
                  script_sha256=sha(__file__), device="cpu",
                  torch_version=torch.__version__, seed=2027,
                  inputs=dict(templates=[1, 2, 6, 128, 128], search=[1, 6, 256, 256]),
                  primary=rows, queries_per_invocation=12,
                  primary_forwards_per_invocation=24,
                  candidate_primary_macs_per_invocation=12 * sum(row["macs"] for row in rows),
                  model_unchanged=True, cuda_initialized=False,
                  exploratory_test=True, test_eligible=False,
                  scope="CPU-executed convolution/dense/attention matrix MACs; "
                        "one MAC equals two profiler FLOPs. Synthetic input shapes "
                        "match candidate crops. Candidate generation executes "
                        "12 queries with two primary modalities each. Excludes "
                        "identity head, initial anchor, base E0 tracking, ROI/crops, "
                        "nonlinearities, normalization, decoding and memory traffic. "
                        "Not full-run MACs, FPS or benchmark accuracy.")
    with output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), sha256=sha(output))), flush=True)


if __name__ == "__main__":
    main()
