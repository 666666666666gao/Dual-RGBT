"""Cross-check CPU-executed E0 route MACs without touching GPU processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "source-review", "source-review-sha256", "checkpoint",
                 "checkpoint-sha256", "factory", "factory-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    root = Path(args.root)
    output = root / args.output
    assert not output.exists()
    review_path = root / args.source_review
    assert sha(review_path) == args.source_review_sha256
    review = json.loads(review_path.read_bytes())
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert all(sha(root / name) == digest
               for name, digest in review["source_sha256"].items())
    assert sha(args.checkpoint) == args.checkpoint_sha256
    assert sha(args.factory) == args.factory_sha256
    sys.path.insert(0, str(root))
    import torch
    from torch import nn
    from torch.profiler import ProfilerActivity, profile
    from dual_rgbt.model import ACTION_NAMES, DualRGBT
    from dual_rgbt.e0_identity import E0CandidateIdentity, initial_identity_anchor
    from dual_rgbt.e0_reliability import E0Reliability
    from lib.models.cadtrack.vit_care import Attention

    torch.set_num_threads(1)
    torch.manual_seed(2027)
    model = DualRGBT(args.factory).eval().requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["net"], strict=True)
    del checkpoint
    before = {name: value.clone() for name, value in model.state_dict().items()}
    m1 = E0Reliability().eval().requires_grad_(False)
    m2 = E0CandidateIdentity().eval().requires_grad_(False)
    templates = torch.randn(1, 2, 6, 128, 128)
    search = torch.randn(1, 6, 256, 256)
    countable = {"aten::conv2d", "aten::mm", "aten::bmm", "aten::addmm"}

    def measure(module, call, extra_macs=0):
        counts = dict(conv2d=0, linear=0, attention_products=0,
                      functional_patch_projection=extra_macs)
        handles = []

        def hook(layer, inputs, result):
            if isinstance(layer, nn.Conv2d):
                counts["conv2d"] += result.numel() * layer.weight[0].numel()
            elif isinstance(layer, nn.Linear):
                counts["linear"] += result.numel() * layer.in_features
            else:
                batch, tokens, width = inputs[0].shape
                counts["attention_products"] += 2 * batch * tokens * tokens * width

        for layer in module.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, Attention)):
                handles.append(layer.register_forward_hook(hook))
        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU],
                                      with_flops=True, record_shapes=True) as prof:
            result = call()
        for handle in handles:
            handle.remove()
        events = prof.key_averages(group_by_input_shape=True)
        operators = [dict(operator=event.key, shapes=event.input_shapes,
                          calls=event.count, flops=int(event.flops))
                     for event in events if event.key in countable]
        assert all(row["flops"] > 0 for row in operators)
        flops = sum(row["flops"] for row in operators)
        assert flops % 2 == 0 and flops // 2 == sum(counts.values()), (
            flops // 2, counts)
        return result, dict(macs=flops // 2, gmacs=flops / 2e9,
                            formula_breakdown=counts, operators=operators,
                            independent_formula_match=True)

    rows = []
    for modality in range(2):
        for action, name in enumerate(ACTION_NAMES):
            patches = 0 if action == 0 else (256 if action == 4 else 64)
            # Selected auxiliary patches use functional linear, bypassing its module hook.
            extra = patches * model.backbone.patch_embed.proj.weight.numel()
            result, cost = measure(model, lambda: model.forward_action(
                templates, search, modality, action), extra)
            assert torch.isfinite(result["box"]).all()
            assert result["primary"]["primary_tokens"] == 385
            assert result["auxiliary_tokens"] == (129 + patches if action else 0)
            rows.append(dict(modality=modality, action=action, name=name,
                             primary_tokens=385,
                             auxiliary_tokens=result["auxiliary_tokens"], **cost))
            print(json.dumps({key: value for key, value in rows[-1].items()
                              if key not in ("operators", "formula_breakdown")}),
                  flush=True)
            del result
    box = torch.tensor([[.5, .5, .2, .2]])
    _, anchor_cost = measure(model, lambda: initial_identity_anchor(
        model, templates, search, box))
    _, m1_cost = measure(m1, lambda: m1(torch.zeros(1, 914)))
    identity_rows = []
    for candidates in (1, 24):
        _, cost = measure(m2, lambda: m2(torch.randn(1, 2, 768),
            torch.randn(1, candidates, 2, 768), torch.ones(1, candidates, 2)))
        assert cost["macs"] == 196608 + 229504 * candidates
        identity_rows.append(dict(candidates=candidates, **cost))
    assert all(torch.equal(value, before[name])
               for name, value in model.state_dict().items())
    assert all(value.grad is None for value in model.parameters())
    assert not torch.cuda.is_initialized()
    report = dict(status="CPU_ROUTE_MAC_CROSSCHECK_COMPLETE",
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        torch_version=torch.__version__, device="cpu", seed=2027,
        inputs=dict(batch=1, templates=[1, 2, 6, 128, 128],
                    search=[1, 6, 256, 256], source="seeded synthetic tensors"),
        checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256,
        factory=args.factory, factory_sha256=args.factory_sha256,
        source_review=args.source_review,
        source_review_sha256=args.source_review_sha256,
        source_sha256=review["source_sha256"],
        profiler_script_sha256=sha(__file__),
        parameter_counts=dict(e0=sum(p.numel() for p in model.parameters()),
            m1=sum(p.numel() for p in m1.parameters()),
            m2=sum(p.numel() for p in m2.parameters())),
        routes=rows, m2_initial_anchor=anchor_cost, m1_head=m1_cost,
        m2_identity_head=identity_rows, model_unchanged=True,
        cuda_initialized=False, test_eligible=False,
        scientific_result_audit_accepted=False,
        scope="Executed E0 forward_action convolution/dense/attention MACs, one MAC "
              "equals two profiler FLOPs; independent module formulas plus explicit "
              "functional patch projection match each route. Excludes bias adds, "
              "normalization, nonlinearities, interpolation, ROI sampling, memory "
              "traffic, decoding and tracker bookkeeping. Route MACs are fixed-shape "
              "arithmetic counts, not dataset-weighted costs or measured FPS. "
              "Head-only M1/M2 costs exclude observer/collection/recovery. M2 initial "
              "two-modality anchor encoding counted separately, not per-frame. "
              "Smoke checkpoint proves architecture execution, not trained efficacy "
              "or completed integration. No process signalling or GPU allocation.")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), sha256=sha(output),
                          parameter_counts=report["parameter_counts"],
                          m1_head_macs=m1_cost["macs"],
                          m2_anchor_gmacs=anchor_cost["gmacs"])), flush=True)


if __name__ == "__main__":
    main()
