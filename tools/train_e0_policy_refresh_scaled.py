"""Explicit batch-scaled FIT entrypoint alongside unchanged collector sources."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


LOSS_FIELDS = ("loss", "quality_loss", "gain_loss", "spatial_loss")


def fit(model, loader, plan, run, window, core):
    assert plan["smoke"] is False and plan["epochs"] == 5
    assert plan["lr_scale"] == plan["batch_size"] / 128
    before = {key: tensor.cpu().clone() for key, tensor in model.state_dict().items()}
    groups = core.configure_policy_refresh(model)
    for group in groups:
        group["lr"] *= plan["lr_scale"]
    assert [group["lr"] for group in groups] == [
        plan["preview_lr"], plan["quality_gain_lr"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    parameters = [parameter for group in groups for parameter in group["params"]]
    device = next(model.parameters()).device
    step, artifacts = 0, {}
    with (run / "steps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["epoch", "step", *LOSS_FIELDS])
        writer.writeheader()
        for epoch in range(1, 6):
            for batch in loader:
                batch = [tensor.to(device, non_blocking=True) for tensor in batch]
                result = core.policy_refresh_loss(model, *batch, window)
                optimizer.zero_grad(set_to_none=True)
                result["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                assert torch.isfinite(norm)
                optimizer.step()
                step += 1
                writer.writerow(dict(epoch=epoch, step=step, **{
                    key: float(result[key]) for key in LOSS_FIELDS}))
                if step % 25 == 0:
                    handle.flush()
                    print(epoch, step, float(result["loss"]), flush=True)
            after = model.state_dict()
            assert all(torch.isfinite(tensor).all() for tensor in after.values())
            for key, tensor in before.items():
                if not key.startswith(core.POLICY_PREFIXES):
                    assert torch.equal(tensor, after[key].cpu()), key
            for prefix in core.POLICY_PREFIXES:
                assert any(not torch.equal(tensor, after[key].cpu())
                           for key, tensor in before.items() if key.startswith(prefix)), prefix
            path = run / ("epoch_{:03d}.pth".format(epoch))
            torch.save(dict(net=after, optimizer=optimizer.state_dict(), epoch=epoch,
                            step=step, config=plan,
                            executor_sha256=plan["inputs"]["executor"]["sha256"]), path)
            saved = torch.load(path, map_location="cpu")
            assert all(torch.equal(tensor.cpu(), saved["net"][key])
                       for key, tensor in after.items())
            expected = model.predict(batch[0][:1], batch[1][:1], batch[2][:1],
                                     plan["cost_weight"], window)
            model.load_state_dict(saved["net"], strict=True)
            actual = model.predict(batch[0][:1], batch[1][:1], batch[2][:1],
                                   plan["cost_weight"], window)
            for key in ("modality", "action"):
                assert expected[key] == actual[key]
            for key in ("box", "gains"):
                assert torch.equal(expected[key], actual[key])
            artifacts[path.name] = core.file_sha(path)
    assert step == len(loader) * 5
    return dict(
        status="COMPLETE", stage="P1B_direct_E0_fit", smoke=False,
        epochs=5, steps=step, records=len(loader.dataset), artifacts=artifacts,
        executor_preserved=True, all_policy_modules_changed=True, strict_reload=True,
        closed_loop_round=plan["round"], branch=plan["branch"], sources=plan["sources"],
        executor_sha256=plan["inputs"]["executor"]["sha256"],
        supplemental_sources=plan["supplemental_sources"],
        lr_scale=plan["lr_scale"], preview_lr=plan["preview_lr"],
        quality_gain_lr=plan["quality_gain_lr"], benchmark_evidence=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    draft = json.loads(Path(args.plan).read_text())
    assert Path.cwd().resolve() == Path(draft["root"]).resolve()
    sys.path.insert(0, draft["root"])
    from dual_rgbt import e0_policy_refresh as core
    entry = Path(draft["root"]) / "scripts/train_e0_policy_refresh.py"
    spec = importlib.util.spec_from_file_location("scaled_fit_dataset", entry)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    StateDataset = trainer.StateDataset
    from lib.test.utils.hann import hann2d

    plan = core.load_plan(args.plan, "fit")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["gpu"]
    own_key = "tools/train_e0_policy_refresh_scaled.py"
    assert plan["supplemental_sources"] == {own_key: core.file_sha(__file__)}
    review = json.loads(Path(plan["review"]["path"]).read_text())
    assert review["supplemental_sources"] == plan["supplemental_sources"]
    accepted_input = plan["collection_acceptance"]
    assert core.file_sha(accepted_input["path"]) == accepted_input["sha256"]
    accepted = json.loads(Path(accepted_input["path"]).read_text())
    assert accepted["status"] == "PASS"
    assert accepted["stage"] == "P1B_direct_E0_full_collection_acceptance"
    assert accepted["sequences"] == 882 and accepted["cuda_initialized"] is False
    for key in ("host", "root", "branch", "round", "inputs", "sources",
                "label_stride", "collections"):
        assert accepted[key] == plan[key], key
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = StateDataset(plan, core.train_sequences(plan["inputs"]["manifest"]["path"]))
    assert len(dataset) == accepted["records"]
    loader = DataLoader(dataset, batch_size=plan["batch_size"], shuffle=False,
                        num_workers=plan["workers"], pin_memory=True)
    run = Path(plan["run"])
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(plan, indent=2) + "\n")
    (run / "precision.json").write_text(json.dumps(dict(
        matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)) + "\n")
    model = core.load_completed_executor(plan)
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    result = fit(model, loader, plan, run, window, core)
    result["plan_sha256"] = core.file_sha(args.plan)
    (run / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
