"""Train matched C2 variants with full-timeline causal TRAIN pairs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt.c2_data import LasHeRPairsC2
from dual_rgbt.c2_executor import C2DualRGBT, executor_parameter_groups
from dual_rgbt.losses import executor_loss
from lib.test.utils.hann import hann2d

VARIANTS = {
    "zero": ("zero", "none"),
    "film": ("learned_film", "none"),
    "shared": ("learned_film", "shared"),
    "routed": ("learned_film", "routed"),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plan(path):
    plan = json.loads(Path(path).read_bytes())
    assert plan["stage"] == "P2_C2_executor_training"
    assert plan["launch_allowed"] is True
    assert type(plan["smoke"]) is bool
    assert plan["variant"] in VARIANTS and plan["rank"] == 8
    assert plan["epochs"] == 20 and plan["samples"] == 20000
    assert plan["batch_size"] == 8 and plan["workers"] == 4
    assert plan["seed"] == 2027 and plan["milestones"] == [10]
    assert plan["backbone_lr"] == 1e-5 and plan["head_lr"] == 1e-4
    assert plan["weight_decay"] == 1e-4 and plan["gradient_clip"] == .1
    assert plan["test_enabled"] is False and plan["dev_enabled"] is False
    assert Path(plan["root"]).resolve() == ROOT
    for name, digest in plan["sources"].items():
        assert sha(ROOT / name) == digest, name
    for item in plan["inputs"].values():
        assert sha(item["path"]) == item["sha256"], item["path"]
    receipt = json.loads(Path(plan["inputs"]["cpu_result"]["path"]).read_bytes())
    assert receipt["status"] == "PASS" and receipt["stage"] == "P2_C2_executor_cpu_check"
    assert receipt["cuda_initialized"] is False and receipt["strict_state_dict_reload"]
    assert len(receipt["cpu_optimizer_updates"]) == 2
    if not plan["smoke"]:
        inputs = plan["inputs"]
        smoke = json.loads(Path(inputs["smoke_receipt"]["path"]).read_bytes())
        native = json.loads(Path(inputs["smoke_native"]["path"]).read_bytes())
        config = json.loads(Path(inputs["smoke_config"]["path"]).read_bytes())
        assert native["state"] == "COMPLETE" and native["exit_code"] == 0
        assert native["task"] == smoke["task"] == config["task"]
        assert native["command"] == config["command"]
        assert smoke["status"] == "PASS" and smoke["updates"] == 10
        assert smoke["route_counts"] == {str(route): 1 for route in range(10)}
        assert smoke["strict_reload"] and smoke["policy_preserved"]
        assert smoke["config_sha256"] == inputs["smoke_config"]["sha256"]
        assert config["smoke"] is True and config["launch_allowed"] is True
        assert config["sources"] == plan["sources"]
        for key in ("variant", "rank", "epochs", "samples", "batch_size", "workers",
                    "seed", "milestones", "backbone_lr", "head_lr", "weight_decay",
                    "gradient_clip", "dev_enabled", "test_enabled"):
            assert config[key] == plan[key], key
        for key in ("factory", "executor", "manifest", "cpu_result"):
            assert config["inputs"][key] == inputs[key], key
        assert set(smoke["artifacts"]) == {"epoch_001.pth"}
        for name, digest in smoke["artifacts"].items():
            assert sha(Path(config["run"]) / name) == digest
    review = json.loads(Path(plan["review"]["path"]).read_bytes())
    assert sha(plan["review"]["path"]) == plan["review"]["sha256"]
    assert review["stage"] == plan["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == plan["sources"]
    binding = {key: value for key, value in plan.items()
               if key not in ("launch_allowed", "review")}
    digest = hashlib.sha256(json.dumps(
        binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest in review["admitted_plan_bindings"]
    return plan


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(plan):
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_training(plan["seed"])
    inputs = plan["inputs"]
    manifest = json.loads(Path(inputs["manifest"]["path"]).read_bytes())
    names = [row["name"] for row in manifest["train"]]
    assert len(names) == len(set(names)) == 882
    assert set(names).isdisjoint(row["name"] for row in manifest.get("dev", []))
    assert set(names).isdisjoint(manifest.get("test_sequences", []))
    run = Path(plan["run"])
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(plan, indent=2) + "\n")
    (run / "precision.json").write_text(json.dumps(dict(
        matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)) + "\n")
    model = C2DualRGBT.from_e0(
        inputs["factory"]["path"], inputs["executor"]["path"],
        *VARIANTS[plan["variant"]], rank=plan["rank"])
    before = {key: value.cpu().clone() for key, value in model.state_dict().items()
              if plan["smoke"] or key.startswith(("preview.", "quality.", "gain."))}
    model.cuda()
    torch.cuda.reset_peak_memory_stats()
    groups = executor_parameter_groups(model, plan["backbone_lr"], plan["head_lr"])
    optimizer = torch.optim.AdamW(groups, weight_decay=plan["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=plan["milestones"], gamma=.1)
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    # Adapter initialization consumes different RNG counts across variants.
    seed_training(plan["seed"])
    step, artifacts, route_counts = 0, {}, {str(route): 0 for route in range(10)}
    parameters = [parameter for group in groups for parameter in group["params"]]
    fields = ["epoch", "step", "modality", "action", "loss", "gradient_norm", "seconds"]
    with (run / "steps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for epoch in range(1, plan["epochs"] + 1):
            started = time.monotonic()
            dataset = LasHeRPairsC2(
                inputs["manifest"]["path"], "train", plan["samples"],
                plan["seed"] + (epoch - 1) * plan["samples"])
            loader = DataLoader(
                dataset, batch_size=plan["batch_size"], shuffle=False,
                num_workers=plan["workers"], pin_memory=True, drop_last=True)
            model.train()
            for module in (model.preview, model.quality, model.gain):
                module.eval()
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
            for batch in loader:
                begin = time.monotonic()
                templates, search, target = [
                    batch[key].cuda(non_blocking=True)
                    for key in ("templates", "search", "target")]
                modality, action = step % 2, (step // 2) % 5
                output = model.forward_action(
                    templates, search, modality, action, window=window)
                loss = executor_loss(output, target)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, plan["gradient_clip"])
                assert torch.isfinite(norm)
                optimizer.step()
                route_counts[str(modality + action * 2)] += 1
                step += 1
                row = dict(epoch=epoch, step=step, modality=modality, action=action,
                           loss=float(loss), gradient_norm=float(norm),
                           seconds=time.monotonic() - begin)
                writer.writerow(row)
                if step % 25 == 0 or plan["smoke"]:
                    handle.flush()
                    print(json.dumps(row), flush=True)
                if plan["smoke"] and step == 10:
                    break
            scheduler.step()
            after = model.state_dict()
            assert all(torch.isfinite(value).all() for value in after.values())
            assert all(torch.equal(value, after[name].cpu())
                       for name, value in before.items()
                       if name.startswith(("preview.", "quality.", "gain.")))
            if plan["smoke"] or epoch > plan["epochs"] - 5:
                path = run / ("epoch_{:03d}.pth".format(epoch))
                torch.save(dict(net=after, optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(), epoch=epoch, step=step,
                                c2_config=model.c2_config, config=plan), path)
                saved = torch.load(path, map_location="cpu")
                assert all(torch.equal(value.cpu(), saved["net"][name])
                           for name, value in after.items())
                restored = C2DualRGBT(inputs["factory"]["path"], **model.c2_config)
                restored.load_state_dict(saved["net"], strict=True)
                del restored
                artifacts[path.name] = sha(path)
            print("EPOCH_COMPLETE", epoch, time.monotonic() - started, flush=True)
            if plan["smoke"]:
                assert set(route_counts.values()) == {1} and step == 10
                changed = {prefix: sum(
                    not torch.equal(value, after[name].cpu())
                    for name, value in before.items() if name.startswith(prefix))
                    for prefix in ("backbone.", "decode.", "box_head.", "missing.")}
                assert all(changed[prefix] > 0
                           for prefix in ("backbone.", "decode.", "box_head."))
                if plan["variant"] != "zero":
                    assert changed["missing."] > 0
                if model.c2_adapters:
                    assert any(torch.count_nonzero(adapter.b) > 0
                               for adapter in model.c2_adapters)
                receipt = dict(status="PASS", task=plan["task"],
                               config_sha256=sha(run / "config.json"),
                               updates=step, route_counts=route_counts,
                               changed_tensors=changed, strict_reload=True, artifacts=artifacts,
                               policy_preserved=True, native_training=True,
                               benchmark_evidence=False, policy_refresh_required=True,
                               gpu_name=torch.cuda.get_device_name(0),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved())
                (run / "SMOKE.json").write_text(json.dumps(receipt, indent=2) + "\n")
                return
    assert step == 50000 and set(route_counts.values()) == {5000}
    (run / "COMPLETE.json").write_text(json.dumps(dict(
        status="COMPLETE", epochs=20, steps=step, route_counts=route_counts,
        artifacts=artifacts, retained_epochs=[16, 17, 18, 19, 20],
        policy_preserved=True, policy_refresh_required=True,
        benchmark_evidence=False), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    train(load_plan(args.plan))


if __name__ == "__main__":
    main()
