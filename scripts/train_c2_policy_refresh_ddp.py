"""Four-GPU DDP FIT for the C2 policy refresh.

This entry is intentionally separate from the historical single-GPU trainer.
The plan keeps the original TRAIN inputs and recipe; only execution is
distributed across four local ranks.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dual_rgbt.c2_policy_loss import configure_policy_refresh, policy_refresh_loss
from dual_rgbt.c2_policy_refresh import file_sha, load_completed_executor
from lib.test.utils.hann import hann2d
from train_c2_policy_refresh import StateDataset, validate_collection_acceptance

LOSS_FIELDS = ("loss", "quality_loss", "gain_loss", "spatial_loss")
POLICY_PREFIXES = ("preview.", "quality.", "gain.")


class LossModule(torch.nn.Module):
    def __init__(self, model, window):
        super().__init__()
        self.model = model
        self.window = window

    def forward(self, batch):
        result = policy_refresh_loss(self.model, *batch, self.window)
        return result["loss"], torch.stack(
            [result[field].detach() for field in LOSS_FIELDS])


def load_ddp_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    assert plan["stage"] == "P2_C2_policy_refresh_fit"
    assert plan["mode"] == "fit" and plan["smoke"] is False
    assert plan["epochs"] == 5 and plan["batch_size"] == 512
    assert plan["workers"] == 4 and plan["seed"] == 2027
    assert plan["test_enabled"] is plan["dev_enabled"] is False
    assert plan["variant"] in ("zero", "film")
    assert Path.cwd().resolve() == Path(plan["root"]).resolve()
    assert plan["sources"]["scripts/train_c2_policy_refresh_ddp.py"] == file_sha(
        ROOT / "scripts/train_c2_policy_refresh_ddp.py")
    return plan


def reduce_metrics(values):
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values / dist.get_world_size()


def run_smoke(plan):
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    acceptance = validate_collection_acceptance(plan)
    dataset = StateDataset(plan, acceptance)
    sampler = DistributedSampler(
        dataset, num_replicas=4, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset, batch_size=128, sampler=sampler, num_workers=0,
        pin_memory=True)
    model = load_completed_executor(plan["executor_binding"], device=device)
    window = hann2d(torch.tensor([16, 16]), centered=True).to(device)
    configure_policy_refresh(model)
    ddp = DistributedDataParallel(
        LossModule(model, window), device_ids=[local_rank],
        output_device=local_rank, broadcast_buffers=False)
    batch = next(iter(loader))
    batch = [tensor.to(device, non_blocking=True) for tensor in batch]
    loss, metrics = ddp(batch)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    dist.barrier()
    if rank == 0:
        smoke = plan["run"] + "_SMOKE.json"
        Path(smoke).write_text(json.dumps({
            "status": "COMPLETE", "world_size": 4, "local_batch_size": 128,
            "global_batch_size": 512, "steps": 1,
            "losses": metrics.detach().cpu().tolist(),
            "gpus": list(range(4)),
        }, indent=2) + "\n")
    dist.barrier()


def train(plan, plan_path):
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    acceptance = validate_collection_acceptance(plan)
    dataset = StateDataset(plan, acceptance)
    sampler = DistributedSampler(
        dataset, num_replicas=4, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset, batch_size=128, sampler=sampler, num_workers=plan["workers"],
        pin_memory=True)
    assert len(loader) == 178
    model = load_completed_executor(plan["executor_binding"], device=device)
    before = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    groups = configure_policy_refresh(model)
    groups[0]["lr"] = plan["preview_lr"]
    groups[1]["lr"] = plan["quality_gain_lr"]
    parameters = [p for group in groups for p in group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=plan["weight_decay"])
    window = hann2d(torch.tensor([16, 16]), centered=True).to(device)
    ddp = DistributedDataParallel(
        LossModule(model, window), device_ids=[local_rank],
        output_device=local_rank, broadcast_buffers=False)
    run = Path(plan["run"])
    if rank == 0:
        run.mkdir(parents=True, exist_ok=False)
        (run / "config.json").write_text(json.dumps(plan, indent=2) + "\n")
        (run / "precision.json").write_text(json.dumps({
            "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "cudnn_benchmark": False, "world_size": 4,
            "local_batch_size": 128,
        }, indent=2) + "\n")
        handle = (run / "steps.csv").open("w", newline="")
        writer = csv.DictWriter(handle, ["epoch", "step", *LOSS_FIELDS])
        writer.writeheader()
    else:
        handle = writer = None
    dist.barrier()
    step = 0
    artifacts = {}
    for epoch in range(1, plan["epochs"] + 1):
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = [tensor.to(device, non_blocking=True) for tensor in batch]
            loss, metrics = ddp(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, plan["gradient_clip"])
            assert torch.isfinite(norm)
            optimizer.step()
            step += 1
            reduced = reduce_metrics(metrics)
            if rank == 0:
                writer.writerow(dict(
                    epoch=epoch, step=step,
                    **{key: float(value) for key, value in zip(
                        LOSS_FIELDS, reduced.detach().cpu())}))
                if step % 25 == 0:
                    handle.flush()
                    print(epoch, step, float(reduced[0]), flush=True)
        assert all(torch.isfinite(value).all()
                   for value in model.state_dict().values())
        for key, value in before.items():
            if not key.startswith(POLICY_PREFIXES):
                assert torch.equal(value, model.state_dict()[key].detach().cpu()), key
        if rank == 0:
            path = run / ("epoch_{:03d}.pth".format(epoch))
            state = model.state_dict()
            torch.save(dict(
                net=state, optimizer=optimizer.state_dict(), epoch=epoch,
                step=step, config=plan,
                executor_sha256=plan["executor_binding"]["inputs"][
                    "executor"]["sha256"]), path)
            saved = torch.load(path, map_location="cpu")
            model.load_state_dict(saved["net"], strict=True)
            artifacts[path.name] = file_sha(path)
        dist.barrier()
    if rank == 0:
        handle.close()
        result = dict(
            status="COMPLETE", stage=plan["stage"], smoke=False,
            epochs=plan["epochs"], steps=step, records=len(dataset),
            retained_epochs=list(range(1, plan["epochs"] + 1)),
            artifacts=artifacts, executor_preserved=True,
            all_policy_modules_changed=True, strict_reload=True,
            executor_binding=plan["executor_binding"],
            executor_sha256=plan["executor_binding"]["inputs"]["executor"]["sha256"],
            benchmark_evidence=False, test_enabled=False, dev_enabled=False,
            world_size=4, local_batch_size=128, global_batch_size=512,
            plan_sha256=file_sha(plan_path),
        )
        (run / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = load_ddp_plan(args.plan)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 4
    dist.init_process_group("nccl")
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"] + rank)
    np.random.seed(plan["seed"] + rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.smoke:
        run_smoke(plan)
    else:
        train(plan, args.plan)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
