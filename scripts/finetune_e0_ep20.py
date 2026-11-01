"""Train E0 executors from CADTrack ep20 while retaining the parent policy."""
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
from dual_rgbt.data import LasHeRPairs
from dual_rgbt.losses import executor_loss
from dual_rgbt.model import DualRGBT


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def initialize(cadtrack, parent):
    model = DualRGBT(cadtrack)
    parent_state = torch.load(parent, map_location="cpu")["net"]
    model.load_state_dict(parent_state, strict=True)
    source = torch.load(cadtrack, map_location="cpu")
    assert source["net_type"] == "CADTrack"
    state = model.state_dict()
    targets = set()
    for name, tensor in source["net"].items():
        assert name.startswith(("backbone.", "box_head.", "decode_fuse_search."))
        target = name.replace("decode_fuse_search.", "decode.", 1)
        assert target not in targets
        assert state[target].shape == tensor.shape and state[target].dtype == tensor.dtype
        state[target] = tensor
        targets.add(target)
    preserved = set(state) - targets
    assert all(name.startswith(("preview.", "quality.", "gain.")) or
               name == "token_cost" for name in preserved)
    model.load_state_dict(state, strict=True)
    frozen = {name: value.clone() for name, value in state.items() if name in preserved}
    for module in (model.preview, model.quality, model.gain):
        module.requires_grad_(False)
    return model, frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cadtrack", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert args.epochs > 0 and args.samples >= args.batch_size > 0
    assert args.samples % args.batch_size == 0 and args.lr > 0
    assert args.samples // args.batch_size >= 10
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    manifest = json.loads(Path(args.manifest).read_text())
    names = [row["name"] for row in manifest["train"]]
    assert len(names) == len(set(names)) == 882
    assert set(names).isdisjoint(row["name"] for row in manifest.get("dev", []))
    assert set(names).isdisjoint(manifest.get("test_sequences", []))
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), train_sequences=names, hashes={
        "cadtrack": sha(args.cadtrack), "parent": sha(args.parent),
        "manifest": sha(args.manifest), "entry": sha(__file__),
    }, milestones=[10, 30], gamma=0.1, policy_frozen=True, dev_enabled=False,
        test_enabled=False, stage="E0_executor_ep20_finetune")
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model, frozen = initialize(args.cadtrack, args.parent)
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()} if args.smoke else None
    model.cuda()
    groups = [
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": list(model.decode.parameters()) + list(model.box_head.parameters()), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.1)
    step = 0
    with (run / "steps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["epoch", "step", "modality", "action", "loss", "seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            dataset = LasHeRPairs(args.manifest, "train", args.samples,
                                 args.seed + (epoch - 1) * args.samples)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True, drop_last=True)
            model.train()
            for module in (model.preview, model.quality, model.gain):
                module.eval()
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
            for batch in loader:
                begin = time.time()
                templates, search, target = [batch[key].cuda(non_blocking=True)
                                              for key in ("templates", "search", "target")]
                modality, action = step % 2, (step // 2) % 5
                output = model.forward_action(templates, search, modality, action)
                loss = executor_loss(output, target)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [value for value in model.parameters() if value.requires_grad], 0.1)
                assert torch.isfinite(norm)
                optimizer.step()
                row = dict(epoch=epoch, step=step, modality=modality, action=action,
                           loss=float(loss), seconds=time.time() - begin)
                writer.writerow(row)
                if step % 50 == 0 or args.smoke:
                    handle.flush()
                    print(json.dumps(row), flush=True)
                step += 1
                if args.smoke and step == 10:
                    break
            scheduler.step()
            assert all(torch.equal(model.state_dict()[key].cpu(), value)
                       for key, value in frozen.items())
            state = dict(net=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), epoch=epoch, step=step, config=config)
            if args.smoke or epoch > args.epochs - 5:
                checkpoint = run / ("epoch_{:03d}.pth".format(epoch))
                torch.save(state, checkpoint)
            print("EPOCH_COMPLETE", epoch, time.time() - started, flush=True)
            if args.smoke:
                clone, _ = initialize(args.cadtrack, args.parent)
                clone.load_state_dict(torch.load(checkpoint, map_location="cpu")["net"], strict=True)
                changed = {prefix: sum(not torch.equal(value, model.state_dict()[key].cpu())
                                       for key, value in before.items() if key.startswith(prefix))
                           for prefix in ("backbone.", "decode.", "box_head.")}
                assert all(value > 0 for value in changed.values())
                (run / "SMOKE.json").write_text(json.dumps(dict(
                    status="PASS", updates=step, routes=10, changed_tensors=changed,
                    strict_reload=True, policy_preserved=True, benchmark_evidence=False), indent=2))
                return
    (run / "COMPLETE.json").write_text(json.dumps(dict(
        status="COMPLETE", epochs=args.epochs, steps=step,
        retained_epochs=list(range(max(1, args.epochs - 4), args.epochs + 1)),
        policy_preserved=True, benchmark_evidence=False), indent=2))


if __name__ == "__main__":
    main()
