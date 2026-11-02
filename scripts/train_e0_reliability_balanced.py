"""Retrain the existing M1 architecture with fit-only inverse-class weights."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from benchmark_inputs import sha
from dual_rgbt.e0_reliability import E0Reliability, localization_targets
from train_e0_reliability import load_collection, fit_normalization, metrics

STAGE = "M1_fit_class_balanced_training"
HISTORICAL = "scripts/run_task_four_gpu.py"
SNAPSHOT = "reports/source_snapshots/m1_balanced_train_r1_20261101/" + HISTORICAL
OLD_WRAPPER = "cc2b87b38d3b5c48a9f875e78218e48abb3640891beb0b5570c68b81a25afde3"


def class_weights(iou):
    counts = torch.bincount(localization_targets(iou), minlength=3)
    assert (counts > 0).all()
    return len(iou) / (3 * counts.float())


def admission(path, review_path, review_sha):
    plan = json.loads(Path(path).read_bytes())
    review = json.loads(Path(review_path).read_bytes())
    assert sha(review_path) == review_sha
    assert plan["stage"] == review["stage"] == STAGE
    assert plan["target_root"] == review["target_root"] == ROOT.as_posix()
    assert plan["target_host"] == "2026" and plan["assigned_gpu"] == "1"
    assert plan["target_hostname"] == socket.gethostname()
    assert plan["split"] == "train" and plan["test_eligible"] is False
    assert plan["exploratory_test"] is False and plan["launch_allowed"] is True
    assert plan["training_allowed"] is True
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    draft_path = ROOT / plan["reviewed_draft"]
    draft = json.loads(draft_path.read_bytes())
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(draft_path)
    assert set(draft) == set(plan) and draft["launch_allowed"] is False
    assert all(plan[k] == v for k, v in draft.items() if k != "launch_allowed")
    for kind in ("source_sha256", "input_sha256"):
        assert review[kind] == plan[kind]
        assert all(sha(ROOT / name) == digest for name, digest in plan[kind].items())
    assert plan["source_sha256"]["scripts/train_e0_reliability_balanced.py"] == sha(__file__)
    assert plan["historical_source_snapshots"] == {HISTORICAL: SNAPSHOT}
    assert plan["source_sha256"][SNAPSHOT] == OLD_WRAPPER
    original = json.loads((ROOT / plan["original_config"]).read_bytes())["plan"]
    assert plan["original_config"] in plan["input_sha256"]
    assert plan["baseline_plan"] == original
    assert original["checkpoint_sha256"] == (
        "ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c")
    assert all(plan[k] == original[k] for k in
               ("seed", "epochs", "batch_states", "lr", "weight_decay", "device"))
    assert plan["loss"] == "fit_inverse_class_weighted_cross_entropy"
    return plan


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "source-review", "source-review-sha256", "task", "run"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    plan = admission(args.plan, args.source_review, args.source_review_sha256)
    assert args.task == plan["task"] and args.run == plan["run"]
    native = json.loads((ROOT / "runs/pipeline_tasks" / args.task / "status.json").read_bytes())
    assert native["state"] == "RUNNING" and native["pid"] == os.getppid()
    assert native["hostname"] == socket.gethostname() and native["task"] == args.task
    assert native["physical_gpu"] == os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert native["command"] == [sys.executable] + sys.argv == [
        str(ROOT / "conda-envs/dual_rgbt/bin/python"),
        str(ROOT / "scripts/train_e0_reliability_balanced.py"),
        "--plan", str(ROOT / args.plan), "--source-review", str(ROOT / args.source_review),
        "--source-review-sha256", args.source_review_sha256, "--task", args.task, "--run", args.run]
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["OPENCV_FOR_THREADS_NUM"] == "1"
    started = time.perf_counter()
    data, used = load_collection(ROOT, plan["baseline_plan"], plan["historical_source_snapshots"])
    assert len(data["fit"]["iou"]) == 71843 and len(data["holdout"]["iou"]) == 19248
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    mean, scale = fit_normalization(data["fit"]["features"])
    weights = class_weights(data["fit"]["iou"]).cuda()
    for part in data.values():
        part["features"] = ((part["features"] - mean) / scale).cuda()
        part["iou"] = part["iou"].cuda()
    head = E0Reliability().cuda()
    assert sum(p.numel() for p in head.parameters()) == 117507
    before = {k: v.detach().clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=plan["lr"], weight_decay=plan["weight_decay"])
    output = ROOT / args.run
    output.mkdir(parents=True, exist_ok=False)
    config = dict(plan=plan, plan_sha256=sha(args.plan), native_command=native["command"],
                  source_review_sha256=args.source_review_sha256, cache_inputs_sha256=used,
                  class_weights=weights.cpu().tolist())
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history, best, updates, batch = [], float("-inf"), 0, plan["batch_states"]
    for epoch in range(1, plan["epochs"] + 1):
        epoch_start = time.perf_counter()
        head.train()
        order = torch.randperm(len(data["fit"]["iou"]), device="cuda")
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            optimizer.zero_grad()
            logits = head(data["fit"]["features"][index])
            loss = F.cross_entropy(logits, localization_targets(data["fit"]["iou"][index]),
                                   weight=weights)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            optimizer.step()
            updates += 1
        head.eval()
        result = {part: metrics(head, values["features"], values["iou"], batch)
                  for part, values in data.items()}
        torch.cuda.synchronize()
        history.append(dict(epoch=epoch, optimizer_updates=updates,
                            seconds=time.perf_counter() - epoch_start, **result))
        checkpoint = dict(net={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                          feature_mean=mean, feature_scale=scale, epoch=epoch, metrics=result,
                          plan_sha256=sha(args.plan),
                          protected_e0_checkpoint_sha256=plan["baseline_plan"]["checkpoint_sha256"],
                          selection_partition="train_component_holdout", total_parameters=117507,
                          loss=plan["loss"], class_weights=weights.cpu())
        torch.save(checkpoint, output / ("epoch_%03d.pth" % epoch))
        torch.save(checkpoint, output / "latest.pth")
        if result["holdout"]["macro_recall"] > best:
            best = result["holdout"]["macro_recall"]
            torch.save(checkpoint, output / "best.pth")
        if epoch > 5:
            (output / ("epoch_%03d.pth" % (epoch - 5))).unlink()
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(history[-1]), flush=True)
    assert updates == 216
    assert any(not torch.equal(v, before[k]) for k, v in head.state_dict().items())
    restored = torch.load(output / "latest.pth", map_location="cpu")
    reloaded = E0Reliability().cuda().eval()
    reloaded.load_state_dict(restored["net"], strict=True)
    with torch.no_grad():
        probe = data["holdout"]["features"][:batch]
        assert torch.equal(head(probe), reloaded(probe))
    admission(args.plan, args.source_review, args.source_review_sha256)
    receipt = dict(task=args.task, split="train", test_eligible=False, optimizer_updates=updates,
                   epochs=12, parameters=117507, strict_reload_bit_equal=True,
                   e0_loaded=False, online_policy_changed=False, plan_sha256=sha(args.plan),
                   seconds=time.perf_counter() - started, best_holdout_macro_recall=best,
                   scope="Same M1 architecture, fit-only class balancing; component holdout seen "
                         "by upstream E0. No TEST or closed-loop gain is established.",
                   artifacts_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    (output / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
