"""Train only the context advantage head; admission belongs to its entry."""
import json
from pathlib import Path
import time

import torch

from dual_rgbt.e0_context_advantage import E0ContextAdvantage, advantage_loss
from current_e0_supplement_contract import sha
from relative_e0_train_data import action_metrics
from relative_e0_training import acceptance_rank


def batch(values, index, device):
    return {key: value[index].to(device) for key, value in values.items()}


@torch.no_grad()
def evaluate(head, values, batch_sets):
    choices, device = [], next(head.parameters()).device
    for start in range(0, len(values["features"]), batch_sets):
        part = batch(values, slice(start, start + batch_sets), device)
        choices.append(head(part["features"], part["observed"]).argmax(1).cpu())
    chosen = torch.cat(choices)
    return action_metrics(chosen, values["current_iou"], values["candidate_iou"],
                          values["sequence_index"], values["labels"])


def train(data, plan, output, plan_sha):
    assert plan["phase"] in ("smoke", "full")
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    head = E0ContextAdvantage()
    head.fit_normalization(data["fit"]["features"], data["fit"]["observed"])
    before = {name: value.clone() for name, value in head.state_dict().items()}
    head.to(plan["device"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=plan["lr"],
                                 weight_decay=plan["weight_decay"])
    assert sum(p.numel() for p in head.parameters()) == 961
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(
        dict(plan=plan, plan_sha256=plan_sha), indent=2) + "\n")
    initial = {part: evaluate(head, values, plan["batch_sets"]) for part, values in data.items()}
    assert all(row["keep"] == row["sets"] for row in initial.values())
    (output / "initial_control.json").write_text(json.dumps(initial, indent=2) + "\n")
    if plan["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started, history, updates, best_rank, best_epoch = time.time(), [], 0, None, None
    for epoch in range(1, (1 if plan["phase"] == "smoke" else plan["epochs"]) + 1):
        epoch_started = time.time()
        head.train()
        order = torch.randperm(len(data["fit"]["features"]))
        loss_sum, regression_sum, protection_sum, seen = 0., 0., 0., 0
        for start in range(0, len(order), plan["batch_sets"]):
            values = batch(data["fit"], order[start:start + plan["batch_sets"]], plan["device"])
            optimizer.zero_grad(set_to_none=True)
            loss, terms = advantage_loss(
                head(values["features"], values["observed"]), values["observed"],
                values["current_iou"], values["candidate_iou"],
                values["advantage_weight"], values["protection_weight"])
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in head.parameters())
            optimizer.step()
            updates += 1
            count = len(values["features"])
            seen += count
            loss_sum += float(loss.detach()) * count
            regression_sum += float(terms["advantage"].detach()) * count
            protection_sum += float(terms["protection"].detach()) * count
            if plan["phase"] == "smoke" and updates == 5:
                break
        head.eval()
        metrics = {part: evaluate(head, values, plan["batch_sets"]) for part, values in data.items()}
        rank = acceptance_rank(metrics["holdout"], epoch)
        if plan["device"] == "cuda":
            torch.cuda.synchronize()
        row = dict(epoch=epoch, optimizer_updates=updates, loss=loss_sum / seen,
                   advantage_loss=regression_sum / seen, protection_loss=protection_sum / seen,
                   epoch_seconds=time.time() - epoch_started, qualifies=rank is not None, **metrics)
        history.append(row)
        state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        checkpoint = dict(net=state, epoch=epoch, plan_sha256=plan_sha,
                          dataset_sha256=plan["dataset_sha256"], metrics=metrics,
                          test_eligible=False, trainable_parameters=961)
        torch.save(checkpoint, output / "latest.pth")
        if plan["phase"] == "full":
            torch.save(checkpoint, output / ("epoch_%03d.pth" % epoch))
            if rank is not None and (best_rank is None or rank > best_rank):
                best_rank, best_epoch = rank, epoch
                torch.save(checkpoint, output / "best.pth")
            if epoch > 5:
                (output / ("epoch_%03d.pth" % (epoch - 5))).unlink()
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    changed = any(not torch.equal(value.detach().cpu(), before[name])
                  for name, value in head.named_parameters())
    assert changed
    assert torch.equal(head.mean.cpu(), before["mean"])
    assert torch.equal(head.scale.cpu(), before["scale"])
    restored = E0ContextAdvantage().to(plan["device"]).eval()
    restored.load_state_dict(torch.load(output / "latest.pth", map_location="cpu")["net"], strict=True)
    with torch.no_grad():
        for start in range(0, len(data["holdout"]["features"]), plan["batch_sets"]):
            values = batch(data["holdout"], slice(start, start + plan["batch_sets"]), plan["device"])
            assert torch.equal(head(values["features"], values["observed"]),
                               restored(values["features"], values["observed"]))
    receipt = dict(
        status="CONTEXT_UPDATE_SMOKE_COMPLETE" if plan["phase"] == "smoke"
               else "CONTEXT_FULL_TRAIN_COMPLETE",
        split="train", test_eligible=False, online_allowed=False, plan_sha256=plan_sha,
        optimizer_updates=updates, epochs=len(history), trainable_parameters=961,
        frozen_appearance_parameters=116482, initial_all_keep=True,
        parameters_changed=changed, finite_gradients=True, normalization_fit_only=True,
        strict_reload_bit_equal=True, latest_sha256=sha(output / "latest.pth"),
        best_qualifying_epoch=best_epoch, retain_protected_e0=best_epoch is None,
        cuda_used=plan["device"] == "cuda", wall_seconds=time.time() - started)
    if plan["device"] == "cuda":
        receipt.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
    with (output / "COMPLETE.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
    return receipt
