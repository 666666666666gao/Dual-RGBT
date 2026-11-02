"""Relative identity TRAIN engine; execution admission belongs to its caller."""
import json
from pathlib import Path
import time

import torch

from dual_rgbt.e0_relative_identity import E0RelativeIdentity
from relative_e0_train_data import (
    action_metrics, qualifies_for_train_acceptance, targets_and_loss)


INPUTS = ("anchor", "current", "current_coverage", "candidates", "coverage")


def forward(head, values):
    return head(*(values[key] for key in INPUTS))


def batch_values(data, index, device):
    return {key: value[index].to(device) for key, value in data.items()}


@torch.no_grad()
def evaluate(head, data, batch):
    head.eval()
    device = next(head.parameters()).device
    choices, loss_sum, supervised = [], 0., 0
    for start in range(0, len(data["labels"]), batch):
        values = batch_values(data, slice(start, start + batch), device)
        logits = forward(head, values)
        choices.append(logits.argmax(1).cpu())
        count = int(values["current_coverage"].eq(1).any(1).sum())
        if count:
            loss, _, _ = targets_and_loss(
                logits, values["current_iou"], values["candidate_iou"],
                values["current_coverage"], values["coverage"], values["labels"])
            loss_sum += float(loss) * count
            supervised += count
    assert supervised > 0
    result = action_metrics(
        torch.cat(choices), data["current_iou"], data["candidate_iou"],
        data["sequence_index"], data["labels"])
    return dict(result, loss=loss_sum / supervised, supervised_sets=supervised)


def update_epoch(head, optimizer, data, batch):
    head.train()
    device = next(head.parameters()).device
    eligible = torch.nonzero(data["current_coverage"].eq(1).any(1)).flatten()
    assert len(eligible) > 0
    order = eligible[torch.randperm(len(eligible))]
    loss_sum, updates = 0., 0
    frozen = head.no_match.detach().clone()
    for start in range(0, len(order), batch):
        values = batch_values(data, order[start:start + batch], device)
        optimizer.zero_grad(set_to_none=True)
        loss, _, trainable = targets_and_loss(
            forward(head, values), values["current_iou"], values["candidate_iou"],
            values["current_coverage"], values["coverage"], values["labels"])
        assert trainable.all() and torch.isfinite(loss)
        loss.backward()
        assert head.no_match.grad is None
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in head.parameters() if p.requires_grad)
        optimizer.step()
        assert torch.equal(head.no_match, frozen)
        loss_sum += float(loss.detach()) * len(values["labels"])
        updates += 1
    return dict(loss=loss_sum / len(eligible), optimizer_updates=updates,
                supervised_sets=len(eligible))


def acceptance_rank(metrics, epoch):
    if not qualifies_for_train_acceptance(metrics):
        return None
    return (metrics["sequence_macro_iou_delta"],
            metrics["lost_to_at_least_half"], -epoch)


def train_head(data, plan, output, provenance):
    """Caller must verify full acceptance, source admission and real update smoke."""
    assert plan["seed"] == 2027 and plan["epochs"] == 12
    assert plan["batch_sets"] == 512 and plan["lr"] == .0003
    assert plan["weight_decay"] == .0001
    assert plan["split"] == "train" and plan["test_eligible"] is False
    assert set(data) == {"fit", "holdout"}
    assert len(data["fit"]["labels"]) == 14762
    assert len(data["holdout"]["labels"]) == 3949
    assert all(value.device.type == "cpu"
               for part in data.values() for value in part.values())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device, batch = plan["device"], plan["batch_sets"]
    head = E0RelativeIdentity().to(device)
    head.load_state_dict(torch.load(plan["initialization"], map_location="cpu")["net"],
                         strict=True)
    frozen = head.no_match.detach().clone()
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=plan["lr"], weight_decay=plan["weight_decay"])
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.time()
    control = {part: evaluate(head, fields, batch) for part, fields in data.items()}
    (output / "config.json").write_text(json.dumps(
        dict(plan=plan, provenance=provenance), indent=2) + "\n")
    (output / "frozen_control.json").write_text(json.dumps(control, indent=2) + "\n")
    history, updates, best_rank, best_epoch = [], 0, None, None
    for epoch in range(1, plan["epochs"] + 1):
        epoch_started = time.time()
        trained = update_epoch(head, optimizer, data["fit"], batch)
        updates += trained["optimizer_updates"]
        result = {part: evaluate(head, fields, batch) for part, fields in data.items()}
        rank = acceptance_rank(result["holdout"], epoch)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        row = dict(epoch=epoch, optimizer_updates=updates, training=trained,
                   epoch_seconds=time.time() - epoch_started,
                   qualifies=rank is not None, **result)
        history.append(row)
        checkpoint = dict(
            net={key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
            epoch=epoch, metrics=result, provenance=provenance,
            selection_partition="train_component_holdout",
            test_eligible=False, total_parameters=sum(p.numel() for p in head.parameters()))
        torch.save(checkpoint, output / ("epoch_%03d.pth" % epoch))
        torch.save(checkpoint, output / "latest.pth")
        if rank is not None and (best_rank is None or rank > best_rank):
            best_rank, best_epoch = rank, epoch
            torch.save(checkpoint, output / "best.pth")
        if epoch > 5:
            (output / ("epoch_%03d.pth" % (epoch - 5))).unlink()
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    assert torch.equal(head.no_match, frozen)
    restored = E0RelativeIdentity().to(device).eval()
    restored.load_state_dict(
        torch.load(output / "latest.pth", map_location="cpu")["net"], strict=True)
    with torch.no_grad():
        for start in range(0, len(data["holdout"]["labels"]), batch):
            probe = batch_values(data["holdout"], slice(start, start + batch), device)
            assert torch.equal(forward(head, probe), forward(restored, probe))
    receipt = dict(
        status="RELATIVE_CURRENT_E0_OFFLINE_TRAINING_COMPLETE", split="train",
        test_eligible=False, scientific_result_audit_accepted=False,
        online_policy_changed=False, protected_e0_loaded=False,
        optimizer_updates=updates, epochs=plan["epochs"], best_qualifying_epoch=best_epoch,
        retain_protected_e0=best_epoch is None, strict_reload_bit_equal=True,
        frozen_no_match_unchanged=True, wall_seconds=time.time() - started,
        total_parameters=sum(p.numel() for p in head.parameters()), provenance=provenance)
    if torch.device(device).type == "cuda":
        receipt.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    (output / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
