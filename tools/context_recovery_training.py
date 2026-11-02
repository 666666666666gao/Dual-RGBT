"""Fixed TRAIN run with one additional recovery regression objective."""
import json
from pathlib import Path
import time

import torch

from context_e0_training import evaluate
from context_recovery_objective import recovery_loss, recovery_weights
from current_e0_supplement_contract import sha
from dual_rgbt.e0_context_advantage import E0ContextAdvantage
from relative_e0_training import acceptance_rank


def train(data, plan, output, plan_sha):
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    head = E0ContextAdvantage()
    head.fit_normalization(data["fit"]["features"], data["fit"]["observed"])
    before = {name: value.clone() for name, value in head.state_dict().items()}
    weights = recovery_weights(data["fit"])
    assert weights.gt(0).any()
    head.to(plan["device"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=plan["lr"],
                                 weight_decay=plan["weight_decay"])
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
        sums = dict(loss=0., advantage_loss=0., protection_loss=0., recovery_loss=0.)
        seen = 0
        for start in range(0, len(order), plan["batch_sets"]):
            index = order[start:start + plan["batch_sets"]]
            values = {key: value[index].to(plan["device"]) for key, value in data["fit"].items()}
            optimizer.zero_grad(set_to_none=True)
            loss, parts = recovery_loss(
                head(values["features"], values["observed"]), values["observed"],
                values["current_iou"], values["candidate_iou"],
                values["advantage_weight"], values["protection_weight"],
                weights[index].to(plan["device"]))
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in head.parameters())
            optimizer.step()
            updates += 1
            count = len(index)
            seen += count
            sums["loss"] += float(loss.detach()) * count
            for name, value in parts.items():
                sums[name + "_loss"] += float(value.detach()) * count
            if plan["phase"] == "smoke" and updates == 5:
                break
        head.eval()
        metrics = {part: evaluate(head, values, plan["batch_sets"]) for part, values in data.items()}
        rank = acceptance_rank(metrics["holdout"], epoch)
        if plan["device"] == "cuda":
            torch.cuda.synchronize()
        row = dict(epoch=epoch, optimizer_updates=updates,
                   **{key: value / seen for key, value in sums.items()},
                   epoch_seconds=time.time() - epoch_started, qualifies=rank is not None, **metrics)
        history.append(row)
        state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        checkpoint = dict(net=state, epoch=epoch, plan_sha256=plan_sha,
                          dataset_sha256=plan["dataset_sha256"], metrics=metrics,
                          test_eligible=False, trainable_parameters=961,
                          objective=plan["objective"])
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
    assert {part: evaluate(restored, values, plan["batch_sets"]) for part, values in data.items()} == metrics
    for values in data.values():
        for start in range(0, len(values["features"]), plan["batch_sets"]):
            sample = {key: value[start:start + plan["batch_sets"]].to(plan["device"])
                      for key, value in values.items()}
            with torch.no_grad():
                assert torch.equal(head(sample["features"], sample["observed"]),
                                   restored(sample["features"], sample["observed"]))
    receipt = dict(
        status="RECOVERY_CONTEXT_SMOKE_COMPLETE" if plan["phase"] == "smoke"
               else "RECOVERY_CONTEXT_FULL_TRAIN_COMPLETE",
        split="train", test_eligible=False, online_allowed=False, plan_sha256=plan_sha,
        dataset_sha256=plan["dataset_sha256"], objective=plan["objective"],
        optimizer_updates=updates, epochs=len(history), trainable_parameters=961,
        initial_all_keep=True, parameters_changed=changed, finite_gradients=True,
        normalization_fit_only=True, recovery_weights_fit_only=True,
        strict_reload_bit_equal=True, latest_sha256=sha(output / "latest.pth"),
        best_qualifying_epoch=best_epoch, retain_protected_e0=best_epoch is None,
        cuda_used=plan["device"] == "cuda", wall_seconds=time.time() - started)
    if plan["device"] == "cuda":
        receipt.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
    with (output / "COMPLETE.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
    return receipt
