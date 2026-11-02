"""Verify terminal context TRAIN artifacts and replay retained decisions on CPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import torch
from context_e0_execution import exported_data, review_sources, verify_smoke
from context_e0_training import evaluate
from current_e0_supplement_contract import sha
from dual_rgbt.e0_context_advantage import E0ContextAdvantage
from relative_e0_training import acceptance_rank


def same_checkpoint(latest, final):
    assert set(latest) == set(final)
    assert {key: value for key, value in latest.items() if key != "net"} == {
        key: value for key, value in final.items() if key != "net"}
    assert set(latest["net"]) == set(final["net"])
    assert all(torch.equal(latest["net"][key], value) for key, value in final["net"].items())


def verify(root, plan_path):
    plan = json.loads((root / plan_path).read_bytes())
    review_sources(root, plan)
    assert plan["phase"] == "full" and plan["target_root"] == str(root)
    verify_smoke(root, plan)
    data = exported_data(root, plan)
    native_path = root / "runs/pipeline_tasks" / plan["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == plan["task"] and native["physical_gpu"] == plan["assigned_gpu"]
    assert native["hostname"] == plan["target_hostname"]
    assert native["command"] == [plan["interpreter"], str(root / "scripts/train_e0_context.py"),
                                 "--plan", plan_path]
    run = root / plan["run"]
    config = json.loads((run / "config.json").read_bytes())
    assert config == dict(plan=plan, plan_sha256=sha(root / plan_path))
    receipt = json.loads((run / "COMPLETE.json").read_bytes())
    assert receipt["status"] == "CONTEXT_FULL_TRAIN_COMPLETE"
    assert receipt["plan_sha256"] == sha(root / plan_path)
    assert receipt["epochs"] == 12 and receipt["optimizer_updates"] == 348
    assert receipt["trainable_parameters"] == 961
    for key in ("initial_all_keep", "parameters_changed", "finite_gradients",
                "normalization_fit_only", "strict_reload_bit_equal", "cuda_used"):
        assert receipt[key] is True
    assert receipt["test_eligible"] is False and receipt["online_allowed"] is False
    assert 0 < receipt["peak_allocated_bytes"] <= receipt["peak_reserved_bytes"]
    history = json.loads((run / "history.json").read_bytes())
    assert len(history) == 12
    ranks = []
    for epoch, row in enumerate(history, 1):
        assert row["epoch"] == epoch and row["optimizer_updates"] == epoch * 29
        assert all(torch.isfinite(torch.tensor(row[key])).item()
                   for key in ("loss", "advantage_loss", "protection_loss", "epoch_seconds"))
        rank = acceptance_rank(row["holdout"], epoch)
        assert row["qualifies"] is (rank is not None)
        if rank is not None:
            ranks.append((rank, epoch))
    best = max(ranks)[1] if ranks else None
    assert receipt["best_qualifying_epoch"] == best
    assert receipt["retain_protected_e0"] is (best is None)
    names = {"config.json", "initial_control.json", "COMPLETE.json", "history.json", "latest.pth"}
    names |= {"epoch_%03d.pth" % epoch for epoch in range(8, 13)}
    if best is not None:
        names.add("best.pth")
    assert {p.name for p in run.iterdir()} == names
    before = {name: sha(run / name) for name in names}
    assert receipt["latest_sha256"] == before["latest.pth"]
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(1)
    initial = E0ContextAdvantage().cpu().eval()
    initial.fit_normalization(data["fit"]["features"], data["fit"]["observed"])
    control = {part: evaluate(initial, values, plan["batch_sets"]) for part, values in data.items()}
    assert json.loads((run / "initial_control.json").read_bytes()) == control
    replayed = {}
    for name in sorted(name for name in names if name.endswith(".pth")):
        checkpoint = torch.load(run / name, map_location="cpu")
        epoch = checkpoint["epoch"]
        assert checkpoint["plan_sha256"] == sha(root / plan_path)
        assert checkpoint["dataset_sha256"] == plan["dataset_sha256"]
        assert checkpoint["test_eligible"] is False and checkpoint["trainable_parameters"] == 961
        expected_epoch = best if name == "best.pth" else 12 if name == "latest.pth" else int(name[6:9])
        assert epoch == expected_epoch
        head = E0ContextAdvantage().cpu().eval()
        head.load_state_dict(checkpoint["net"], strict=True)
        assert torch.equal(head.mean, initial.mean) and torch.equal(head.scale, initial.scale)
        assert any(not torch.equal(value, initial.state_dict()[key])
                   for key, value in head.named_parameters())
        result = {part: evaluate(head, values, plan["batch_sets"]) for part, values in data.items()}
        assert checkpoint["metrics"] == result
        assert result == {part: history[epoch - 1][part] for part in data}
        replayed[name] = result
    assert torch.load(run / "latest.pth", map_location="cpu")["epoch"] == 12
    latest = torch.load(run / "latest.pth", map_location="cpu")
    final = torch.load(run / "epoch_012.pth", map_location="cpu")
    same_checkpoint(latest, final)
    assert before == {name: sha(run / name) for name in names}
    assert not torch.cuda.is_initialized()
    return dict(
        status="CONTEXT_FULL_TRAIN_EXECUTION_VERIFIED", plan_sha256=sha(root / plan_path),
        native_sha256=sha(native_path), artifacts_sha256=before,
        best_qualifying_epoch=best, retain_protected_e0=best is None,
        trainable_parameters=961, optimizer_updates=348, retained_cpu_replay=replayed,
        cuda_initialized=False, online_allowed=False, test_eligible=False,
        scientific_result_audit_accepted=False,
        scope="Terminal native/artifact/retained checkpoint verification. Deleted early "
              "epochs have structural history checks only. No closed-loop or benchmark admission.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists()
    result = verify(ROOT, args.plan)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
