"""Exact reviewed bindings and terminal replay for the recovery objective."""
import json
from pathlib import Path
import socket

import torch

from context_e0_execution import exported_data
from context_e0_training import evaluate
from current_e0_supplement_contract import sha
from dual_rgbt.e0_context_advantage import E0ContextAdvantage
from relative_e0_training import acceptance_rank
from verify_context_e0_training import same_checkpoint


STAGE = "M2_context_recovery_execution"
OBJECTIVE = "original_advantage_plus_protection_plus_fit_recovery_regression"
BINDINGS = {"launch_allowed", "source_review_sha256",
            "smoke_receipt_sha256", "smoke_native_sha256"}


def review_sources(root, plan):
    assert plan["stage"] == STAGE and plan["objective"] == OBJECTIVE
    assert plan["split"] == "train" and plan["test_eligible"] is False
    assert plan["online_allowed"] is False and plan["scientific_result_audit_accepted"] is False
    assert plan["launch_allowed"] is True
    assert plan["target_host"] == "2025" and plan["assigned_gpu"] == "2"
    assert plan["seed"] == 2027 and plan["epochs"] == 12 and plan["batch_sets"] == 512
    assert plan["lr"] == .0003 and plan["weight_decay"] == .0001
    for name, digest in plan["source_sha256"].items():
        assert sha(root / name) == digest, name
    review_path = root / plan["source_review"]
    assert sha(review_path) == plan["source_review_sha256"]
    review = json.loads(review_path.read_bytes())
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["stage"] == STAGE and review["target_root"] == plan["target_root"]
    assert review["source_sha256"] == plan["source_sha256"]
    draft_path = root / plan["reviewed_draft"]
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(draft_path)
    draft = json.loads(draft_path.read_bytes())
    assert set(plan) == set(draft) and draft["launch_allowed"] is False
    assert all(draft[key] is None for key in BINDINGS - {"launch_allowed"})
    assert {k: v for k, v in plan.items() if k not in BINDINGS} == {
        k: v for k, v in draft.items() if k not in BINDINGS}


def data_for_plan(root, plan):
    path = root / plan["base_plan"]
    assert sha(path) == plan["base_plan_sha256"]
    base = json.loads(path.read_bytes())
    assert all(plan["source_sha256"][name] == digest
               for name, digest in base["source_sha256"].items())
    for key in ("seed", "epochs", "batch_sets", "lr", "weight_decay", "assigned_gpu",
                "dataset", "dataset_sha256", "export_receipt", "export_receipt_sha256",
                "interpreter"):
        assert plan[key] == base[key], key
    return exported_data(root, base)


def verify_run(root, plan_path):
    plan = json.loads((root / plan_path).read_bytes())
    review_sources(root, plan)
    assert plan["target_root"] == str(root)
    assert plan["target_hostname"] == socket.gethostname()
    data = data_for_plan(root, plan)
    native_path = root / "runs/pipeline_tasks" / plan["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == plan["task"] and native["physical_gpu"] == plan["assigned_gpu"]
    assert native["hostname"] == plan["target_hostname"]
    assert native["command"] == [plan["interpreter"],
                                 str(root / "scripts/train_e0_context_recovery.py"),
                                 "--plan", plan_path]
    run = root / plan["run"]
    assert json.loads((run / "config.json").read_bytes()) == dict(
        plan=plan, plan_sha256=sha(root / plan_path))
    receipt = json.loads((run / "COMPLETE.json").read_bytes())
    phase = plan["phase"]
    assert phase in ("smoke", "full")
    expected_status = ("RECOVERY_CONTEXT_SMOKE_COMPLETE" if phase == "smoke"
                       else "RECOVERY_CONTEXT_FULL_TRAIN_COMPLETE")
    assert receipt["status"] == expected_status
    assert receipt["plan_sha256"] == sha(root / plan_path)
    assert receipt["dataset_sha256"] == plan["dataset_sha256"]
    assert receipt["objective"] == plan["objective"]
    epochs, updates = (1, 5) if phase == "smoke" else (12, 348)
    assert receipt["epochs"] == epochs and receipt["optimizer_updates"] == updates
    assert receipt["trainable_parameters"] == 961
    assert receipt["test_eligible"] is False and receipt["online_allowed"] is False
    for key in ("initial_all_keep", "parameters_changed", "finite_gradients",
                "normalization_fit_only", "recovery_weights_fit_only",
                "strict_reload_bit_equal", "cuda_used"):
        assert receipt[key] is True
    assert 0 < receipt["peak_allocated_bytes"] <= receipt["peak_reserved_bytes"]
    history = json.loads((run / "history.json").read_bytes())
    assert len(history) == epochs
    ranks = []
    for epoch, row in enumerate(history, 1):
        assert row["epoch"] == epoch
        assert row["optimizer_updates"] == (5 if phase == "smoke" else epoch * 29)
        assert all(torch.isfinite(torch.tensor(row[key])).item() for key in (
            "loss", "advantage_loss", "protection_loss", "recovery_loss", "epoch_seconds"))
        rank = acceptance_rank(row["holdout"], epoch)
        assert row["qualifies"] is (rank is not None)
        if phase == "full" and rank is not None:
            ranks.append((rank, epoch))
    best = max(ranks)[1] if ranks else None
    assert receipt["best_qualifying_epoch"] == best
    assert receipt["retain_protected_e0"] is (best is None)
    names = {"config.json", "initial_control.json", "COMPLETE.json",
             "history.json", "latest.pth"}
    if phase == "full":
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
    assert json.loads((run / "initial_control.json").read_bytes()) == {
        part: evaluate(initial, values, plan["batch_sets"]) for part, values in data.items()}
    replayed = {}
    for name in sorted(name for name in names if name.endswith(".pth")):
        checkpoint = torch.load(run / name, map_location="cpu")
        expected_epoch = (best if name == "best.pth" else epochs if name == "latest.pth"
                          else int(name[6:9]))
        assert checkpoint["epoch"] == expected_epoch
        assert checkpoint["plan_sha256"] == sha(root / plan_path)
        assert checkpoint["dataset_sha256"] == plan["dataset_sha256"]
        assert checkpoint["objective"] == plan["objective"]
        assert checkpoint["test_eligible"] is False and checkpoint["trainable_parameters"] == 961
        head = E0ContextAdvantage().cpu().eval()
        head.load_state_dict(checkpoint["net"], strict=True)
        assert torch.equal(head.mean, initial.mean) and torch.equal(head.scale, initial.scale)
        assert any(not torch.equal(value, initial.state_dict()[key])
                   for key, value in head.named_parameters())
        actual = {part: evaluate(head, values, plan["batch_sets"]) for part, values in data.items()}
        assert actual == checkpoint["metrics"] == {
            part: history[expected_epoch - 1][part] for part in data}
        replayed[name] = actual
    if phase == "full":
        same_checkpoint(torch.load(run / "latest.pth", map_location="cpu"),
                        torch.load(run / "epoch_012.pth", map_location="cpu"))
    assert before == {name: sha(run / name) for name in names}
    assert not torch.cuda.is_initialized()
    return dict(status="RECOVERY_CONTEXT_" + phase.upper() + "_EXECUTION_VERIFIED",
                plan_sha256=sha(root / plan_path), native_sha256=sha(native_path),
                artifacts_sha256=before, optimizer_updates=updates,
                best_qualifying_epoch=best, retain_protected_e0=best is None,
                retained_cpu_replay=replayed, cuda_initialized=False,
                online_allowed=False, test_eligible=False,
                scientific_result_audit_accepted=False)


def verify_smoke(root, plan):
    smoke_path = root / plan["smoke_plan"]
    smoke = json.loads(smoke_path.read_bytes())
    assert smoke["phase"] == "smoke" and smoke["task"] == plan["smoke_task"]
    for key in set(plan) - {
            "phase", "task", "run", "reviewed_draft",
            "smoke_receipt_sha256", "smoke_native_sha256"}:
        assert smoke[key] == plan[key], key
    assert sha(root / smoke["run"] / "COMPLETE.json") == plan["smoke_receipt_sha256"]
    assert sha(root / "runs/pipeline_tasks" / smoke["task"] / "status.json") == plan["smoke_native_sha256"]
    return verify_run(root, plan["smoke_plan"])
