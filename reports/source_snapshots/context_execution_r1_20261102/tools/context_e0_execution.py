"""Source, exported-data and native bindings for one context-head experiment."""
import json
from pathlib import Path
import socket
import sys

import torch

from context_e0_train_data import sequence_weights
from current_e0_supplement_contract import sha


STAGE = "M2_context_advantage_execution"
BINDINGS = {"launch_allowed", "source_review_sha256", "dataset_sha256",
            "export_receipt_sha256", "smoke_receipt_sha256", "smoke_native_sha256"}


def review_sources(root, plan):
    assert plan["stage"] == STAGE
    assert plan["split"] == "train" and plan["test_eligible"] is False
    assert plan["online_allowed"] is False and plan["scientific_result_audit_accepted"] is False
    assert plan["seed"] == 2027 and plan["epochs"] == 12
    assert plan["batch_sets"] == 512 and plan["lr"] == .0003
    assert plan["weight_decay"] == .0001 and plan["assigned_gpu"] == "2"
    assert plan["target_host"] == "2025"
    assert plan["launch_allowed"] is True
    for name, digest in plan["source_sha256"].items():
        assert sha(root / name) == digest, name
    review_path = root / plan["source_review"]
    assert sha(review_path) == plan["source_review_sha256"]
    review = json.loads(review_path.read_bytes())
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["stage"] == STAGE and review["source_sha256"] == plan["source_sha256"]
    assert review["target_root"] == plan["target_root"]
    assert review["export_root"] == plan["export_root"]
    draft_path = root / plan["reviewed_draft"]
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(draft_path)
    draft = json.loads(draft_path.read_bytes())
    assert set(plan) == set(draft) and draft["launch_allowed"] is False
    assert {k: v for k, v in plan.items() if k not in BINDINGS} == {
        k: v for k, v in draft.items() if k not in BINDINGS}
    assert all(draft[key] is None for key in BINDINGS - {"launch_allowed"})


def exported_data(root, plan):
    receipt_path, dataset_path = root / plan["export_receipt"], root / plan["dataset"]
    assert sha(receipt_path) == plan["export_receipt_sha256"]
    assert sha(dataset_path) == plan["dataset_sha256"]
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["status"] == "CONTEXT_TRAIN_TENSORS_EXPORTED"
    assert receipt["dataset_sha256"] == plan["dataset_sha256"]
    assert receipt["split"] == "train" and receipt["test_eligible"] is False
    assert receipt["cuda_initialized"] is False and receipt["optimizer_updates"] == 0
    for key in ("parent_plan_sha256", "parent_acceptance_sha256",
                "collection_acceptance_sha256", "initialization_sha256",
                "core_review_sha256", "source_sha256"):
        assert receipt[key] == plan[key], key
    data = torch.load(dataset_path, map_location="cpu")
    assert set(data) == {"fit", "holdout"}
    fields = {"features", "observed", "labels", "current_iou", "candidate_iou",
              "sequence_index", "advantage_weight", "protection_weight"}
    for part, count, sequences in (("fit", 14762, 706), ("holdout", 3949, 176)):
        values = data[part]
        assert set(values) == fields
        assert values["features"].shape == (count, 24, 28)
        assert values["features"].dtype == torch.float32
        assert values["observed"].shape == values["candidate_iou"].shape == (count, 24)
        assert values["observed"].dtype == torch.bool
        assert values["labels"].shape == (count, 25)
        assert values["current_iou"].shape == values["sequence_index"].shape == (count,)
        assert values["current_iou"].dtype == values["candidate_iou"].dtype == torch.float64
        assert values["sequence_index"].dtype == torch.long
        assert len(values["sequence_index"].unique()) == sequences
        assert all(v.device.type == "cpu" and not v.requires_grad and torch.isfinite(v).all()
                   for v in values.values())
        assert ((values["current_iou"] >= 0) & (values["current_iou"] <= 1)).all()
        assert ((values["candidate_iou"] >= 0) & (values["candidate_iou"] <= 1)).all()
        eligible = values["observed"].any(1)
        degrading = (values["observed"] & values["current_iou"][:, None].ge(.5)
                     & values["candidate_iou"].lt(values["current_iou"][:, None])).any(1)
        for key, mask in (("advantage_weight", eligible), ("protection_weight", degrading)):
            assert torch.equal(values[key], sequence_weights(values["sequence_index"], mask))
    assert set(data["fit"]["sequence_index"].tolist()).isdisjoint(
        data["holdout"]["sequence_index"].tolist())
    assert int(data["holdout"]["current_iou"].ge(.5).sum()) == 2899
    assert int(data["holdout"]["current_iou"].lt(.1).sum()) == 800
    return data


def verify_smoke(root, plan):
    native_path = root / "runs/pipeline_tasks" / plan["smoke_task"] / "status.json"
    assert sha(native_path) == plan["smoke_native_sha256"]
    native = json.loads(native_path.read_bytes())
    assert native["task"] == plan["smoke_task"]
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["physical_gpu"] == plan["assigned_gpu"]
    assert native["command"] == [plan["interpreter"], str(root / "scripts/train_e0_context.py"),
                                  "--plan", plan["smoke_plan"]]
    assert native["hostname"] == plan["target_hostname"]
    smoke_plan_path = root / plan["smoke_plan"]
    smoke_plan = json.loads(smoke_plan_path.read_bytes())
    review_sources(root, smoke_plan)
    assert smoke_plan["phase"] == "smoke" and smoke_plan["task"] == plan["smoke_task"]
    for key in ("dataset_sha256", "export_receipt_sha256", "source_review_sha256",
                "source_sha256", "target_root", "target_hostname", "seed", "batch_sets",
                "epochs", "lr", "weight_decay", "assigned_gpu", "interpreter"):
        assert smoke_plan[key] == plan[key], key
    receipt_path = root / smoke_plan["run"] / "COMPLETE.json"
    assert sha(receipt_path) == plan["smoke_receipt_sha256"]
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["status"] == "CONTEXT_UPDATE_SMOKE_COMPLETE"
    assert receipt["plan_sha256"] == sha(smoke_plan_path)
    assert receipt["optimizer_updates"] == 5 and receipt["parameters_changed"] is True
    assert receipt["strict_reload_bit_equal"] is True
    assert receipt["normalization_fit_only"] is True
    assert receipt["trainable_parameters"] == 961 and receipt["cuda_used"] is True
    assert receipt["initial_all_keep"] is True and receipt["finite_gradients"] is True
    assert sha(root / smoke_plan["run"] / "latest.pth") == receipt["latest_sha256"]
    assert receipt["split"] == "train" and receipt["test_eligible"] is False
    assert receipt["online_allowed"] is False and receipt["epochs"] == 1
    assert 0 < receipt["peak_allocated_bytes"] <= receipt["peak_reserved_bytes"]
    run = root / smoke_plan["run"]
    assert json.loads((run / "config.json").read_bytes()) == dict(
        plan=smoke_plan, plan_sha256=sha(smoke_plan_path))
    checkpoint = torch.load(run / "latest.pth", map_location="cpu")
    assert checkpoint["plan_sha256"] == sha(smoke_plan_path)
    assert checkpoint["dataset_sha256"] == plan["dataset_sha256"]
    assert checkpoint["epoch"] == 1 and checkpoint["trainable_parameters"] == 961
    assert checkpoint["test_eligible"] is False
    from dual_rgbt.e0_context_advantage import E0ContextAdvantage
    from context_e0_training import evaluate
    torch.manual_seed(plan["seed"])
    initial = E0ContextAdvantage().cpu().eval()
    data = exported_data(root, plan)
    initial.fit_normalization(data["fit"]["features"], data["fit"]["observed"])
    head = E0ContextAdvantage().cpu().eval()
    head.load_state_dict(checkpoint["net"], strict=True)
    assert torch.equal(head.mean, initial.mean) and torch.equal(head.scale, initial.scale)
    assert any(not torch.equal(value, initial.state_dict()[name])
               for name, value in head.named_parameters())
    assert json.loads((run / "initial_control.json").read_bytes()) == {
        part: evaluate(initial, values, plan["batch_sets"]) for part, values in data.items()}
    history = json.loads((run / "history.json").read_bytes())
    assert len(history) == 1 and history[0]["optimizer_updates"] == 5
    actual = {part: evaluate(head, values, plan["batch_sets"]) for part, values in data.items()}
    assert checkpoint["metrics"] == actual == {
        part: history[0][part] for part in data}


def training_admission(root, plan, plan_path):
    review_sources(root, plan)
    assert plan["target_root"] == str(root)
    assert plan["target_hostname"] == socket.gethostname()
    assert plan["phase"] in ("smoke", "full")
    data = exported_data(root, plan)
    if plan["phase"] == "full":
        verify_smoke(root, plan)
    else:
        assert plan["smoke_receipt_sha256"] is None and plan["smoke_native_sha256"] is None
    assert sys.executable == plan["interpreter"]
    native_path = root / "runs/pipeline_tasks" / plan["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["state"] == "RUNNING" and native["task"] == plan["task"]
    assert native["hostname"] == plan["target_hostname"]
    assert native["physical_gpu"] == plan["assigned_gpu"]
    assert native["command"] == [plan["interpreter"], str(root / "scripts/train_e0_context.py"),
                                 "--plan", plan_path]
    return data, native
