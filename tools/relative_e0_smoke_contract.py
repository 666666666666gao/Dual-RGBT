"""Admission, real prefix joins and terminal evidence for relative update smoke."""
import json
from pathlib import Path
import socket

import numpy as np
import torch

from current_e0_supplement_contract import sha, verify_run
from full_current_e0_supplement_contract import SMOKE_PLAN_SHA, SMOKE_RECEIPT_SHA, SMOKE_NATIVE_SHA
from replay_m2_train_states import box_iou, collected_iou


STAGE = "M2_relative_current_E0_update_smoke"
INITIAL_SHA = "7e7df5d25d3e8667fe38a4ca8f6ec3662308ffcb52a09bd16aecaab59a9cb4be"
NEW_SOURCES = (
    "dual_rgbt/e0_relative_identity.py", "tools/relative_e0_train_data.py",
    "tools/relative_e0_training.py", "tools/replay_m2_train_states.py",
    "tests/test_e0_relative_identity.py", "tests/test_relative_e0_train_data.py",
    "tests/test_relative_e0_training.py", "tools/relative_e0_smoke_contract.py",
    "scripts/train_e0_relative_smoke.py", "tools/prepare_relative_e0_smoke.py",
    "tests/test_relative_e0_smoke.py")


def inputs(root, plan):
    assert plan["stage"] == STAGE and plan["phase"] == "smoke"
    assert plan["target_host"] == "2026" and plan["target_root"] == str(root)
    assert plan["target_hostname"] == socket.gethostname()
    assert plan["split"] == "train" and plan["test_eligible"] is False
    assert plan["scientific_result_audit_accepted"] is False
    assert plan["full_training_allowed"] is False and plan["online_allowed"] is False
    assert plan["device"] == "cuda" and plan["assigned_gpu"] in ("0", "1")
    assert plan["seed"] == 2027 and plan["optimizer_updates"] == 5
    assert plan["batch_sets"] == 6 and plan["lr"] == .0003 and plan["weight_decay"] == .0001
    assert plan["initialization_sha256"] == INITIAL_SHA == sha(root / plan["initialization"])
    assert plan["smoke_plan_sha256"] == SMOKE_PLAN_SHA == sha(root / plan["smoke_plan"])
    smoke = json.loads((root / plan["smoke_plan"]).read_bytes())
    assert smoke["run"] == plan["smoke_run"]
    assert plan["smoke_receipt_sha256"] == SMOKE_RECEIPT_SHA == sha(
        root / plan["smoke_run"] / "COMPLETE.json")
    assert plan["smoke_native_sha256"] == SMOKE_NATIVE_SHA == sha(
        root / "runs/pipeline_tasks" / smoke["task"] / "status.json")
    actual = verify_run(root, root / plan["smoke_run"])
    assert actual["receipt_sha256"] == SMOKE_RECEIPT_SHA
    assert actual["native_sha256"] == SMOKE_NATIVE_SHA
    full = json.loads((root / plan["full_supplement_plan"]).read_bytes())
    assert sha(root / plan["full_supplement_plan"]) == plan["full_supplement_plan_sha256"]
    expected = dict(full["source_sha256"])
    for name in NEW_SOURCES:
        expected[name] = sha(root / name)
    assert plan["source_sha256"] == expected
    for name, expected_sha in plan["source_sha256"].items():
        assert sha(root / name) == expected_sha, name
    assert sha(root / plan["host_audit"]) == plan["host_audit_sha256"]
    audit = json.loads((root / plan["host_audit"]).read_bytes())
    assert audit["target_root"] == str(root) and audit["target_hostname"] == socket.gethostname()
    assert audit["source_sha256"] == full["source_sha256"]
    assert audit["initialization_sha256"] == INITIAL_SHA
    assert audit["smoke_verified"] == actual
    assert audit["task"] == plan["task"] and audit["run"] == plan["run"]
    assert audit["task_absent"] is True and audit["run_absent"] is True
    return smoke


def admission(root, plan):
    smoke = inputs(root, plan)
    assert plan["launch_allowed"] is True
    review = json.loads((root / plan["source_review"]).read_bytes())
    assert sha(root / plan["source_review"]) == plan["source_review_sha256"]
    assert review["stage"] == STAGE and review["status"] == "SOURCE_PASS"
    assert review["blocking_issues"] == 0
    assert review["target_host"] == "2026" and review["target_root"] == str(root)
    assert review["source_sha256"] == plan["source_sha256"]
    draft = json.loads((root / plan["reviewed_draft"]).read_bytes())
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(root / plan["reviewed_draft"])
    assert draft["launch_allowed"] is False and set(draft) == set(plan)
    omit = {"launch_allowed", "source_review_sha256"}
    assert {k: v for k, v in draft.items() if k not in omit} == {
        k: v for k, v in plan.items() if k not in omit}
    return smoke


def native_command(root, plan_path):
    return [str(root / "conda-envs/dual_rgbt/bin/python"),
            str(root / "scripts/train_e0_relative_smoke.py"), "--plan", plan_path]


def verify_native(root, plan, native, plan_path, state):
    assert native["state"] == state and native["task"] == plan["task"]
    assert native["hostname"] == plan["target_hostname"] == socket.gethostname()
    assert native["physical_gpu"] == plan["assigned_gpu"]
    assert native["command"] == native_command(root, plan_path)


def load_prefix_data(root, smoke):
    manifest = json.loads((root / smoke["manifest"]).read_bytes())
    sequences = {row["name"]: row for row in manifest["train"]}
    caches = {Path(name).stem: name for name in smoke["cache_sha256"]}
    parts = {key: [] for key in (
        "anchor", "current", "current_coverage", "candidates", "coverage",
        "labels", "current_iou", "candidate_iou", "sequence_index")}
    for sequence_index, name in enumerate(smoke["sequence_names"]):
        with np.load(root / smoke["run"] / (name + ".npz"), allow_pickle=False) as current, \
                np.load(root / caches[name], allow_pickle=False) as original:
            frames = current["frame"]
            selected = original["frame"] < 31
            assert np.array_equal(frames, original["frame"][selected])
            assert np.array_equal(current["predictions"], original["predictions"][:31])
            gt = np.asarray(sequences[name]["boxes"])[frames]
            boxes = original["box"][selected]
            assert np.array_equal(original["iou"][selected], collected_iou(boxes, gt))
            count = len(frames)
            values = dict(
                anchor=np.repeat(original["anchor"][None], count, axis=0),
                current=current["e0_features"].copy(), current_coverage=current["e0_coverage"].copy(),
                candidates=original["features"][selected].copy(),
                coverage=original["coverage"][selected].copy(),
                labels=original["labels"][selected].copy(),
                current_iou=box_iou(original["predictions"][frames], gt),
                candidate_iou=box_iou(boxes.reshape(-1, 4),
                                     np.repeat(gt, 24, axis=0)).reshape(count, 24),
                sequence_index=np.full(count, sequence_index, np.int64))
            for key, value in values.items():
                parts[key].append(value)
    data = {key: torch.from_numpy(np.concatenate(values)) for key, values in parts.items()}
    assert len(data["labels"]) == 6
    return data


def verify_completed(root, plan_path):
    plan = json.loads((root / plan_path).read_bytes())
    admission(root, plan)
    directory = root / plan["run"]
    config = json.loads((directory / "config.json").read_bytes())
    receipt = json.loads((directory / "COMPLETE.json").read_bytes())
    native_path = root / "runs/pipeline_tasks" / plan["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    verify_native(root, plan, native, plan_path, "COMPLETE")
    assert native["exit_code"] == 0
    assert config["plan"] == plan and config["plan_path"] == plan_path
    assert config["native_command"] == native_command(root, plan_path)
    assert config["plan_sha256"] == receipt["plan_sha256"] == sha(root / plan_path)
    assert receipt["status"] == "RELATIVE_CURRENT_E0_REAL_UPDATE_SMOKE_COMPLETE"
    assert receipt["task"] == plan["task"] and receipt["candidate_sets"] == 6
    assert receipt["optimizer_updates"] == 5 and len(receipt["losses"]) == 5
    assert np.isfinite(receipt["losses"]).all()
    for key in ("parameters_changed", "frozen_no_match_unchanged",
                "finite_gradients", "strict_reload_bit_equal"):
        assert receipt[key] is True
    for key in ("test_eligible", "full_training_allowed", "online_allowed",
                "scientific_result_audit_accepted", "protected_e0_loaded"):
        assert receipt[key] is False
    assert receipt["split"] == "train"
    assert set(receipt["artifacts_sha256"]) == {"config.json", "smoke.pth"}
    assert {p.name for p in directory.iterdir()} == {"config.json", "smoke.pth", "COMPLETE.json"}
    for name, expected in receipt["artifacts_sha256"].items():
        assert sha(directory / name) == expected
    checkpoint = torch.load(directory / "smoke.pth", map_location="cpu")
    initial = torch.load(root / plan["initialization"], map_location="cpu")["net"]
    assert checkpoint["plan_sha256"] == sha(root / plan_path)
    assert set(checkpoint["net"]) == set(initial)
    assert all(value.shape == initial[key].shape and value.dtype == initial[key].dtype
               for key, value in checkpoint["net"].items())
    assert all(torch.isfinite(value).all() for value in checkpoint["net"].values())
    assert torch.equal(checkpoint["net"]["no_match"], initial["no_match"])
    assert any(not torch.equal(value, initial[key])
               for key, value in checkpoint["net"].items() if key != "no_match")
    return dict(status="RELATIVE_CURRENT_E0_UPDATE_SMOKE_VERIFIED",
                plan_sha256=sha(root / plan_path), receipt_sha256=sha(directory / "COMPLETE.json"),
                native_sha256=sha(native_path), full_training_allowed=False,
                online_allowed=False, test_eligible=False,
                scope="Five real six-set TRAIN prefix updates; not full training or benchmark evidence.")
