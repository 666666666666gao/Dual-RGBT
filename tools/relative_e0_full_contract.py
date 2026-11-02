"""Full relative TRAIN admission, bound to complete collection and real smoke."""
import json
from pathlib import Path
import socket

from current_e0_supplement_contract import sha
from relative_e0_smoke_contract import INITIAL_SHA, verify_completed
from relative_e0_train_data import load_verified_data


STAGE = "M2_relative_current_E0_full_training"
NEW_SOURCES = ("tools/relative_e0_full_contract.py", "scripts/train_e0_relative.py",
               "tools/prepare_relative_e0_full.py", "tests/test_relative_e0_full.py")
BINDINGS = {"launch_allowed", "source_review_sha256", "collection_acceptance_sha256",
            "update_smoke_plan_sha256", "update_smoke_receipt_sha256", "update_smoke_native_sha256"}


def recipe(plan):
    assert plan["stage"] == STAGE and plan["phase"] == "full"
    assert plan["split"] == "train" and plan["test_eligible"] is False
    assert plan["scientific_result_audit_accepted"] is False and plan["online_allowed"] is False
    assert plan["device"] == "cuda" and plan["assigned_gpu"] in ("0", "1")
    assert plan["seed"] == 2027 and plan["epochs"] == 12 and plan["batch_sets"] == 512
    assert plan["lr"] == .0003 and plan["weight_decay"] == .0001
    assert plan["initialization_sha256"] == INITIAL_SHA


def bind_draft(plan, draft):
    assert set(plan) == set(draft) and draft["launch_allowed"] is False
    assert {k: v for k, v in plan.items() if k not in BINDINGS} == {
        k: v for k, v in draft.items() if k not in BINDINGS}
    assert draft["source_review_sha256"] is None
    for key in BINDINGS - {"launch_allowed", "source_review_sha256"}:
        assert draft[key] is None
        assert isinstance(plan[key], str) and len(plan[key]) == 64


def admission(root, plan):
    recipe(plan)
    assert plan["launch_allowed"] is True
    assert plan["target_root"] == str(root) and plan["target_host"] == "2026"
    assert plan["target_hostname"] == socket.gethostname()
    assert sha(root / plan["initialization"]) == INITIAL_SHA
    smoke_draft = json.loads((root / plan["update_smoke_draft"]).read_bytes())
    assert sha(root / plan["update_smoke_draft"]) == plan["update_smoke_draft_sha256"]
    expected = dict(smoke_draft["source_sha256"])
    for name in NEW_SOURCES:
        expected[name] = sha(root / name)
    assert plan["source_sha256"] == expected
    for name, expected_sha in expected.items():
        assert sha(root / name) == expected_sha, name
    review = json.loads((root / plan["source_review"]).read_bytes())
    assert sha(root / plan["source_review"]) == plan["source_review_sha256"]
    assert review["stage"] == STAGE and review["status"] == "SOURCE_PASS"
    assert review["blocking_issues"] == 0
    assert review["target_host"] == "2026" and review["target_root"] == str(root)
    assert review["source_sha256"] == plan["source_sha256"]
    draft_path = root / plan["reviewed_draft"]
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(draft_path)
    bind_draft(plan, json.loads(draft_path.read_bytes()))
    smoke = json.loads((root / plan["update_smoke_plan"]).read_bytes())
    assert smoke["reviewed_draft"] == plan["update_smoke_draft"]
    assert smoke["initialization"] == plan["initialization"]
    assert smoke["full_supplement_plan"] == plan["collection_plan"]
    assert smoke["full_supplement_plan_sha256"] == plan["collection_plan_sha256"]
    assert smoke["task"] == plan["update_smoke_task"]
    assert smoke["run"] == plan["update_smoke_run"]
    verified = verify_completed(root, plan["update_smoke_plan"])
    assert verified["plan_sha256"] == plan["update_smoke_plan_sha256"]
    assert verified["receipt_sha256"] == plan["update_smoke_receipt_sha256"]
    assert verified["native_sha256"] == plan["update_smoke_native_sha256"]
    assert sha(root / plan["collection_plan"]) == plan["collection_plan_sha256"]
    assert sha(root / plan["collection_acceptance"]) == plan["collection_acceptance_sha256"]
    return load_verified_data(root, plan["collection_plan"], plan["collection_acceptance"])


def native_command(root, plan_path):
    return [str(root / "conda-envs/dual_rgbt/bin/python"),
            str(root / "scripts/train_e0_relative.py"), "--plan", plan_path]


def verify_native(root, plan, native, plan_path):
    assert native["state"] == "RUNNING" and native["task"] == plan["task"]
    assert native["hostname"] == plan["target_hostname"] == socket.gethostname()
    assert native["physical_gpu"] == plan["assigned_gpu"]
    assert native["command"] == native_command(root, plan_path)
