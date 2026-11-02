"""Admission for fixed paired TRAIN trajectories after qualified relative training."""
import json
from pathlib import Path
import socket

from current_e0_supplement_contract import sha


STAGE = "M2_relative_current_E0_paired_train"
FOLDER = "reports/e0_relative_pair_train_r1_20261102"
FULL = "reports/e0_relative_full_training_r1_20261102"
RUN = "runs/e0_relative_pair_train_r1_p26_20261102"
TASK = "check_e0_relative_pair_train_r1_p26_20261102"
NAMES = ("boysumbrella", "boyshorts", "blackcarturn183", "leftclosedexersicebook",
         "whiteboyback", "gonemoto_ab", "stronggirl", "ab_motoinrain")
PARENT = "ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c"
M1 = "658c84275ad1e1624d4b453fee89dd312273633664024bf0b42ae78dcc189ebf"
CORE_REVIEW = "reports/E0_M2_RELATIVE_RUNTIME_CORE_SOURCE_REVIEW_R1_20261102.json"
CORE_SHA = "aeec8061961b1d2c084dedf1a4f93b75aa4fda044aa7593e64ba7e5d7d347c17"
NEW_SOURCES = (
    "tools/relative_e0_pair_contract.py", "scripts/check_e0_relative_pair_train.py",
    "tools/prepare_e0_relative_pair_train.py", "tests/test_relative_e0_pair.py",
    "tools/verify_relative_e0_training.py", "tests/test_verify_relative_e0_training.py",
    "dual_rgbt/e0_reliability.py")
BINDINGS = {"launch_allowed", "source_review_sha256", "training_plan_sha256",
            "training_acceptance_sha256", "relative_checkpoint_sha256", "qualifying_epoch"}


def recipe(plan):
    assert plan["stage"] == STAGE and plan["phase"] == "paired_train"
    assert plan["split"] == "train"
    for key in ("training_allowed", "exploratory_test", "test_eligible",
                "benchmark_allowed", "scientific_result_audit_accepted"):
        assert plan[key] is False
    assert plan["target_host"] == "2026" and plan["target_root"] == "/data2/gb/Dual-RGBT"
    assert plan["assigned_gpu"] == "0" and plan["seed"] == 2027 and plan["cost_weight"] == 0.
    assert plan["task"] == TASK and plan["run"] == RUN
    assert plan["reviewed_draft"] == FOLDER + "/draft.json"
    assert plan["source_review"] == "reports/E0_M2_RELATIVE_PAIR_TRAIN_SOURCE_REVIEW_R1_20261102.json"
    assert plan["training_draft"] == FULL + "/draft.json"
    assert plan["training_plan"] == FULL + "/plan.json"
    assert plan["training_acceptance"] == "reports/E0_M2_RELATIVE_FULL_TRAIN_ACCEPTANCE_R1_20261102.json"
    assert plan["relative_checkpoint"] == "runs/e0_relative_full_r1_p26_20261102/best.pth"
    assert plan["sequence_names"] == list(NAMES) and plan["frames"] == 3076
    assert plan["arms"] == ["e0_m1_control", "e0_m1_relative_m2"]
    assert plan["input_sha256"][plan["parent_checkpoint"]] == PARENT
    assert plan["input_sha256"][plan["m1_checkpoint"]] == M1


def bind_draft(plan, draft):
    recipe(plan)
    recipe(draft)
    assert set(plan) == set(draft) and draft["launch_allowed"] is False
    assert {k: v for k, v in plan.items() if k not in BINDINGS} == {
        k: v for k, v in draft.items() if k not in BINDINGS}
    for key in BINDINGS - {"launch_allowed", "qualifying_epoch"}:
        assert draft[key] is None
        assert isinstance(plan[key], str) and len(plan[key]) == 64
        assert all(c in "0123456789abcdef" for c in plan[key])
    assert draft["qualifying_epoch"] is None
    assert type(plan["qualifying_epoch"]) is int and 1 <= plan["qualifying_epoch"] <= 12


def immutable_inputs(root, plan):
    for name, digest in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        assert sha(root / name) == digest, name
    for name, key in ((plan["training_plan"], "training_plan_sha256"),
                      (plan["training_acceptance"], "training_acceptance_sha256"),
                      (plan["relative_checkpoint"], "relative_checkpoint_sha256"),
                      (plan["source_review"], "source_review_sha256")):
        assert sha(root / name) == plan[key], name


def qualifying_training(root, plan):
    from verify_relative_e0_training import verify_completed

    verified = verify_completed(root, plan["training_plan"])
    assert json.loads((root / plan["training_acceptance"]).read_bytes()) == verified
    assert verified["status"] == "RELATIVE_CURRENT_E0_FULL_TRAINING_VERIFIED"
    assert verified["retain_protected_e0"] is False
    assert verified["best_qualifying_epoch"] == plan["qualifying_epoch"]
    assert verified["artifacts_sha256"]["best.pth"] == plan["relative_checkpoint_sha256"]
    assert verified["plan_sha256"] == plan["training_plan_sha256"]
    return verified


def admission(root, plan):
    recipe(plan)
    assert plan["launch_allowed"] is True
    assert str(root) == plan["target_root"] and socket.gethostname() == plan["target_hostname"]
    immutable_inputs(root, plan)
    assert sha(root / CORE_REVIEW) == CORE_SHA
    core = json.loads((root / CORE_REVIEW).read_bytes())
    assert core["stage"] == "M2_relative_current_E0_runtime_core"
    assert core["status"] == "CORE_SOURCE_PASS" and core["blocking_issues"] == 0
    training = json.loads((root / plan["training_draft"]).read_bytes())
    expected = dict(training["source_sha256"])
    for mapping in (core["source_sha256"], core["read_only_dependency_sha256"]):
        for name, digest in mapping.items():
            if name in expected:
                assert expected[name] == digest, name
            expected[name] = digest
    for name in NEW_SOURCES:
        expected[name] = sha(root / name)
    assert plan["source_sha256"] == expected
    review = json.loads((root / plan["source_review"]).read_bytes())
    assert review["stage"] == STAGE and review["status"] == "SOURCE_PASS"
    assert review["blocking_issues"] == 0
    assert review["target_host"] == "2026" and review["target_root"] == str(root)
    assert review["source_sha256"] == plan["source_sha256"]
    assert review["input_sha256"] == plan["input_sha256"]
    draft_path = root / plan["reviewed_draft"]
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(draft_path)
    bind_draft(plan, json.loads(draft_path.read_bytes()))
    verified = qualifying_training(root, plan)
    manifest = json.loads((root / plan["manifest"]).read_bytes())
    rows = {row["name"]: row for row in manifest["train"]}
    forbidden = set(manifest["test_sequences"]) | {row["name"] for row in manifest["dev"]}
    assert not set(NAMES) & forbidden and set(NAMES) <= set(rows)
    selected = [rows[name] for name in NAMES]
    assert {row["name"]: len(row["rgb"]) for row in selected} == plan["sequence_frames"]
    assert sum(len(row["rgb"]) for row in selected) == plan["frames"]
    assert all(len(row["rgb"]) == len(row["tir"]) == len(row["boxes"]) for row in selected)
    return selected, verified


def native_command(root, plan_path):
    return [str(root / "conda-envs/dual_rgbt/bin/python"),
            str(root / "scripts/check_e0_relative_pair_train.py"),
            "--plan", str(root / plan_path)]
