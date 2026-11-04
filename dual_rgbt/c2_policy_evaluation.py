"""Authenticate a completed C2 policy FIT and reconstruct its exact architecture."""
import hashlib
import json
from pathlib import Path

import torch

from dual_rgbt.c2_policy_refresh import completed_executor_state, file_sha

POLICY_PREFIXES = ("preview.", "quality.", "gain.")


def completed_policy_state(evidence, checkpoint, checkpoint_sha256):
    """Validate fixed epoch005 against terminal FIT and matching C2 executor."""
    assert set(evidence) == {
        "plan", "config", "native", "complete", "precision", "review",
        "acceptance", "replay_native",
    }
    documents = {}
    for name, item in evidence.items():
        assert file_sha(item["path"]) == item["sha256"], name
        documents[name] = json.loads(Path(item["path"]).read_bytes())
    plan = documents["plan"]
    assert plan == documents["config"]
    assert plan["stage"] == "P2_C2_policy_refresh_fit" and plan["mode"] == "fit"
    assert plan["host"] in ("2025", "2026") and plan["variant"] in ("zero", "film")
    assert plan["launch_allowed"] is True and plan["smoke"] is False
    assert plan["test_enabled"] is plan["dev_enabled"] is False
    assert (plan["epochs"], plan["batch_size"], plan["workers"]) == (5, 512, 4)
    assert (plan["seed"], plan["label_stride"], plan["cost_weight"]) == (2027, 5, .05)
    assert (plan["preview_lr"], plan["quality_gain_lr"],
            plan["weight_decay"], plan["gradient_clip"]) == (4e-5, 4e-4, 1e-4, 1.)
    assert plan["command"][-3:] == [
        str(Path(plan["root"]) / "scripts/train_c2_policy_refresh.py"),
        "--plan", evidence["plan"]["path"],
    ]
    assert plan["review"] == evidence["review"]
    assert plan["collection_acceptance"] == evidence["acceptance"]
    assert plan["collection_acceptance_native"] == evidence["replay_native"]
    review = documents["review"]
    assert review["stage"] == plan["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == plan["sources"]
    binding = hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items()
         if key not in ("launch_allowed", "review")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert binding in review["admitted_plan_bindings"]
    for name, digest in plan["sources"].items():
        assert file_sha(Path(plan["root"]) / name) == digest, name
    native = documents["native"]
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == plan["task"] and native["physical_gpu"] == plan["gpu"]
    assert native["command"] == plan["command"]
    assert documents["precision"] == dict(
        matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)

    accepted = documents["acceptance"]
    assert accepted["stage"] == "P2_C2_policy_refresh_full_collection_acceptance"
    assert accepted["status"] == "PASS" and accepted["cuda_initialized"] is True
    assert (accepted["sequences"], accepted["tracked_frames"],
            accepted["records"]) == (882, 458553, 91091)
    assert accepted["benchmark_evidence"] is accepted["policy_fit_admitted"] is False
    for name in ("host", "variant", "c2_task_id", "executor_binding"):
        assert accepted[name] == plan[name], name
    replay = documents["replay_native"]
    assert replay["state"] == "COMPLETE" and replay["exit_code"] == 0
    assert evidence["replay_native"]["path"] == str(
        Path(accepted["root"]) / "runs/pipeline_tasks" / replay["task"] / "status.json")
    assert replay["command"] == [
        plan["command"][0], "-B",
        str(Path(accepted["root"]) / "tools/verify_c2_policy_refresh_collection.py"),
        "--plans", *[row["plan"]["command"][-1] for row in accepted["collections"]],
        "--output", evidence["acceptance"]["path"],
    ]
    executor_binding = plan["executor_binding"]
    for name in ("host", "variant", "c2_task_id"):
        assert executor_binding[name] == plan[name], name
    original = completed_executor_state(executor_binding)

    complete = documents["complete"]
    assert complete["status"] == "COMPLETE" and complete["stage"] == plan["stage"]
    assert complete["smoke"] is False
    assert (complete["epochs"], complete["steps"], complete["records"]) == (5, 890, 91091)
    assert complete["retained_epochs"] == [1, 2, 3, 4, 5]
    assert complete["executor_preserved"] is complete["all_policy_modules_changed"] is True
    assert complete["strict_reload"] is True and complete["benchmark_evidence"] is False
    assert complete["test_enabled"] is complete["dev_enabled"] is False
    assert complete["plan_sha256"] == evidence["plan"]["sha256"]
    assert complete["collection_acceptance"] == plan["collection_acceptance"]
    assert complete["executor_binding"] == executor_binding
    assert complete["executor_sha256"] == executor_binding["inputs"]["executor"]["sha256"]
    assert set(complete["artifacts"]) == {"epoch_%03d.pth" % epoch for epoch in range(1, 6)}
    for name, digest in complete["artifacts"].items():
        assert file_sha(Path(plan["run"]) / name) == digest, name
    assert checkpoint == str(Path(plan["run"]) / "epoch_005.pth")
    assert file_sha(checkpoint) == checkpoint_sha256 == complete["artifacts"]["epoch_005.pth"]
    saved = torch.load(checkpoint, map_location="cpu")
    assert saved["config"] == plan and saved["epoch"] == 5 and saved["step"] == 890
    assert saved["executor_sha256"] == complete["executor_sha256"]
    assert set(saved["net"]) == set(original["net"])
    for name, tensor in saved["net"].items():
        assert torch.isfinite(tensor).all(), name
        if not name.startswith(POLICY_PREFIXES):
            assert torch.equal(tensor, original["net"][name]), name
    for prefix in POLICY_PREFIXES:
        assert any(not torch.equal(value, saved["net"][name])
                   for name, value in original["net"].items() if name.startswith(prefix)), prefix
    return saved, original["c2_config"]


def load_completed_policy(evidence, checkpoint, checkpoint_sha256, *, device):
    """Load the refreshed C2 state without the incompatible P1B factory path."""
    from dual_rgbt.c2_executor import C2DualRGBT

    saved, architecture = completed_policy_state(evidence, checkpoint, checkpoint_sha256)
    factory = saved["config"]["executor_binding"]["inputs"]["factory"]["path"]
    model = C2DualRGBT(factory, **architecture)
    model.load_state_dict(saved["net"], strict=True)
    return model.to(device).eval().requires_grad_(False)
