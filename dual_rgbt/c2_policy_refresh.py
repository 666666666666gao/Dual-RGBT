"""C2 completion binding and strict reconstruction for later policy refresh.

This module does not admit or launch collection/training. A future versioned
entry must independently admit its sources, recipe and this executor binding.
"""
import hashlib
import json
from pathlib import Path

import torch


VARIANTS = {
    "zero": ("zero", "none"),
    "film": ("learned_film", "none"),
    "shared": ("learned_film", "shared"),
    "routed": ("learned_film", "routed"),
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def completed_executor_state(binding):
    """Verify the fixed epoch20 full C2 executor, returning its CPU state."""
    assert binding["stage"] == "P2_C2_policy_refresh"
    assert binding["host"] in ("2025", "2026")
    inputs = binding["inputs"]
    assert set(inputs) == {
        "factory", "manifest", "executor", "c2_plan", "c2_config",
        "c2_native", "c2_complete", "c2_precision",
    }
    for item in inputs.values():
        assert file_sha(item["path"]) == item["sha256"], item["path"]
    config = json.loads(Path(inputs["c2_config"]["path"]).read_bytes())
    assert config == json.loads(Path(inputs["c2_plan"]["path"]).read_bytes())
    assert config["stage"] == "P2_C2_executor_training"
    assert config["host"] == binding["host"]
    assert config["task"] == binding["c2_task_id"]
    assert config["variant"] == binding["variant"] and config["rank"] == 8
    assert config["smoke"] is False and config["launch_allowed"] is True
    assert config["epochs"] == 20 and config["samples"] == 20000
    assert config["test_enabled"] is False and config["dev_enabled"] is False
    for name in ("factory", "manifest"):
        assert config["inputs"][name] == inputs[name], name
    precision = json.loads(Path(inputs["c2_precision"]["path"]).read_bytes())
    assert precision == dict(
        matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)

    review = json.loads(Path(config["review"]["path"]).read_bytes())
    assert file_sha(config["review"]["path"]) == config["review"]["sha256"]
    assert review["stage"] == config["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == config["sources"]
    admitted = {key: value for key, value in config.items()
                if key not in ("launch_allowed", "review")}
    digest = hashlib.sha256(json.dumps(
        admitted, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest in review["admitted_plan_bindings"]

    native = json.loads(Path(inputs["c2_native"]["path"]).read_bytes())
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == config["task"]
    assert native["physical_gpu"] == config["gpu"]
    assert native["command"] == config["command"]
    assert native["command"][-2:] == ["--plan", inputs["c2_plan"]["path"]]
    terminal = json.loads(Path(inputs["c2_complete"]["path"]).read_bytes())
    assert terminal["status"] == "COMPLETE"
    assert terminal["epochs"] == 20 and terminal["steps"] == 50000
    assert terminal["route_counts"] == {str(route): 5000 for route in range(10)}
    assert terminal["retained_epochs"] == [16, 17, 18, 19, 20]
    assert terminal["policy_preserved"] is True
    assert terminal["policy_refresh_required"] is True
    assert terminal["benchmark_evidence"] is False
    expected = {"epoch_{:03d}.pth".format(epoch) for epoch in range(16, 21)}
    assert set(terminal["artifacts"]) == expected
    run = Path(config["run"])
    for name, digest in terminal["artifacts"].items():
        assert file_sha(run / name) == digest, name
    executor = Path(inputs["executor"]["path"])
    assert executor == run / "epoch_020.pth"
    assert terminal["artifacts"][executor.name] == inputs["executor"]["sha256"]
    for name, digest in config["sources"].items():
        assert file_sha(Path(config["root"]) / name) == digest, name

    saved = torch.load(executor, map_location="cpu")
    assert saved["config"] == config
    assert saved["epoch"] == 20 and saved["step"] == 50000
    missing, lora = VARIANTS[config["variant"]]
    assert saved["c2_config"] == dict(missing=missing, lora=lora, rank=config["rank"])
    assert all(torch.isfinite(tensor).all() for tensor in saved["net"].values())
    return saved


def load_completed_executor(binding, *, device):
    """Reconstruct the C2 architecture, never the original P1A DualRGBT."""
    from dual_rgbt.c2_executor import C2DualRGBT

    saved = completed_executor_state(binding)
    model = C2DualRGBT(binding["inputs"]["factory"]["path"], **saved["c2_config"])
    model.load_state_dict(saved["net"], strict=True)
    return model.to(device).eval()
