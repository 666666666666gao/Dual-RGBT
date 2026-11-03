"""CPU artifact-contract tests; fixtures are not real training acceptance."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from dual_rgbt.c2_policy_refresh import VARIANTS, completed_executor_state, file_sha


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return dict(path=str(path), sha256=file_sha(path))


@pytest.fixture(params=list(VARIANTS))
def completed(tmp_path, request):
    run = tmp_path / "run"
    run.mkdir()
    source = tmp_path / "source.py"
    source.write_text("# fixture source\n")
    inputs = {name: write_json(tmp_path / (name + ".json"), {})
              for name in ("factory", "manifest")}
    config = dict(
        stage="P2_C2_executor_training", host="2025", task="full_fixture",
        variant=request.param, rank=8, smoke=False, launch_allowed=True,
        epochs=20, samples=20000, test_enabled=False, dev_enabled=False,
        gpu="1", root=str(tmp_path), run=str(run), inputs=copy.deepcopy(inputs),
        sources={"source.py": file_sha(source)},
        command=["python", "-B", "train_c2_executor.py", "--plan",
                 str(tmp_path / "plan.json")])
    digest = hashlib.sha256(json.dumps(
        {key: value for key, value in config.items()
         if key not in ("launch_allowed", "review")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config["review"] = write_json(tmp_path / "review.json", dict(
        stage=config["stage"], status="SOURCE_PASS", blocking_issues=0,
        sources=config["sources"], admitted_plan_bindings=[digest]))
    inputs["c2_plan"] = write_json(tmp_path / "plan.json", config)
    inputs["c2_config"] = write_json(run / "config.json", config)
    inputs["c2_precision"] = write_json(run / "precision.json", dict(
        matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False))
    inputs["c2_native"] = write_json(tmp_path / "native.json", dict(
        state="COMPLETE", exit_code=0, task=config["task"], physical_gpu="1",
        command=config["command"]))
    missing, lora = VARIANTS[request.param]
    state = {"fixture.weight": torch.ones(2, 2)}
    artifacts = {}
    for epoch in range(16, 21):
        path = run / ("epoch_{:03d}.pth".format(epoch))
        torch.save(dict(
            epoch=epoch, step=epoch * 2500, config=config,
            c2_config=dict(missing=missing, lora=lora, rank=8), net=state), path)
        artifacts[path.name] = file_sha(path)
    inputs["executor"] = dict(path=str(run / "epoch_020.pth"),
                              sha256=artifacts["epoch_020.pth"])
    inputs["c2_complete"] = write_json(run / "COMPLETE.json", dict(
        status="COMPLETE", epochs=20, steps=50000,
        route_counts={str(route): 5000 for route in range(10)},
        retained_epochs=[16, 17, 18, 19, 20], artifacts=artifacts,
        policy_preserved=True, policy_refresh_required=True, benchmark_evidence=False))
    return dict(stage="P2_C2_policy_refresh", host="2025", c2_task_id=config["task"],
                variant=request.param, inputs=inputs)


def rewrite(binding, name, mutate):
    item = binding["inputs"][name]
    value = json.loads(Path(item["path"]).read_bytes())
    mutate(value)
    binding["inputs"][name] = write_json(Path(item["path"]), value)


def test_completed_four_variant_contracts(completed):
    saved = completed_executor_state(completed)
    assert saved["epoch"] == 20 and saved["step"] == 50000
    assert saved["net"]["fixture.weight"].device.type == "cpu"
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("name,field,value", [
    ("c2_native", "state", "RUNNING"), ("c2_native", "exit_code", 1),
    ("c2_native", "physical_gpu", "2"),
    ("c2_complete", "steps", 49999), ("c2_complete", "route_counts", {"0": 50000}),
    ("c2_complete", "artifacts", {}), ("c2_complete", "policy_preserved", False),
    ("c2_complete", "policy_refresh_required", False),
    ("c2_complete", "benchmark_evidence", True),
    ("c2_precision", "matmul_allow_tf32", True),
])
def test_rejects_incomplete_or_incompatible_evidence(completed, name, field, value):
    rewrite(completed, name, lambda doc: doc.update({field: value}))
    with pytest.raises(AssertionError):
        completed_executor_state(completed)


def test_rejects_changed_artifact_even_with_unchanged_receipt(completed):
    path = Path(completed["inputs"]["executor"]["path"]).with_name("epoch_016.pth")
    path.write_bytes(b"changed-retained-weight")
    with pytest.raises(AssertionError):
        completed_executor_state(completed)


def test_rejects_changed_executor_source(completed):
    root = Path(completed["inputs"]["c2_plan"]["path"]).parent
    (root / "source.py").write_text("# source changed after training\n")
    with pytest.raises(AssertionError):
        completed_executor_state(completed)


def test_rejects_different_branch_or_variant(completed):
    completed["host"] = "2026"
    with pytest.raises(AssertionError):
        completed_executor_state(completed)


def test_rejects_unbound_source_only_training_review(completed):
    config = json.loads(Path(completed["inputs"]["c2_config"]["path"]).read_bytes())
    path = Path(config["review"]["path"])
    review = json.loads(path.read_bytes())
    review["admitted_plan_bindings"] = []
    config["review"] = write_json(path, review)
    for name in ("c2_config", "c2_plan"):
        completed["inputs"][name] = write_json(
            Path(completed["inputs"][name]["path"]), config)
    with pytest.raises(AssertionError):
        completed_executor_state(completed)


@pytest.mark.parametrize("change", ["architecture", "config", "step", "nonfinite"])
def test_rejects_rehashed_but_incompatible_checkpoint(completed, change):
    path = Path(completed["inputs"]["executor"]["path"])
    saved = torch.load(path, map_location="cpu")
    if change == "architecture":
        saved["c2_config"]["rank"] = 16
    elif change == "config":
        saved["config"]["seed"] = 2028
    elif change == "step":
        saved["step"] = 49999
    else:
        saved["net"]["fixture.weight"][0, 0] = float("nan")
    torch.save(saved, path)
    digest = file_sha(path)
    completed["inputs"]["executor"]["sha256"] = digest
    rewrite(completed, "c2_complete",
            lambda doc: doc["artifacts"].update({path.name: digest}))
    with pytest.raises(AssertionError):
        completed_executor_state(completed)
