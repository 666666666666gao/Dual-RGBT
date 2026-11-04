"""CPU fixtures test the FIT boundary; they do not establish real completion."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch

from dual_rgbt import c2_policy_evaluation as evaluation
from dual_rgbt.c2_policy_refresh import file_sha


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return dict(path=str(path), sha256=file_sha(path))


@pytest.fixture(params=["zero", "film"])
def fit(tmp_path, monkeypatch, request):
    source = tmp_path / "entry.py"
    source.write_text("# fixture source\n")
    executor = dict(
        stage="P2_C2_policy_refresh", host="2025", variant=request.param,
        c2_task_id="executor_fixture", inputs=dict(
            factory=dict(path="/factory"),
            executor=dict(path="/executor", sha256="executor-sha")))
    original = dict(
        net={name: torch.ones(2) for name in (
            "backbone.weight", "missing.weight", "preview.weight", "quality.weight", "gain.weight")},
        c2_config=dict(missing="zero" if request.param == "zero" else "learned_film",
                       lora="none", rank=8))
    monkeypatch.setattr(evaluation, "completed_executor_state",
                        lambda binding: copy.deepcopy(original))
    accepted = dict(
        stage="P2_C2_policy_refresh_full_collection_acceptance", status="PASS",
        host="2025", variant=request.param, c2_task_id="executor_fixture",
        root=str(tmp_path / "collection"), executor_binding=executor,
        cuda_initialized=True, sequences=882, tracked_frames=458553, records=91091,
        benchmark_evidence=False, policy_fit_admitted=False,
        collections=[dict(plan=dict(command=["python", "--plan", "/s%d" % index]))
                     for index in range(4)])
    acceptance = write_json(tmp_path / "acceptance.json", accepted)
    replay_path = Path(accepted["root"]) / "runs/pipeline_tasks/replay/status.json"
    replay = write_json(replay_path, dict(
        state="COMPLETE", exit_code=0, task="replay",
        command=["python", "-B", str(Path(accepted["root"]) /
                 "tools/verify_c2_policy_refresh_collection.py"), "--plans",
                 "/s0", "/s1", "/s2", "/s3", "--output", acceptance["path"]]))
    plan_path = tmp_path / "plan.json"
    plan = dict(
        stage="P2_C2_policy_refresh_fit", mode="fit", host="2025",
        variant=request.param, c2_task_id="executor_fixture",
        launch_allowed=True, smoke=False, test_enabled=False, dev_enabled=False,
        epochs=5, batch_size=512, workers=4, seed=2027, label_stride=5, cost_weight=.05,
        preview_lr=4e-5, quality_gain_lr=4e-4, weight_decay=1e-4, gradient_clip=1.,
        task="fit_fixture", gpu="1", root=str(tmp_path), run=str(tmp_path / "run"),
        sources={"entry.py": file_sha(source)}, executor_binding=executor,
        collection_acceptance=acceptance, collection_acceptance_native=replay,
        command=["python", "-B", str(tmp_path / "scripts/train_c2_policy_refresh.py"),
                 "--plan", str(plan_path)])
    binding = hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items()
         if key not in ("review", "launch_allowed")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    review = write_json(tmp_path / "review.json", dict(
        stage=plan["stage"], status="SOURCE_PASS", blocking_issues=0,
        sources=plan["sources"], admitted_plan_bindings=[binding]))
    plan["review"] = review
    evidence = dict(
        plan=write_json(plan_path, plan),
        config=write_json(tmp_path / "run/config.json", plan),
        review=review, acceptance=acceptance, replay_native=replay,
        native=write_json(tmp_path / "native.json", dict(
            state="COMPLETE", exit_code=0, task=plan["task"], physical_gpu="1",
            command=plan["command"])),
        precision=write_json(tmp_path / "precision.json", dict(
            matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)))
    artifacts = {}
    state = copy.deepcopy(original["net"])
    for name in state:
        if name.startswith(evaluation.POLICY_PREFIXES):
            state[name] += 1
    for epoch in range(1, 6):
        path = tmp_path / "run" / ("epoch_%03d.pth" % epoch)
        torch.save(dict(config=plan, epoch=epoch, step=epoch * 178,
                        executor_sha256="executor-sha", net=state), path)
        artifacts[path.name] = file_sha(path)
    evidence["complete"] = write_json(tmp_path / "complete.json", dict(
        status="COMPLETE", stage=plan["stage"], smoke=False, epochs=5, steps=890,
        records=91091, retained_epochs=[1, 2, 3, 4, 5], artifacts=artifacts,
        executor_preserved=True, all_policy_modules_changed=True, strict_reload=True,
        benchmark_evidence=False, test_enabled=False, dev_enabled=False,
        plan_sha256=evidence["plan"]["sha256"], collection_acceptance=acceptance,
        executor_binding=executor, executor_sha256="executor-sha"))
    return evidence, str(tmp_path / "run/epoch_005.pth"), artifacts["epoch_005.pth"], original


def rewrite(evidence, name, **updates):
    path = Path(evidence[name]["path"])
    value = json.loads(path.read_bytes())
    value.update(updates)
    evidence[name] = write_json(path, value)


def test_completed_fixed_final_state(fit):
    evidence, checkpoint, digest, original = fit
    saved, architecture = evaluation.completed_policy_state(evidence, checkpoint, digest)
    assert architecture == original["c2_config"]
    assert saved["epoch"] == 5 and saved["net"]["missing.weight"].device.type == "cpu"
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("name,updates", [
    ("native", dict(state="RUNNING")), ("native", dict(exit_code=1)),
    ("native", dict(physical_gpu="2")), ("native", dict(command=["wrong"])),
    ("complete", dict(steps=889)), ("complete", dict(retained_epochs=[5])),
    ("complete", dict(artifacts={})), ("complete", dict(strict_reload=False)),
    ("complete", dict(executor_sha256="wrong")),
    ("complete", dict(benchmark_evidence=True)),
    ("precision", dict(cudnn_benchmark=True)),
    ("review", dict(admitted_plan_bindings=[])),
    ("replay_native", dict(state="RUNNING")),
])
def test_terminal_mismatch_rejected(fit, name, updates):
    evidence, checkpoint, digest, _ = fit
    rewrite(evidence, name, **updates)
    with pytest.raises(AssertionError):
        evaluation.completed_policy_state(evidence, checkpoint, digest)


@pytest.mark.parametrize("change", ["executor", "missing", "preview", "nan", "config", "step"])
def test_rehashed_checkpoint_mismatch_rejected(fit, change):
    evidence, checkpoint, digest, _ = fit
    saved = torch.load(checkpoint, map_location="cpu")
    if change in ("executor", "missing"):
        saved["net"]["backbone.weight" if change == "executor" else "missing.weight"] += 1
    elif change == "preview":
        saved["net"]["preview.weight"] -= 1
    elif change == "nan":
        saved["net"]["gain.weight"][0] = float("nan")
    elif change == "config":
        saved["config"]["seed"] = 2028
    else:
        saved["step"] = 889
    torch.save(saved, checkpoint)
    digest = file_sha(checkpoint)
    complete = json.loads(Path(evidence["complete"]["path"]).read_bytes())
    complete["artifacts"]["epoch_005.pth"] = digest
    rewrite(evidence, "complete", artifacts=complete["artifacts"])
    with pytest.raises(AssertionError):
        evaluation.completed_policy_state(evidence, checkpoint, digest)


def test_corrupted_retained_weight_rejected(fit):
    evidence, checkpoint, digest, _ = fit
    Path(checkpoint).with_name("epoch_001.pth").write_bytes(b"corrupted")
    with pytest.raises(AssertionError):
        evaluation.completed_policy_state(evidence, checkpoint, digest)


@pytest.mark.parametrize("change", ["evidence_hash", "source", "checkpoint_hash", "epoch"])
def test_input_integrity_rejected(fit, change):
    evidence, checkpoint, digest, _ = fit
    if change == "evidence_hash":
        evidence["native"]["sha256"] = "wrong"
    elif change == "source":
        (Path(evidence["plan"]["path"]).parent / "entry.py").write_text("# changed\n")
    elif change == "checkpoint_hash":
        digest = "wrong"
    else:
        checkpoint = str(Path(checkpoint).with_name("epoch_004.pth"))
        digest = file_sha(checkpoint)
    with pytest.raises(AssertionError):
        evaluation.completed_policy_state(evidence, checkpoint, digest)


def test_exact_c2_reconstruction(fit, monkeypatch):
    evidence, checkpoint, digest, original = fit
    observed = {}

    class Model(torch.nn.Module):
        def __init__(self, factory, **architecture):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            observed.update(factory=factory, architecture=architecture)

        def load_state_dict(self, state, strict=True):
            observed.update(strict=strict, state=state)

    executor = types.ModuleType("dual_rgbt.c2_executor")
    executor.C2DualRGBT = Model
    monkeypatch.setitem(sys.modules, "dual_rgbt.c2_executor", executor)
    model = evaluation.load_completed_policy(evidence, checkpoint, digest, device="cpu")
    assert observed["factory"] == "/factory" and observed["strict"] is True
    assert observed["architecture"] == original["c2_config"]
    assert not model.training and not model.weight.requires_grad
