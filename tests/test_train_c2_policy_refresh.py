"""CPU contract tests for the future C2 policy-refresh FIT entry."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/train_c2_policy_refresh.py"
SPEC = importlib.util.spec_from_file_location("c2_policy_fit_entry", ENTRY)
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")
    return file_sha(path)


def plan_fixture(tmp_path):
    plan = dict(
        stage="P2_C2_policy_refresh_fit", mode="fit", host="2026",
        variant="film", launch_allowed=True, smoke=False, epochs=5,
        batch_size=2, workers=0, gradient_clip=1.0, preview_lr=1e-5,
        quality_gain_lr=1e-4, weight_decay=1e-4, test_enabled=False,
        dev_enabled=False, task="fit_c2_policy_film", c2_task_id="c2_film",
        root=str(tmp_path).replace("\\", "/"),
        run_root=str(tmp_path / "runs").replace("\\", "/"),
        run=str(tmp_path / "runs" / "fit_c2_policy_film").replace("\\", "/"),
        gpu="2",
        seed=2027, label_stride=1, cost_weight=.05,
        command=["python", "-B", str(ENTRY), "--plan",
                 str(tmp_path / "plan.json")],
        executor_binding=dict(
            stage="P2_C2_policy_refresh", host="2026", variant="film",
            c2_task_id="c2_film", inputs={}),
        collection_acceptance=dict(path="acceptance.json", sha256="0" * 64),
        sources={"scripts/train_c2_policy_refresh.py": file_sha(ENTRY)},
    )
    binding = trainer.plan_binding(plan)
    review = dict(
        stage=plan["stage"], status="SOURCE_PASS", blocking_issues=0,
        sources=plan["sources"], admitted_plan_bindings=[binding])
    review_path = tmp_path / "review.json"
    review_hash = write_json(review_path, review)
    plan["review"] = dict(path=str(review_path), sha256=review_hash)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan_path, plan


def test_plan_requires_exact_reviewed_source_and_binding(tmp_path):
    plan_path, plan = plan_fixture(tmp_path)
    loaded = trainer.load_plan(plan_path)
    assert loaded["task"] == plan["task"]

    plan["command"][-3] = str(ROOT / "scripts/collect_c2_policy_refresh.py")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    with pytest.raises(AssertionError):
        trainer.load_plan(plan_path)

    plan["command"][-3] = str(ENTRY)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    review = json.loads(Path(plan["review"]["path"]).read_text())
    review["admitted_plan_bindings"] = []
    Path(plan["review"]["path"]).write_text(json.dumps(review))
    with pytest.raises(AssertionError):
        trainer.load_plan(plan_path)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.preview = nn.Linear(1, 1)
        self.quality = nn.Linear(1, 1)
        self.gain = nn.Linear(1, 1)

    @torch.no_grad()
    def predict(self, templates, search, history, cost_weight, window):
        value = (self.preview(search) + self.quality(search) +
                 self.gain(search))
        return dict(modality=0, action=0, box=value, gains=value)


def test_fit_updates_only_policy_and_keeps_last_five_strict_reloads(
        tmp_path, monkeypatch):
    torch.manual_seed(3)
    model = TinyModel()
    rows = TensorDataset(
        torch.ones(4, 1), torch.arange(4, dtype=torch.float32)[:, None],
        torch.zeros(4, 1), torch.ones(4, 1))
    loader = DataLoader(rows, batch_size=2, shuffle=False)
    monkeypatch.setattr(
        trainer, "policy_refresh_loss",
        lambda model, templates, search, history, target, window: dict(
            loss=((model.preview(search) + model.quality(search) +
                   model.gain(search) - target) ** 2).mean(),
            quality_loss=search.new_ones(()),
            gain_loss=search.new_ones(()),
            spatial_loss=search.new_ones(())))
    plan = dict(
        stage="P2_C2_policy_refresh_fit", epochs=5,
        preview_lr=1e-3, quality_gain_lr=2e-3,
        weight_decay=0.0, gradient_clip=1.0, cost_weight=.05,
        collection_acceptance={},
        executor_binding=dict(inputs=dict(executor=dict(sha256="a" * 64))))
    run = tmp_path / "run"
    run.mkdir()
    before = {key: value.clone() for key, value in model.state_dict().items()}

    result = trainer.fit(model, loader, plan, run, torch.ones(1))

    assert result["steps"] == 10
    assert result["retained_epochs"] == [1, 2, 3, 4, 5]
    assert set(result["artifacts"]) == {
        "epoch_001.pth", "epoch_002.pth", "epoch_003.pth",
        "epoch_004.pth", "epoch_005.pth",
    }
    assert all(torch.equal(before[name], value)
               for name, value in model.state_dict().items()
               if not name.startswith(trainer.POLICY_PREFIXES))
    assert all(any(not torch.equal(before[name], model.state_dict()[name])
                   for name in before if name.startswith(prefix))
               for prefix in trainer.POLICY_PREFIXES)
    for path in run.glob("epoch_*.pth"):
        saved = torch.load(path, map_location="cpu")
        assert saved["net"]["backbone.weight"].equal(before["backbone.weight"])
        assert saved["step"] == int(path.stem[-3:]) * 2
    assert len((run / "steps.csv").read_text().splitlines()) == 11


def test_collection_acceptance_must_be_full_and_nonbenchmark(monkeypatch):
    plan = dict(
        collection_acceptance=dict(path="unused", sha256="0" * 64),
        host="2026", variant="film", c2_task_id="c2_film",
        executor_binding={})
    acceptance = dict(
        status="PASS", stage="P2_C2_policy_refresh_full_collection_acceptance",
        host="2026", variant="film", c2_task_id="c2_film",
        executor_binding={}, sequences=881, tracked_frames=10, records=10,
        benchmark_evidence=False, policy_fit_admitted=False,
        collections=[1, 2, 3, 4],
        native_and_artifact_evidence=[1, 2, 3, 4])
    monkeypatch.setattr(trainer, "file_sha", lambda _: "0" * 64)
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: json.dumps(
        acceptance))
    with pytest.raises(AssertionError):
        trainer.validate_collection_acceptance(plan)
