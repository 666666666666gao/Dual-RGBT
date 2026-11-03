"""Check explicit learning-rate scaling and frozen-executor FIT receipts."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dual_rgbt import e0_policy_refresh as core
from tools.train_e0_policy_refresh_scaled import fit


def tiny_model():
    path = Path(__file__).with_name("test_e0_policy_refresh.py")
    spec = importlib.util.spec_from_file_location("scaled_fit_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)

    class ReloadableExecutor(fixture.TinyExecutor):
        @torch.no_grad()
        def predict(self, templates, search, history, cost_weight, window):
            values = self.preview(search + history)
            return dict(modality=0, action=0, box=values, gains=self.gain(values))

    return ReloadableExecutor()


def test_scaled_fit_keeps_executor_and_saves_all_five_epochs(tmp_path):
    torch.manual_seed(7)
    model = tiny_model()
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    loader = DataLoader(TensorDataset(
        torch.ones(1280, 1), torch.ones(1280, 1), torch.ones(1280, 1),
        torch.tensor([[.3, .3, .3, .3]]).expand(1280, -1).clone()), batch_size=640)
    plan = dict(smoke=False, epochs=5, batch_size=640, lr_scale=5,
                preview_lr=1e-5 * 5, quality_gain_lr=1e-4 * 5,
                inputs={"executor": {"sha256": "fixture"}}, cost_weight=.05,
                round=1, branch="fixture", sources={},
                supplemental_sources={"fixture": "fixture"})
    result = fit(model, loader, plan, tmp_path, torch.ones(1), core)
    assert result["steps"] == 10 and result["records"] == 1280
    assert result["executor_preserved"] and result["strict_reload"]
    assert result["lr_scale"] == 5 and result["quality_gain_lr"] == 1e-4 * 5
    assert sorted(result["artifacts"]) == ["epoch_%03d.pth" % epoch for epoch in range(1, 6)]
    last = torch.load(tmp_path / "epoch_005.pth", map_location="cpu", weights_only=False)
    assert [group["lr"] for group in last["optimizer"]["param_groups"]] == [
        plan["preview_lr"], plan["quality_gain_lr"]]
    assert all(torch.equal(tensor, last["net"][key])
               for key, tensor in before.items() if not key.startswith(core.POLICY_PREFIXES))
    rows = list(csv.DictReader((tmp_path / "steps.csv").open()))
    assert len(rows) == 10 and rows[-1]["epoch"] == "5"


def test_scaled_fit_rejects_inconsistent_batch_learning_rate_pair(tmp_path):
    plan = dict(smoke=False, epochs=5, batch_size=640, lr_scale=1)
    with pytest.raises(AssertionError):
        fit(tiny_model(), [], plan, tmp_path, torch.ones(1), core)


def test_entry_uses_bound_dataset_with_conflicting_scripts_package(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = tmp_path / "disabled.json"
    plan.write_text(json.dumps(dict(
        root=str(root), stage="P1B_direct_E0", mode="fit",
        host="2026", launch_allowed=False)))
    command = (
        "import runpy,sys,types;"
        "sys.modules['scripts']=types.ModuleType('scripts');"
        "sys.argv=[sys.argv[1],'--plan',sys.argv[2]];"
        "runpy.run_path(sys.argv[0],run_name='__main__')")
    result = subprocess.run(
        [sys.executable, "-c", command,
         str(root / "tools/train_e0_policy_refresh_scaled.py"), str(plan)],
        cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    assert result.returncode != 0
    assert "AssertionError" in result.stderr
    assert 'assert plan["launch_allowed"] is True' in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
