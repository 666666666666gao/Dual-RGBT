"""CPU entry contracts and artifact export; no actual C2/GPU completion."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_c2_policy_completed as evaluation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return dict(path=str(path), sha256=evaluation.sha(path))


def sequence(name, frames):
    boxes = [[1., 2., 3., 4.] for _ in range(frames)]
    return dict(name=name, rgb=list(range(frames)), tir=list(range(frames)),
                initial_box=boxes[0], boxes=boxes,
                visible_boxes=boxes, infrared_boxes=[[2., 3., 4., 5.]] * frames)


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sources = {}
    for name in (
        evaluation.ENTRY, "dual_rgbt/c2_policy_evaluation.py",
        "dual_rgbt/c2_policy_refresh.py", "dual_rgbt/c2_executor.py",
        "dual_rgbt/model.py", "dual_rgbt/data.py", "dual_rgbt/tracker.py",
        "scripts/benchmark_inputs.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
        sources[name] = evaluation.sha(path)
    manifest = dict(benchmark="LasHeR", split="train_smoke", sequence_count=1,
                    frame_count=12, sequences=[sequence("train", 12)])
    manifest_item = write(tmp_path / "manifest.json", manifest)
    fit = dict(host="2025", variant="film", c2_task_id="executor", cost_weight=.05,
               run=str(tmp_path / "fit"))
    fit_item = write(tmp_path / "fit_plan.json", fit)
    path = tmp_path / "plan.json"
    plan = dict(
        stage="P2_C2_policy_evaluation", launch_allowed=True, host="2025",
        variant="film", root=str(tmp_path), task="eval_fixture", gpu="0",
        c2_task_id="executor", run_root=str(tmp_path / "eval"),
        run=str(tmp_path / "eval/eval_fixture"), smoke_frames=11,
        cost_weight=.05, smoke=True, exploratory_test=False, test_eligible=False,
        dev_enabled=False, sources=sources, fit_evidence=dict(plan=fit_item),
        checkpoint=dict(path=str(tmp_path / "fit/epoch_005.pth"), sha256="checkpoint"),
        manifest=manifest_item, benchmark="LasHeR",
        command=["python", "-B", str(tmp_path / evaluation.ENTRY), "--plan", str(path)])
    return plan, path, tmp_path, manifest


def seal(plan, root):
    binding = hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items() if key not in ("launch_allowed", "review")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan["review"] = write(root / "review.json", dict(
        stage=plan["stage"], status="SOURCE_PASS", blocking_issues=0,
        sources=plan["sources"], admitted_plan_bindings=[binding]))


def test_admitted_train_smoke(admitted):
    plan, path, root, manifest = admitted
    seal(plan, root)
    assert evaluation.validate_plan(plan, path, root) == manifest


@pytest.mark.parametrize("key,value", [
    ("launch_allowed", False), ("host", "other"), ("variant", "shared"),
    ("cost_weight", 0.), ("smoke_frames", 10), ("dev_enabled", True),
    ("test_eligible", True), ("exploratory_test", True),
    ("gpu", "4"), ("task", "../other"), ("run", "/different"),
    ("command", ["python", "wrong"]), ("c2_task_id", "other"),
])
def test_recipe_and_binding_rejected(admitted, key, value):
    plan, path, root, _ = admitted
    plan[key] = value
    seal(plan, root)
    with pytest.raises(AssertionError):
        evaluation.validate_plan(plan, path, root)


@pytest.mark.parametrize("change", ["review", "source", "manifest", "fit_plan", "final_epoch"])
def test_input_or_admission_mismatch(admitted, change):
    plan, path, root, _ = admitted
    if change == "final_epoch":
        plan["checkpoint"]["path"] = str(root / "fit/epoch_004.pth")
    seal(plan, root)
    if change == "review":
        plan["review"]["sha256"] = "wrong"
    elif change == "source":
        (root / evaluation.ENTRY).write_text("# changed\n")
    elif change in ("manifest", "fit_plan"):
        item = plan["manifest"] if change == "manifest" else plan["fit_evidence"]["plan"]
        Path(item["path"]).write_text("{}")
    with pytest.raises(AssertionError):
        evaluation.validate_plan(plan, path, root)


@pytest.mark.parametrize("benchmark", ["LasHeR", "RGBT210", "RGBT234"])
def test_full_test_requires_exact_counts(admitted, benchmark):
    plan, path, root, manifest = admitted
    plan.update(smoke=False, smoke_frames=0, exploratory_test=True, benchmark=benchmark)
    manifest.update(benchmark=benchmark, split="test")
    plan["manifest"] = write(root / "manifest.json", manifest)
    seal(plan, root)
    with pytest.raises(AssertionError):
        evaluation.validate_plan(plan, path, root)


class Tracker:
    def initialize(self, image, initial_box):
        self.box = np.asarray(initial_box)

    def track(self, image):
        return self.box.copy(), dict(
            modality=0, action=0, confidence=.7, seconds=.001,
            primary_tokens=256, auxiliary_tokens=0)


@pytest.mark.parametrize("benchmark", ["LasHeR", "RGBT210", "RGBT234"])
def test_full_tracking_exports_all_sequences(tmp_path, benchmark):
    manifest = dict(benchmark=benchmark, frame_count=6,
                    sequences=[sequence("a", 3), sequence("b", 3)])
    for name in ("traces", "predictions", "groundtruth"):
        (tmp_path / name).mkdir()
    observed = []

    def read_pair(row, frame):
        observed.append((row["name"], frame))
        return frame

    progress, timing = evaluation.track_sequences(Tracker(), manifest, tmp_path, 0, read_pair)
    assert progress == [dict(sequence="a", frames=3), dict(sequence="b", frames=3)]
    assert observed == [(name, frame) for name in ("a", "b") for frame in range(3)]
    assert timing.shape == (6,) and (timing > 0).all()
    for row in manifest["sequences"]:
        name = row["name"]
        assert np.loadtxt(tmp_path / "predictions" / (name + ".txt"), delimiter=",").shape == (3, 4)
        with (tmp_path / "traces" / (name + ".csv")).open() as handle:
            assert len(list(csv.DictReader(handle))) == 2
        if benchmark == "LasHeR":
            np.testing.assert_array_equal(
                np.loadtxt(tmp_path / "groundtruth" / (name + ".txt"), delimiter=","),
                row["boxes"])
        else:
            for modality in ("visible", "infrared"):
                expected = row[modality + "_boxes"] if benchmark == "RGBT234" else row["boxes"]
                np.testing.assert_array_equal(
                    np.loadtxt(tmp_path / "groundtruth" / name / (modality + ".txt"), delimiter=","),
                    expected)


def test_smoke_limits_to_first_eleven_train_frames(tmp_path):
    manifest = dict(benchmark="LasHeR", frame_count=24,
                    sequences=[sequence("a", 12), sequence("b", 12)])
    for name in ("traces", "predictions", "groundtruth"):
        (tmp_path / name).mkdir()
    observed = []
    progress, timing = evaluation.track_sequences(
        Tracker(), manifest, tmp_path, 11,
        lambda row, frame: observed.append((row["name"], frame)))
    assert progress == [dict(sequence="a", frames=11)] and timing.shape == (11,)
    assert observed == [("a", frame) for frame in range(11)]
    assert not (tmp_path / "predictions/b.txt").exists()


@pytest.mark.parametrize("behavior", ["valid", "mutated_state", "invalid_box"])
def test_main_terminal_receipt_requires_unchanged_valid_tracking(admitted, monkeypatch, behavior):
    plan, path, root, _ = admitted
    monkeypatch.setattr(evaluation, "ROOT", root)
    argv = [str(root / evaluation.ENTRY), "--plan", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    plan["command"] = [sys.executable, "-B", *argv]
    seal(plan, root)
    write(path, plan)
    write(root / "runs/pipeline_tasks" / plan["task"] / "status.json", dict(
        state="RUNNING", task=plan["task"], pid=os.getppid(),
        physical_gpu=plan["gpu"], command=plan["command"]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", plan["gpu"])
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "mock GPU")
    model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    monkeypatch.setattr(evaluation, "load_completed_policy", lambda *args, **kwargs: model)

    class EntryTracker(Tracker):
        def __init__(self, actual_model, cost):
            assert actual_model is model and cost == .05

        def track(self, image):
            box, route = super().track(image)
            if behavior == "mutated_state":
                model.weight.add_(1)
            elif behavior == "invalid_box":
                box[2] = 0
            return box, route

    data = types.ModuleType("dual_rgbt.data")
    data.read_pair = lambda sequence, frame: frame
    tracker_module = types.ModuleType("dual_rgbt.tracker")
    tracker_module.Tracker = EntryTracker
    monkeypatch.setitem(sys.modules, "dual_rgbt.data", data)
    monkeypatch.setitem(sys.modules, "dual_rgbt.tracker", tracker_module)
    run = Path(plan["run"])
    if behavior != "valid":
        with pytest.raises(AssertionError):
            evaluation.main()
        assert not (run / "SMOKE.json").exists() and not (run / "COMPLETE.json").exists()
        return
    evaluation.main()
    assert not (run / "COMPLETE.json").exists()
    receipt = json.loads((run / "SMOKE.json").read_bytes())
    assert receipt["frames"] == 11 and receipt["sequences"] == 1
    assert receipt["state_unchanged"] is True and receipt["smoke_only"] is True
    assert receipt["optimizer_updates"] == receipt["backward_calls"] == 0
    assert receipt["test_eligible"] is receipt["exploratory_test"] is False
    assert receipt["config_sha256"] == evaluation.sha(run / "config.json")
    assert json.loads((run / "precision.json").read_bytes()) == dict(
        matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)
