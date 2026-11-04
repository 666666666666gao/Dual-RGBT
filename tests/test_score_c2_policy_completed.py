"""Synthetic CPU artifacts test C2 scoring gates, never real model acceptance."""
import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("c2_scorer", ROOT / "tools/score_c2_policy_completed.py")
scoring = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scoring)
METRIC_CURVES = scoring.metric_curves


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return dict(path=str(path), sha256=scoring.sha(path))


def seal(root, plan, path):
    plan["review"] = write(root / "review.json", dict(
        stage=plan["stage"], status="SOURCE_PASS", blocking_issues=0,
        sources=plan["sources"], admitted_plan_bindings=[scoring.binding(plan)]))
    write(path, plan)
    run = Path(plan["run"])
    config = scoring.read(run / "config.json")
    config.update(plan=copy.deepcopy(plan), plan_sha256=scoring.sha(path))
    write(run / "config.json", config)
    complete = scoring.read(run / "COMPLETE.json")
    complete.update(plan_sha256=scoring.sha(path), config_sha256=scoring.sha(run / "config.json"))
    write(run / "COMPLETE.json", complete)


@pytest.fixture(params=["LasHeR", "RGBT210", "RGBT234"])
def case(tmp_path, monkeypatch, request):
    root, benchmark = tmp_path, request.param
    monkeypatch.setattr(scoring, "ROOT", root)
    monkeypatch.setattr(scoring, "COUNTS", {benchmark: (2, 6)})
    sources = {}
    for name in (
        scoring.ENTRY, "scripts/benchmark_inputs.py", "scripts/eval_c2_policy_completed.py",
        "dual_rgbt/c2_policy_evaluation.py", "dual_rgbt/c2_policy_refresh.py",
        "dual_rgbt/c2_executor.py", "dual_rgbt/model.py",
        "dual_rgbt/data.py", "dual_rgbt/tracker.py",
    ):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# fixture\n")
        sources[name] = scoring.sha(file)
    toolkit = root / "toolkit"
    toolkit_sources = {}
    for name in (
        "rgbt/metrics/base.py", "rgbt/metrics/metrics.py", "rgbt/utils.py",
        "rgbt/dataset/basedataset.py", "rgbt/dataset/lasher_dataset.py",
        "rgbt/dataset/rgbt210_dataset.py", "rgbt/dataset/rgbt234_dataset.py",
    ):
        file = toolkit / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# toolkit fixture\n")
        toolkit_sources[name] = scoring.sha(file)
    protocol = write(root / "protocol.json", dict(toolkit_code_sha256=toolkit_sources))
    annotation = toolkit / "annotation.txt"
    annotation.write_text("annotation fixture\n")
    rows = []
    for name in ("a", "b"):
        boxes = [[1., 2., 3., 4.]] * 3
        rows.append(dict(name=name, rgb=[0, 1, 2], tir=[0, 1, 2],
                         boxes=boxes, visible_boxes=boxes,
                         infrared_boxes=[[2., 3., 4., 5.]] * 3, initial_box=boxes[0]))
    manifest = dict(benchmark=benchmark, split="test", sequence_count=2,
                    frame_count=6, sequences=rows,
                    annotation_file_sha256={"annotation.txt": scoring.sha(annotation)})
    manifest_item = write(root / "manifest.json", manifest)
    fit_root = root / "fit"
    fit_root.mkdir()
    (fit_root / "epoch_005.pth").write_bytes(b"synthetic checkpoint bytes")
    checkpoint = dict(path=str(fit_root / "epoch_005.pth"),
                      sha256=scoring.sha(fit_root / "epoch_005.pth"))
    fit_plan = dict(stage="P2_C2_policy_refresh_fit", host="2025", variant="zero",
                    c2_task_id="c2", cost_weight=.05, task="fit", command=["fit"],
                    run=str(fit_root))
    evidence = {name: write(root / ("fit_" + name + ".json"), value) for name, value in (
        ("plan", fit_plan), ("config", fit_plan),
        ("native", dict(state="COMPLETE", exit_code=0, task="fit", command=["fit"])),
        ("precision", dict(matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)),
        ("review", dict(status="SOURCE_PASS")), ("acceptance", dict(status="PASS")),
        ("replay_native", dict(state="COMPLETE", exit_code=0)),
    )}
    evidence["complete"] = write(root / "fit_complete.json", dict(
        status="COMPLETE", epochs=5, steps=890, records=91091,
        plan_sha256=evidence["plan"]["sha256"], artifacts={"epoch_005.pth": checkpoint["sha256"]}))
    run = root / "eval/eval_fixture"
    run.mkdir(parents=True)
    path = root / "plan.json"
    command = ["python", "-B", str(root / "scripts/eval_c2_policy_completed.py"), "--plan", str(path)]
    plan = dict(
        stage="P2_C2_policy_evaluation", launch_allowed=True, root=str(root),
        run=str(run), run_root=str(run.parent), task="eval_fixture",
        host="2025", variant="zero", c2_task_id="c2", gpu="0",
        smoke=False, smoke_frames=0, cost_weight=.05, exploratory_test=True,
        test_eligible=False, dev_enabled=False, command=command,
        sources=sources, fit_evidence=evidence, checkpoint=checkpoint,
        manifest=manifest_item, benchmark=benchmark, toolkit=str(toolkit), protocol=protocol)
    native_path = root / "runs/pipeline_tasks/eval_fixture/status.json"
    write(native_path, dict(state="COMPLETE", exit_code=0, task=plan["task"],
                           command=command, physical_gpu="0"))
    write(run / "config.json", dict(
        native_status=str(native_path), native_command=command, physical_gpu="0",
        exploratory_test=True, test_eligible=False, source_sha256=sources,
        benchmark=benchmark, split="test", split_sequences=2, split_frames=6,
        sequences=["a", "b"], total_parameters=10, gpu_name="fixture",
        input_sizes=dict(search=256, template=128, templates=2), timing_scope="fixture"))
    write(run / "COMPLETE.json", dict(
        task=plan["task"], stage=plan["stage"], checkpoint_sha256=checkpoint["sha256"],
        smoke_only=False, state_unchanged=True, optimizer_updates=0, backward_calls=0,
        exploratory_test=True, test_eligible=False, sequences=2, frames=6, seconds=6.))
    write(run / "precision.json", scoring.read(evidence["precision"]["path"]))
    write(run / "progress.json", [dict(sequence=row["name"], frames=3) for row in rows])
    for folder in ("predictions", "groundtruth", "traces"):
        (run / folder).mkdir()
    for row in rows:
        name = row["name"]
        np.savetxt(run / "predictions" / (name + ".txt"), row["boxes"], delimiter=",")
        if benchmark == "LasHeR":
            np.savetxt(run / "groundtruth" / (name + ".txt"), row["boxes"], delimiter=",")
        else:
            (run / "groundtruth" / name).mkdir()
            for modality in ("visible", "infrared"):
                boxes = row[modality + "_boxes"] if benchmark == "RGBT234" else row["boxes"]
                np.savetxt(run / "groundtruth" / name / (modality + ".txt"), boxes, delimiter=",")
        with (run / "traces" / (name + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, ["frame", "seconds_including_decode"])
            writer.writeheader()
            writer.writerows([dict(frame=index, seconds_including_decode=1.) for index in (1, 2)])
    np.save(run / "frame_seconds.npy", np.ones(6))
    seal(root, plan, path)

    class Dataset:
        seqs_name = ["a", "b"]

        def __getitem__(self, name):
            row = next(row for row in rows if row["name"] == name)
            if benchmark == "LasHeR":
                return row["boxes"]
            return {modality: row[modality + "_boxes"] if benchmark == "RGBT234" else row["boxes"]
                    for modality in ("visible", "infrared")}

    package = types.ModuleType("rgbt")
    package.LasHeR = package.RGBT210 = package.RGBT234 = Dataset
    monkeypatch.setitem(sys.modules, "rgbt", package)
    keys = ("PR", "NPR", "SR") if benchmark == "LasHeR" else (
        ("PR", "SR") if benchmark == "RGBT210" else ("MPR", "MSR"))
    arrays = {key: np.full((2, 21), .5) for key in keys}
    monkeypatch.setattr(scoring, "metric_curves", lambda *args: arrays)
    args = argparse.Namespace(plan=str(path), run=str(run), toolkit=str(toolkit),
                              protocol=protocol["path"], protocol_sha256=protocol["sha256"],
                              output=str(root / "score.json"))
    return args, plan, root, run, native_path, Dataset()


def test_score_and_reuse_without_metric_reexecution(case, monkeypatch):
    args, _, _, run, _, _ = case
    before = scoring.artifacts(run)
    result = scoring.score(args)
    assert result["status"] == "SCORED_COMPLETE" and result["fps"] == 1.
    assert result["test_eligible"] is False and result["fixed_epoch"] == 5
    assert scoring.artifacts(run) == before
    monkeypatch.setattr(scoring, "metric_curves", lambda *args: pytest.fail("must not rescore"))
    assert scoring.score(args) == result


@pytest.mark.parametrize("target,key,value", [
    ("native", "state", "RUNNING"), ("native", "command", ["wrong"]),
    ("native", "physical_gpu", "1"), ("complete", "frames", 5),
    ("complete", "optimizer_updates", 1), ("complete", "smoke_only", True),
    ("complete", "state_unchanged", False), ("complete", "config_sha256", "wrong"),
    ("config", "source_sha256", {}), ("precision", "cudnn_benchmark", True),
])
def test_runtime_failure_produces_no_score(case, target, key, value):
    args, _, _, run, native, _ = case
    path = native if target == "native" else run / {
        "complete": "COMPLETE.json", "config": "config.json", "precision": "precision.json"}[target]
    document = scoring.read(path)
    document[key] = value
    write(path, document)
    with pytest.raises(AssertionError):
        scoring.score(args)
    assert not Path(args.output).exists() and not Path(args.output).with_suffix(".npz").exists()


@pytest.mark.parametrize("target", ["checkpoint", "fit_plan", "toolkit", "manifest", "source"])
def test_changed_pinned_bytes_rejected(case, target):
    args, plan, root, _, _, _ = case
    paths = dict(
        checkpoint=Path(plan["checkpoint"]["path"]),
        fit_plan=Path(plan["fit_evidence"]["plan"]["path"]),
        toolkit=Path(args.toolkit) / "rgbt/metrics/metrics.py",
        manifest=Path(plan["manifest"]["path"]), source=root / "dual_rgbt/tracker.py")
    paths[target].write_text("changed")
    with pytest.raises(AssertionError):
        scoring.score(args)


@pytest.mark.parametrize("change", ["trace", "extra_trace", "prediction", "gt", "progress", "timing"])
def test_bad_full_artifacts_rejected(case, change):
    args, _, _, run, _, _ = case
    if change == "trace":
        (run / "traces/a.csv").write_text("frame,seconds_including_decode\n2,1\n1,1\n")
    elif change == "extra_trace":
        (run / "traces/extra.csv").write_text("extra")
    elif change == "prediction":
        np.savetxt(run / "predictions/a.txt", [[1, 2, -3, 4]] * 3, delimiter=",")
    elif change == "gt":
        path = run / ("groundtruth/a.txt" if scoring.read(run / "config.json")["benchmark"] == "LasHeR"
                      else "groundtruth/a/infrared.txt")
        np.savetxt(path, [[9, 2, 3, 4]] * 3, delimiter=",")
    elif change == "progress":
        write(run / "progress.json", [])
    else:
        np.save(run / "frame_seconds.npy", np.zeros(6))
    with pytest.raises(AssertionError):
        scoring.score(args)


@pytest.mark.parametrize("change", ["metric", "row", "curves", "artifact", "native"])
def test_saved_score_tampering_rejected_without_rescore(case, monkeypatch, change):
    args, _, _, run, native, _ = case
    scoring.score(args)
    monkeypatch.setattr(scoring, "metric_curves", lambda *args: pytest.fail("must not rescore"))
    if change in ("metric", "row"):
        saved = scoring.read(args.output)
        key = next(key for key in ("PR", "MPR") if key in saved)
        if change == "metric":
            saved[key] = 99.
        else:
            saved["per_sequence"][0][key] = 99.
        write(Path(args.output), saved)
    elif change == "curves":
        Path(args.output).with_suffix(".npz").write_bytes(b"changed")
    elif change == "artifact":
        (run / "extra.txt").write_text("changed")
    else:
        document = scoring.read(native)
        document["new_field"] = "changed"
        write(native, document)
    with pytest.raises(AssertionError):
        scoring.score(args)


def test_full_counts_remain_required(case, monkeypatch):
    args, _, _, _, _, _ = case
    monkeypatch.setattr(scoring, "COUNTS", {
        "LasHeR": (245, 220703), "RGBT210": (210, 104706), "RGBT234": (234, 116650)})
    with pytest.raises(AssertionError):
        scoring.score(args)


def test_direct_toolkit_metric_dispatch_and_aggregation(case, monkeypatch):
    args, _, _, run, _, dataset = case
    manifest = scoring.load_manifest(scoring.read(args.plan)["manifest"]["path"])
    observed = []
    metric_module = types.ModuleType("rgbt.metrics.metrics")
    classes = ("PR", "SR", "MPR", "MSR", "PR_LasHeR", "NPR", "SR_LasHeR")
    for index, name in enumerate(classes):
        def call(self, actual_dataset, result, names, metric_name=name, value=(index + 1) / 10):
            assert actual_dataset is dataset and names == ["a", "b"]
            observed.append(metric_name)
            array = np.full((2, 21), value)
            return array.mean(axis=0), array
        setattr(metric_module, name, type(name, (), {"__call__": call}))
    result_module = types.ModuleType("rgbt.dataset.basedataset")

    def tracker_result(name, predictions, names, suffix, box_format):
        assert predictions == str(run / "predictions") and names == ["a", "b"]
        assert suffix == "" and box_format == "ltwh"
        return object()

    result_module.TrackerResult = tracker_result
    monkeypatch.setitem(sys.modules, "rgbt.metrics.metrics", metric_module)
    monkeypatch.setitem(sys.modules, "rgbt.dataset.basedataset", result_module)
    arrays = METRIC_CURVES(run, manifest, dataset)
    expected = {"LasHeR": ["PR_LasHeR", "NPR", "SR_LasHeR"],
                "RGBT210": ["PR", "SR"], "RGBT234": ["MPR", "MSR"]}[manifest["benchmark"]]
    assert observed == expected
    assert all(np.isfinite(value) for value in scoring.summarize(arrays).values())


@pytest.mark.parametrize("change", ["protocol", "binding", "review", "smoke", "cost"])
def test_admission_contract_rejected(case, change):
    args, plan, root, _, _, _ = case
    if change == "protocol":
        plan["protocol"]["sha256"] = "wrong"
    elif change == "smoke":
        plan.update(smoke=True, smoke_frames=11)
    elif change == "cost":
        plan["cost_weight"] = 0.
    seal(root, plan, Path(args.plan))
    if change in ("binding", "review"):
        report = scoring.read(plan["review"]["path"])
        report["admitted_plan_bindings" if change == "binding" else "status"] = (
            [] if change == "binding" else "SOURCE_FAIL")
        plan["review"] = write(Path(plan["review"]["path"]), report)
        write(Path(args.plan), plan)
    with pytest.raises(AssertionError):
        scoring.score(args)
