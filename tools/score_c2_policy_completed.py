"""Score complete C2 artifacts with the unchanged supplied toolkit, CPU only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_inputs import groundtruth, load_manifest, sha

ENTRY = "tools/score_c2_policy_completed.py"
COUNTS = {"LasHeR": (245, 220703), "RGBT210": (210, 104706),
          "RGBT234": (234, 116650)}


def read(path):
    return json.loads(Path(path).read_bytes())


def pinned(item):
    assert sha(item["path"]) == item["sha256"], item["path"]
    return read(item["path"])


def binding(plan):
    return hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items()
         if key not in ("launch_allowed", "review")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifacts(run):
    return {path.relative_to(run).as_posix(): sha(path)
            for path in run.rglob("*") if path.is_file()}


def validate_run(plan_path, run, protocol_path, protocol_sha256, toolkit):
    plan = read(plan_path)
    assert plan["stage"] == "P2_C2_policy_evaluation"
    assert plan["launch_allowed"] is True
    assert Path(plan["root"]).resolve() == ROOT.resolve()
    assert Path(plan["run"]).resolve() == run.resolve()
    assert Path(plan["run"]) == Path(plan["run_root"]) / plan["task"]
    assert plan["host"] in ("2025", "2026") and plan["variant"] in ("zero", "film")
    assert plan["gpu"] in ("0", "1", "2", "3")
    assert plan["smoke"] is False and plan["smoke_frames"] == 0
    assert plan["cost_weight"] == .05
    assert plan["exploratory_test"] is True
    assert plan["test_eligible"] is plan["dev_enabled"] is False
    assert plan["command"][-4:] == [
        "-B", str(ROOT / "scripts/eval_c2_policy_completed.py"), "--plan", str(plan_path)]
    review = pinned(plan["review"])
    assert review["stage"] == plan["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == plan["sources"]
    assert binding(plan) in review["admitted_plan_bindings"]
    assert {ENTRY, "scripts/benchmark_inputs.py",
            "scripts/eval_c2_policy_completed.py", "dual_rgbt/c2_policy_evaluation.py",
            "dual_rgbt/c2_policy_refresh.py", "dual_rgbt/c2_executor.py",
            "dual_rgbt/model.py", "dual_rgbt/data.py",
            "dual_rgbt/tracker.py"}.issubset(plan["sources"])
    for name, digest in plan["sources"].items():
        assert sha(ROOT / name) == digest, name
    assert set(plan["fit_evidence"]) == {
        "plan", "config", "native", "complete", "precision",
        "review", "acceptance", "replay_native",
    }
    fit = {name: pinned(item) for name, item in plan["fit_evidence"].items()}
    assert fit["plan"] == fit["config"]
    assert fit["plan"]["stage"] == "P2_C2_policy_refresh_fit"
    for name in ("host", "variant", "c2_task_id", "cost_weight"):
        assert fit["plan"][name] == plan[name], name
    assert fit["native"]["state"] == "COMPLETE" and fit["native"]["exit_code"] == 0
    assert fit["native"]["task"] == fit["plan"]["task"]
    assert fit["native"]["command"] == fit["plan"]["command"]
    assert fit["complete"]["status"] == "COMPLETE"
    assert (fit["complete"]["epochs"], fit["complete"]["steps"],
            fit["complete"]["records"]) == (5, 890, 91091)
    assert fit["complete"]["plan_sha256"] == plan["fit_evidence"]["plan"]["sha256"]
    assert plan["checkpoint"]["path"] == str(Path(fit["plan"]["run"]) / "epoch_005.pth")
    assert sha(plan["checkpoint"]["path"]) == plan["checkpoint"]["sha256"]
    assert fit["complete"]["artifacts"]["epoch_005.pth"] == plan["checkpoint"]["sha256"]
    config = read(run / "config.json")
    complete = read(run / "COMPLETE.json")
    assert config["plan"] == plan
    assert config["plan_sha256"] == complete["plan_sha256"] == sha(plan_path)
    assert complete["config_sha256"] == sha(run / "config.json")
    native_path = ROOT / "runs/pipeline_tasks" / plan["task"] / "status.json"
    assert Path(config["native_status"]).resolve() == native_path.resolve()
    native = read(native_path)
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == complete["task"] == plan["task"]
    assert native["command"] == config["native_command"] == plan["command"]
    assert native["physical_gpu"] == config["physical_gpu"] == plan["gpu"]
    assert complete["stage"] == plan["stage"]
    assert complete["checkpoint_sha256"] == plan["checkpoint"]["sha256"]
    assert complete["smoke_only"] is False and complete["state_unchanged"] is True
    assert complete["optimizer_updates"] == complete["backward_calls"] == 0
    assert config["exploratory_test"] is complete["exploratory_test"] is True
    assert config["test_eligible"] is complete["test_eligible"] is False
    assert config["source_sha256"] == plan["sources"]
    precision = dict(matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)
    assert read(run / "precision.json") == fit["precision"] == precision
    manifest = pinned(plan["manifest"])
    assert manifest == load_manifest(plan["manifest"]["path"])
    benchmark = plan["benchmark"]
    assert manifest["benchmark"] == config["benchmark"] == benchmark
    assert manifest["split"] == config["split"] == "test"
    expected = COUNTS[benchmark]
    assert (manifest["sequence_count"], manifest["frame_count"]) == expected
    assert (config["split_sequences"], config["split_frames"]) == expected
    assert (complete["sequences"], complete["frames"]) == expected
    names = [row["name"] for row in manifest["sequences"]]
    assert config["sequences"] == names
    assert read(run / "progress.json") == [
        dict(sequence=row["name"], frames=len(row["rgb"])) for row in manifest["sequences"]]
    for folder, suffix in (("predictions", "*.txt"), ("traces", "*.csv")):
        assert {path.stem for path in (run / folder).glob(suffix)} == set(names)
    if benchmark == "LasHeR":
        assert {path.stem for path in (run / "groundtruth").glob("*.txt")} == set(names)
    else:
        assert {path.name for path in (run / "groundtruth").iterdir()} == set(names)
        for name in names:
            assert {path.name for path in (run / "groundtruth" / name).iterdir()} == {
                "visible.txt", "infrared.txt"}
    timing = np.load(run / "frame_seconds.npy", allow_pickle=False)
    assert timing.shape == (manifest["frame_count"],)
    assert np.isfinite(timing).all() and (timing > 0).all()
    assert np.isclose(timing.sum(), complete["seconds"], rtol=1e-12, atol=0)
    offset = 0
    for row in manifest["sequences"]:
        with (run / "traces" / (row["name"] + ".csv")).open() as handle:
            records = list(csv.DictReader(handle))
        assert [int(record["frame"]) for record in records] == list(range(1, len(row["rgb"])))
        seconds = np.asarray([float(record["seconds_including_decode"]) for record in records])
        assert np.array_equal(seconds, timing[offset + 1:offset + len(row["rgb"])])
        offset += len(row["rgb"])
    assert Path(plan["toolkit"]).resolve() == toolkit.resolve()
    assert plan["protocol"] == dict(path=str(protocol_path), sha256=protocol_sha256)
    assert sha(protocol_path) == protocol_sha256
    protocol = read(protocol_path)
    assert {
        "rgbt/metrics/base.py", "rgbt/metrics/metrics.py", "rgbt/utils.py",
        "rgbt/dataset/basedataset.py", "rgbt/dataset/lasher_dataset.py",
        "rgbt/dataset/rgbt210_dataset.py", "rgbt/dataset/rgbt234_dataset.py",
    }.issubset(protocol["toolkit_code_sha256"])
    for name, digest in protocol["toolkit_code_sha256"].items():
        assert sha(toolkit / name.replace("\\", "/")) == digest, name
    if benchmark != "LasHeR":
        assert manifest["annotation_file_sha256"]
        for name, digest in manifest["annotation_file_sha256"].items():
            assert sha(toolkit / name.replace("\\", "/")) == digest, name
    provenance = dict(
        evaluation_plan_sha256=sha(plan_path), source_sha256=plan["sources"],
        source_review_sha256=plan["review"]["sha256"],
        checkpoint_sha256=plan["checkpoint"]["sha256"],
        fit_evidence_sha256={name: item["sha256"] for name, item in plan["fit_evidence"].items()},
        manifest_sha256=plan["manifest"]["sha256"],
        native_status_sha256=sha(native_path),
        toolkit_protocol_sha256=protocol_sha256,
        toolkit_code_sha256=protocol["toolkit_code_sha256"],
        scoring_script_sha256=sha(__file__),
        metric_implementation_sha256={"scripts/benchmark_inputs.py": sha(ROOT / "scripts/benchmark_inputs.py")},
        artifacts_sha256=artifacts(run))
    return plan, config, manifest, timing, provenance


def validate_boxes(run, manifest, dataset):
    assert set(dataset.seqs_name) == {row["name"] for row in manifest["sequences"]}
    for row in manifest["sequences"]:
        name = row["name"]
        visible, infrared = groundtruth(row, manifest["benchmark"])
        prediction = np.loadtxt(run / "predictions" / (name + ".txt"),
                                delimiter=",", dtype=np.float64, ndmin=2)
        assert prediction.shape == visible.shape
        assert np.isfinite(prediction).all() and (prediction[:, 2:] > 0).all()
        if manifest["benchmark"] == "LasHeR":
            saved = np.loadtxt(run / "groundtruth" / (name + ".txt"),
                               delimiter=",", dtype=np.float32, ndmin=2)
            assert np.array_equal(saved, visible)
            assert np.array_equal(visible.round(0), np.asarray(dataset[name]))
        else:
            for modality, boxes in (("visible", visible), ("infrared", infrared)):
                saved = np.loadtxt(run / "groundtruth" / name / (modality + ".txt"),
                                   delimiter=",", dtype=np.float32, ndmin=2)
                assert np.array_equal(saved, boxes)
                assert np.array_equal(boxes, np.asarray(dataset[name][modality], dtype=np.float32))


def metric_curves(run, manifest, dataset):
    from rgbt.dataset.basedataset import TrackerResult
    from rgbt.metrics.metrics import PR, SR, MPR, MSR, PR_LasHeR, NPR, SR_LasHeR

    benchmark = manifest["benchmark"]
    metrics = (("PR", PR_LasHeR()), ("NPR", NPR()), ("SR", SR_LasHeR())) if benchmark == "LasHeR" else (
        (("PR", PR()), ("SR", SR())) if benchmark == "RGBT210" else
        (("MPR", MPR()), ("MSR", MSR())))
    names = [row["name"] for row in manifest["sequences"]]
    result = TrackerResult("Dual-RGBT-C2", str(run / "predictions"), names, "", "ltwh")
    arrays = {}
    for key, metric in metrics:
        _, curve = metric(dataset, result, names)
        curve = np.asarray(curve)
        assert curve.ndim == 2 and curve.shape[0] == len(names) and curve.shape[1] > 0
        assert np.isfinite(curve).all()
        if key in ("PR", "NPR", "MPR"):
            assert curve.shape[1] > 20
        arrays[key] = curve
    return arrays


def summarize(arrays):
    return {key: 100 * float(value[:, 20].mean() if key in ("PR", "NPR", "MPR")
                             else value.mean()) for key, value in arrays.items()}


def score(args):
    plan_path, run = Path(args.plan).resolve(), Path(args.run).resolve()
    output, toolkit = Path(args.output).resolve(), Path(args.toolkit).resolve()
    assert output != run and run not in output.parents
    plan, config, manifest, timing, provenance = validate_run(
        plan_path, run, Path(args.protocol), args.protocol_sha256, toolkit)
    sys.path.insert(0, str(toolkit))
    from rgbt import LasHeR, RGBT210, RGBT234

    dataset = {"LasHeR": LasHeR, "RGBT210": RGBT210, "RGBT234": RGBT234}[manifest["benchmark"]]()
    validate_boxes(run, manifest, dataset)
    summary = dict(
        status="SCORED_COMPLETE", stage=plan["stage"], task=plan["task"],
        host=plan["host"], variant=plan["variant"], c2_task_id=plan["c2_task_id"],
        benchmark=manifest["benchmark"], split="test",
        sequences=manifest["sequence_count"], frames=manifest["frame_count"],
        fixed_epoch=5, cost_weight=.05, exploratory_test=True, test_eligible=False,
        total_parameters=config["total_parameters"], gpu_name=config["gpu_name"],
        physical_gpu=config["physical_gpu"], input_sizes=config["input_sizes"],
        mean_ms=1000 * float(timing.mean()), p95_ms=1000 * float(np.quantile(timing, .95)),
        fps=len(timing) / float(timing.sum()), timing_scope=config["timing_scope"],
        image_decoding_included=True, initialization_included=True, model_loading_excluded=True,
        **provenance)
    curves_path = output.with_suffix(".npz")
    if output.exists():
        saved = read(output)
        assert all(saved[key] == value for key, value in summary.items())
        assert saved["curves_sha256"] == sha(curves_path)
        with np.load(curves_path, allow_pickle=False) as curves:
            expected_keys = {"PR", "NPR", "SR"} if manifest["benchmark"] == "LasHeR" else (
                {"PR", "SR"} if manifest["benchmark"] == "RGBT210" else {"MPR", "MSR"})
            assert set(curves.files) == expected_keys
            arrays = {key: curves[key] for key in curves.files}
        for key, value in arrays.items():
            assert value.ndim == 2 and value.shape[0] == manifest["sequence_count"]
            assert np.isfinite(value).all()
        assert all(saved[key] == value for key, value in summarize(arrays).items())
        assert saved["per_sequence"] == [
            dict(sequence=row["name"], **summarize(
                {key: value[index:index + 1] for key, value in arrays.items()}))
            for index, row in enumerate(manifest["sequences"])]
        print(json.dumps(dict(status="VALIDATED_EXISTING_SCORE", task=plan["task"])))
        return saved
    assert not curves_path.exists()
    arrays = metric_curves(run, manifest, dataset)
    rows = []
    for index, row in enumerate(manifest["sequences"]):
        rows.append(dict(sequence=row["name"], **summarize(
            {key: value[index:index + 1] for key, value in arrays.items()})))
    summary.update(summarize(arrays), per_sequence=rows)
    assert provenance["artifacts_sha256"] == artifacts(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    with curves_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    summary["curves_sha256"] = sha(curves_path)
    with output.open("x") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("per_sequence", "artifacts_sha256", "source_sha256")}))
    return summary


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "run", "toolkit", "protocol", "protocol-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
