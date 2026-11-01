"""Score one complete protected E0 / M1 run with the unchanged official toolkit."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_inputs import groundtruth, load_manifest, sha
from joint_candidate_results import official_metrics


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "toolkit", "protocol", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    assert not output.exists() and not output.with_suffix(".npz").exists()
    config = json.loads((run / "config.json").read_bytes())
    complete = json.loads((run / "COMPLETE.json").read_bytes())
    plan = json.loads(Path(config["plan"]).read_bytes())
    review = json.loads(Path(config["source_review"]).read_bytes())
    assert sha(config["plan"]) == config["plan_sha256"]
    assert sha(config["source_review"]) == config["source_review_sha256"]
    assert config["stage"] == plan["stage"] == review["stage"] == "E0_M1_template_guard_evaluation"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["target_root"] == plan["target_root"] == ROOT.as_posix()
    assert review["reviewed_drafts"][Path(config["plan"]).name] == config["plan_sha256"]
    assert config["source_sha256"] == plan["source_sha256"] == review["source_sha256"]
    assert all(sha(ROOT / name) == digest for name, digest in plan["source_sha256"].items())
    assert all(sha(ROOT / name) == digest for name, digest in plan["input_sha256"].items())
    assert all(config[key] == value for key, value in plan.items())
    assert plan["cost_weight"] == 0.0
    assert plan["matmul_tf32"] is plan["cudnn_tf32"] is False
    assert plan["training_allowed"] is plan["test_eligible"] is False
    assert plan["mode"] in ("control", "guard")
    assert config["split"] == "test" and plan["smoke_frames"] == 0
    native = json.loads(Path(config["native_status"]).read_bytes())
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == complete["task"] == config["task"]
    assert native["command"] == config["native_command"]
    assert native["physical_gpu"] == config["physical_gpu"]
    assert complete["config_sha256"] == sha(run / "config.json")
    assert complete["optimizer_updates"] == complete["backward_calls"] == 0
    assert complete["state_unchanged"] and not complete["smoke_only"]
    assert config["exploratory_test"] is complete["exploratory_test"] is True
    assert config["test_eligible"] is complete["test_eligible"] is False
    assert complete["checkpoint_sha256"] == plan["checkpoint_sha256"]
    assert complete["m1_sha256"] == plan["m1_sha256"]
    assert complete["artifacts_sha256"] == {
        p.relative_to(run).as_posix(): sha(p) for p in run.rglob("*")
        if p.is_file() and p.name != "COMPLETE.json"}
    manifest = load_manifest(ROOT / plan["manifest"])
    benchmark = manifest["benchmark"]
    assert config["benchmark"] == benchmark
    names = [row["name"] for row in manifest["sequences"]]
    assert config["sequences"] == names
    assert {p.stem for p in (run / "predictions").glob("*.txt")} == set(names)
    expected = {"LasHeR": 245, "RGBT210": 210, "RGBT234": 234}[benchmark]
    assert complete["sequences"] == config["split_sequences"] == manifest["sequence_count"] == expected
    assert complete["frames"] == config["split_frames"] == manifest["frame_count"]
    toolkit = Path(args.toolkit)
    protocol = json.loads(Path(args.protocol).read_bytes())
    assert sha(args.protocol) == plan["toolkit_protocol_sha256"]
    assert all(sha(toolkit / name.replace("\\", "/")) == digest
               for name, digest in protocol["toolkit_code_sha256"].items())
    sys.path.insert(0, str(toolkit))
    from rgbt import LasHeR, RGBT210, RGBT234
    dataset = {"LasHeR": LasHeR, "RGBT210": RGBT210, "RGBT234": RGBT234}[benchmark]()
    assert set(dataset.seqs_name) == set(names)
    if benchmark != "LasHeR":
        assert all(sha(toolkit / name.replace("\\", "/")) == digest
                   for name, digest in manifest["annotation_file_sha256"].items())
    proposed, blocked = 0, 0
    for sequence in manifest["sequences"]:
        name = sequence["name"]
        visible, infrared = groundtruth(sequence, benchmark)
        prediction = np.loadtxt(run / "predictions" / (name + ".txt"),
                                delimiter=",", dtype=np.float32, ndmin=2)
        assert prediction.shape == visible.shape
        assert np.isfinite(prediction).all() and (prediction[:, 2:] > 0).all()
        if benchmark == "LasHeR":
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
        with (run / "traces" / (name + ".csv")).open() as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row["frame"]) for row in rows] == list(range(1, len(visible)))
        for row in rows:
            write, reject = int(row["template_write_proposed"]), int(row["template_write_blocked"])
            assert write == int(int(row["frame"]) % 5 == 0 and float(row["confidence"]) >= .65)
            if plan["mode"] == "control":
                assert int(row["m1_state"]) == -1 and reject == 0
            else:
                probabilities = np.array([float(row[key]) for key in (
                    "m1_reliable", "m1_uncertain", "m1_lost")])
                assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
                assert np.isclose(probabilities.sum(), 1, atol=1e-6)
                assert int(row["m1_state"]) == int(probabilities.argmax())
                assert reject == int(write and int(row["m1_state"]) != 0)
            proposed += write
            blocked += reject
    assert proposed == complete["template_write_proposed"] and blocked == complete["template_write_blocked"]
    timings = np.load(run / "frame_seconds.npy")
    assert timings.shape == (manifest["frame_count"],)
    assert np.isfinite(timings).all() and (timings > 0).all()
    assert np.isclose(timings.sum(), complete["seconds"], rtol=1e-12)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = official_metrics(toolkit, benchmark, "test", [run], [config], output)
    result.update(status="SCORED_COMPLETE", stage=config["stage"], mode=plan["mode"],
                  benchmark=benchmark, split="test", sequences=expected, frames=manifest["frame_count"],
                  exploratory_test=True, test_eligible=False,
                  checkpoint_sha256=plan["checkpoint_sha256"], m1_sha256=plan["m1_sha256"],
                  manifest_sha256=plan["manifest_sha256"], plan_sha256=config["plan_sha256"],
                  source_sha256=plan["source_sha256"], source_review_sha256=config["source_review_sha256"],
                  total_parameters=config["total_parameters"], fps=len(timings) / float(timings.sum()),
                  mean_ms=1000 * float(timings.mean()), p95_ms=1000 * float(np.quantile(timings, .95)),
                  timing_scope=config["timing_scope"], template_write_proposed=proposed,
                  template_write_blocked=blocked, scoring_script_sha256=sha(__file__),
                  toolkit_protocol_sha256=sha(args.protocol), curves_sha256=sha(output.with_suffix(".npz")),
                  native_status_sha256=sha(config["native_status"]),
                  artifacts_sha256={p.relative_to(run).as_posix(): sha(p)
                                    for p in run.rglob("*") if p.is_file()})
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("per_sequence", "artifacts_sha256")}), flush=True)


if __name__ == "__main__":
    main()
