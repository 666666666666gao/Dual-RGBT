"""Validate complete E0 predictions and invoke the unchanged RGBT toolkit."""
import argparse
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
    parser.add_argument("--run", required=True)
    parser.add_argument("--toolkit", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    output = Path(args.output)
    assert not output.exists() and not output.with_suffix(".npz").exists()
    config = json.loads((run / "config.json").read_bytes())
    complete = json.loads((run / "COMPLETE.json").read_bytes())
    native = json.loads(Path(config["native_status"]).read_bytes())
    assert config["stage"] == "E0_finetuned_evaluation"
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == config["task"] == complete["task"]
    assert native["command"] == config["native_command"]
    assert native["physical_gpu"] == config["physical_gpu"]
    assert complete["config_sha256"] == sha(run / "config.json")
    assert complete["optimizer_updates"] == complete["backward_calls"] == 0
    assert complete["state_unchanged"] and not complete["smoke_only"]
    assert config["smoke_frames"] == 0 and config["split"] == "test"
    assert config["exploratory_test"] is complete["exploratory_test"] is True
    assert config["test_eligible"] is complete["test_eligible"] is False
    assert sha(config["manifest"]) == config["manifest_sha256"]
    assert sha(config["checkpoint"]) == config["checkpoint_sha256"] == complete["checkpoint_sha256"]
    assert sha(config["factory"]) == config["factory_sha256"]
    assert sha(config["source_review"]) == config["source_review_sha256"]
    review = json.loads(Path(config["source_review"]).read_bytes())
    assert review["stage"] == "E0_finetuned_evaluation"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert config["source_sha256"] == review["source_sha256"]
    assert all(sha(ROOT / name) == digest for name, digest in config["source_sha256"].items())
    manifest = load_manifest(config["manifest"])
    benchmark = manifest["benchmark"]
    assert config["benchmark"] == benchmark
    assert config["sequences"] == [row["name"] for row in manifest["sequences"]]
    assert {path.stem for path in (run / "predictions").glob("*.txt")} == set(config["sequences"])
    expected = {"LasHeR": 245, "RGBT210": 210, "RGBT234": 234}[benchmark]
    assert complete["sequences"] == config["split_sequences"] == manifest["sequence_count"] == expected
    assert complete["frames"] == config["split_frames"] == manifest["frame_count"]
    toolkit = Path(args.toolkit)
    protocol = json.loads(Path(args.protocol).read_bytes())
    assert all(sha(toolkit / name.replace("\\", "/")) == digest
               for name, digest in protocol["toolkit_code_sha256"].items())
    sys.path.insert(0, str(toolkit))
    from rgbt import LasHeR, RGBT210, RGBT234
    dataset = {"LasHeR": LasHeR, "RGBT210": RGBT210, "RGBT234": RGBT234}[benchmark]()
    assert set(dataset.seqs_name) == set(config["sequences"])
    if benchmark != "LasHeR":
        assert all(sha(toolkit / name.replace("\\", "/")) == digest
                   for name, digest in manifest["annotation_file_sha256"].items())
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
    timings = np.load(run / "frame_seconds.npy")
    assert timings.shape == (manifest["frame_count"],)
    assert np.isfinite(timings).all() and (timings > 0).all()
    assert np.isclose(timings.sum(), complete["seconds"], rtol=1e-12)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = official_metrics(toolkit, benchmark, "test", [run], [config], output)
    result.update(status="SCORED_COMPLETE", benchmark=benchmark, split="test",
                  sequences=expected, frames=manifest["frame_count"],
                  exploratory_test=True, test_eligible=False,
                  checkpoint_sha256=config["checkpoint_sha256"],
                  manifest_sha256=config["manifest_sha256"],
                  source_sha256=config["source_sha256"],
                  source_review_sha256=config["source_review_sha256"],
                  total_parameters=config["total_parameters"],
                  mean_ms=1000 * float(timings.mean()),
                  p95_ms=1000 * float(np.quantile(timings, 0.95)),
                  fps=len(timings) / float(timings.sum()),
                  timing_scope=config["timing_scope"],
                  toolkit_code_sha256=protocol["toolkit_code_sha256"],
                  scoring_script_sha256=sha(__file__),
                  metric_implementation_sha256={name: sha(ROOT / name) for name in (
                      "tools/joint_candidate_results.py", "scripts/score_tracking.py",
                      "scripts/joint_evaluation_contract.py", "scripts/benchmark_inputs.py")},
                  toolkit_protocol_sha256=sha(args.protocol),
                  curves_sha256=sha(output.with_suffix(".npz")),
                  native_status_sha256=sha(config["native_status"]),
                  artifacts_sha256={path.relative_to(run).as_posix(): sha(path)
                                    for path in run.rglob("*") if path.is_file()})
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("per_sequence", "artifacts_sha256")}), flush=True)


if __name__ == "__main__":
    main()
