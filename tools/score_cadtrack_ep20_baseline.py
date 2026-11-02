"""Score only completed original-CADTrack ep20 full benchmark runs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tools")]
from benchmark_inputs import groundtruth, load_manifest, sha
from joint_candidate_results import official_metrics


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "output", "toolkit", "protocol"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    assert not output.exists() and not output.with_suffix(".npz").exists()
    config = json.loads((run / "config.json").read_bytes())
    receipt = json.loads((run / "COMPLETE.json").read_bytes())
    native = json.loads(Path(config["native_status"]).read_bytes())
    assert config["stage"] == receipt["stage"] == "CADTrack_ep20_original_tracker_evaluation"
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["command"] == config["native_command"]
    assert native["physical_gpu"] == config["physical_gpu"]
    assert native["task"] == config["task"] == receipt["task"]
    assert receipt["config_sha256"] == sha(run / "config.json")
    assert receipt["optimizer_updates"] == receipt["backward_calls"] == 0
    assert receipt["state_unchanged"] and not receipt["smoke_only"]
    assert config["smoke_frames"] == 0 and config["split"] == "test"
    assert config["exploratory_test"] is receipt["exploratory_test"] is True
    assert config["test_eligible"] is receipt["test_eligible"] is False
    assert sha(config["checkpoint"]) == config["checkpoint_sha256"] == receipt["checkpoint_sha256"]
    assert sha(config["manifest"]) == config["manifest_sha256"]
    assert sha(ROOT / config["source_review"]) == config["source_review_sha256"]
    review = json.loads((ROOT / config["source_review"]).read_bytes())
    assert review["stage"] == config["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["source_sha256"] == config["source_sha256"]
    assert all(sha(ROOT / n) == h for n, h in config["source_sha256"].items())
    manifest = load_manifest(config["manifest"])
    assert manifest["benchmark"] == config["benchmark"]
    assert config["sequences"] == [s["name"] for s in manifest["sequences"]]
    assert receipt["sequences"] == config["split_sequences"] == manifest["sequence_count"]
    assert receipt["frames"] == config["split_frames"] == manifest["frame_count"]
    toolkit = Path(args.toolkit)
    protocol = json.loads(Path(args.protocol).read_bytes())
    assert all(sha(toolkit / n.replace("\\", "/")) == h
               for n, h in protocol["toolkit_code_sha256"].items())
    sys.path.insert(0, str(toolkit))
    from rgbt import LasHeR, RGBT210, RGBT234
    dataset = {"LasHeR": LasHeR, "RGBT210": RGBT210, "RGBT234": RGBT234}[config["benchmark"]]()
    assert set(dataset.seqs_name) == set(config["sequences"])
    assert {p.stem for p in (run / "predictions").glob("*.txt")} == set(config["sequences"])
    if config["benchmark"] != "LasHeR":
        assert all(sha(toolkit / n.replace("\\", "/")) == h
                   for n, h in manifest["annotation_file_sha256"].items())
    for sequence in manifest["sequences"]:
        visible, infrared = groundtruth(sequence, config["benchmark"])
        name = sequence["name"]
        prediction = np.loadtxt(run / "predictions" / (name + ".txt"),
                                delimiter=",", dtype=np.float32, ndmin=2)
        assert prediction.shape == visible.shape and np.isfinite(prediction).all()
        assert (prediction[:, 2:] > 0).all()
        if config["benchmark"] == "LasHeR":
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
    timing = np.load(run / "frame_seconds.npy")
    assert timing.shape == (manifest["frame_count"],)
    assert np.isfinite(timing).all() and (timing > 0).all()
    assert np.isclose(timing.sum(), receipt["seconds"], rtol=1e-12)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = official_metrics(toolkit, config["benchmark"], "test", [run], [config], output)
    result.update(status="SCORED_COMPLETE", stage=config["stage"],
                  benchmark=config["benchmark"], split="test",
                  sequences=receipt["sequences"], frames=receipt["frames"],
                  checkpoint_sha256=config["checkpoint_sha256"],
                  source_sha256=config["source_sha256"],
                  source_review_sha256=config["source_review_sha256"],
                  total_parameters=config["total_parameters"],
                  fps=len(timing) / float(timing.sum()), mean_ms=1000 * float(timing.mean()),
                  p95_ms=1000 * float(np.quantile(timing, .95)), timing_scope=config["timing_scope"],
                  native_status_sha256=sha(config["native_status"]),
                  curves_sha256=sha(output.with_suffix(".npz")),
                  exploratory_test=True, test_eligible=False,
                  scoring_script_sha256=sha(__file__),
                  toolkit_protocol_sha256=sha(args.protocol))
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sequence"}), flush=True)


if __name__ == "__main__":
    main()
