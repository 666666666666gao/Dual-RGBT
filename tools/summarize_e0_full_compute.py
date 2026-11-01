"""Attach trace-weighted MACs to an already scored complete E0 benchmark."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def trace_counts(path, frames, routes):
    counts = {key: 0 for key in routes}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        last = 0
        for expected, row in enumerate(reader, 1):
            assert int(row["frame"]) == expected and expected < frames
            key = int(row["modality"]), int(row["action"])
            cost = routes[key]
            assert int(row["primary_tokens"]) == cost["primary_tokens"]
            assert int(row["auxiliary_tokens"]) == cost["auxiliary_tokens"]
            counts[key] += 1
            last = expected
    assert last == frames - 1
    return counts


def bind_sources(profile, config):
    common = set(profile["source_sha256"]) & set(config["source_sha256"])
    assert {"dual_rgbt/model.py", "dual_rgbt/tracker.py",
            "lib/models/cadtrack/vit_care.py",
            "lib/models/cadtrack/cadtrack.py",
            "lib/config/cadtrack/config.py",
            "experiments/cadtrack/cadtrack.yaml"} <= common
    assert all(profile["source_sha256"][name] == config["source_sha256"][name]
               for name in common), "Profile source differs from evaluated source."
    assert config["input_sizes"] == dict(search=256, template=128, templates=2)
    return sorted(common)


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "run", "score", "score-sha256", "profile",
                 "profile-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root, run, output = Path(args.root), Path(args.run), Path(args.output)
    assert not output.exists()
    assert sha(args.score) == args.score_sha256
    assert sha(args.profile) == args.profile_sha256
    score = json.loads(Path(args.score).read_bytes())
    profile = json.loads(Path(args.profile).read_bytes())
    assert score["status"] == "SCORED_COMPLETE"
    assert profile["status"] == "CPU_ROUTE_MAC_CROSSCHECK_COMPLETE"
    assert profile["model_unchanged"] and profile["cuda_initialized"] is False
    assert all(row["independent_formula_match"] for row in profile["routes"])
    config = json.loads((run / "config.json").read_bytes())
    complete = json.loads((run / "COMPLETE.json").read_bytes())
    native = json.loads(Path(config["native_status"]).read_bytes())
    assert config["stage"] in ("E0_finetuned_evaluation",
                               "E0_M1_template_guard_evaluation")
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == config["task"] == complete["task"]
    assert native["command"] == config["native_command"]
    assert native["physical_gpu"] == config["physical_gpu"]
    assert score["native_status_sha256"] == sha(config["native_status"])
    assert complete["config_sha256"] == sha(run / "config.json")
    assert complete["state_unchanged"] and not complete["smoke_only"]
    assert complete["optimizer_updates"] == complete["backward_calls"] == 0
    assert config["smoke_frames"] == 0 and config["split"] == score["split"] == "test"
    assert config["test_eligible"] is score["test_eligible"] is False
    assert config["exploratory_test"] is score["exploratory_test"] is True
    assert score["checkpoint_sha256"] == config["checkpoint_sha256"] == complete["checkpoint_sha256"]
    assert score["source_sha256"] == config["source_sha256"]
    common = bind_sources(profile, config)
    assert all(sha(root / name) == digest
               for name, digest in config["source_sha256"].items())
    assert score["artifacts_sha256"] == {
        path.relative_to(run).as_posix(): sha(path)
        for path in run.rglob("*") if path.is_file()}
    assert score["manifest_sha256"] == config["manifest_sha256"]
    manifest_path = root / config["manifest"]
    assert sha(manifest_path) == config["manifest_sha256"]
    manifest = json.loads(manifest_path.read_bytes())
    benchmark = config["benchmark"]
    assert benchmark == score["benchmark"] == manifest["benchmark"]
    expected = {"LasHeR": 245, "RGBT210": 210, "RGBT234": 234}[benchmark]
    assert complete["sequences"] == score["sequences"] == manifest["sequence_count"] == expected
    frames = manifest["frame_count"]
    assert complete["frames"] == score["frames"] == frames
    assert config["sequences"] == [row["name"] for row in manifest["sequences"]]
    assert {p.stem for p in (run / "traces").glob("*.csv")} == set(config["sequences"])
    routes = {(row["modality"], row["action"]): row for row in profile["routes"]}
    assert set(routes) == {(modality, action) for modality in range(2) for action in range(5)}
    counts = {key: 0 for key in routes}
    for sequence in manifest["sequences"]:
        selected = trace_counts(run / "traces" / (sequence["name"] + ".csv"),
                                len(sequence["rgb"]), routes)
        for key, value in selected.items():
            counts[key] += value
    tracked = sum(counts.values())
    assert tracked == frames - expected
    e0_macs = sum(routes[key]["macs"] * count for key, count in counts.items())
    m1_enabled = (config["stage"] == "E0_M1_template_guard_evaluation"
                  and config["mode"] == "guard")
    m1_macs = profile["m1_head"]["macs"] * tracked if m1_enabled else 0
    parameters = profile["parameter_counts"]["e0"]
    if m1_enabled:
        parameters += profile["parameter_counts"]["m1"]
    assert parameters == config["total_parameters"] == score["total_parameters"]
    timing = np.load(run / "frame_seconds.npy", allow_pickle=False)
    assert timing.shape == (frames,) and np.isfinite(timing).all() and (timing > 0).all()
    fps = frames / float(timing.sum())
    assert np.isclose(score["fps"], fps, rtol=1e-12)
    result = dict(status="COMPLETE_FULL_RUN_COMPUTE_SUMMARY", task=config["task"],
        benchmark=benchmark, stage=config["stage"], frames=frames, tracked_frames=tracked,
        initialization_frames=expected, total_parameters=parameters,
        fps=score["fps"], mean_ms=score["mean_ms"], p95_ms=score["p95_ms"],
        timing_scope=score["timing_scope"],
        route_counts=[dict(modality=key[0], action=key[1], frames=count,
                           macs_per_frame=routes[key]["macs"])
                      for key, count in sorted(counts.items())],
        e0_tracked_frame_macs=e0_macs, m1_head_tracked_frame_macs=m1_macs,
        mean_tracked_frame_gmacs=(e0_macs + m1_macs) / tracked / 1e9,
        run=str(run), score=str(args.score), score_sha256=args.score_sha256,
        profile=str(args.profile), profile_sha256=args.profile_sha256,
        common_source_paths=common, script_sha256=sha(__file__),
        native_status_sha256=score["native_status_sha256"],
        complete_sha256=sha(run / "COMPLETE.json"),
        exploratory_test=True, test_eligible=False,
        scientific_result_audit_accepted=False,
        scope="Actual full-run action counts weighted by matching-source CPU-executed "
              "convolution/dense/attention MACs. Excludes initialization arithmetic, "
              "nonlinear/normalization/ROI/decode/memory/bookkeeping operations. "
              "M1 guard adds only head matrix MACs. FPS is the existing full-run "
              "measurement including initialization under concurrent load, not "
              "isolated throughput. Official accuracy score is referenced unchanged; "
              "no scorer rerun, threshold selection or benchmark inference.")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
