"""Attach measured component MACs to one already scored complete M2 TEST run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_counts(path, frames, routes):
    counts = {key: 0 for key in routes}
    invocations, selections, last = 0, 0, 0
    with Path(path).open(newline="") as handle:
        for expected, row in enumerate(csv.DictReader(handle), 1):
            assert int(row["frame"]) == expected and expected < frames
            key = int(row["modality"]), int(row["action"])
            cost = routes[key]
            assert int(row["primary_tokens"]) == cost["primary_tokens"]
            assert int(row["auxiliary_tokens"]) == cost["auxiliary_tokens"]
            invoked = int(row["m2_invoked"])
            # Full TEST uses every frame as valid; sampling starts at frame 1.
            assert invoked == int((expected - 1) % 25 == 0)
            selected = int(row["m2_class_index"])
            assert (0 <= selected <= 24) if invoked else selected == -1
            invocations += invoked
            selections += int(selected > 0)
            counts[key] += 1
            last = expected
    assert last == frames - 1
    return counts, invocations, selections


def bind_profiles(route, candidate, plan):
    assert route["status"] == "CPU_ROUTE_MAC_CROSSCHECK_COMPLETE"
    assert candidate["status"] == "CPU_M2_PRIMARY_MAC_CROSSCHECK_COMPLETE"
    assert route["model_unchanged"] and route["cuda_initialized"] is False
    assert candidate["model_unchanged"] and candidate["cuda_initialized"] is False
    assert candidate["source_sha256"] == plan["source_sha256"]
    assert candidate["input_sha256"] == plan["input_sha256"]
    assert candidate["parent_checkpoint_sha256"] == plan["parent_checkpoint_sha256"]
    common = set(route["source_sha256"]) & set(plan["source_sha256"])
    required = {"dual_rgbt/model.py", "dual_rgbt/tracker.py", "dual_rgbt/e0_identity.py",
                "lib/models/cadtrack/vit_care.py", "lib/models/cadtrack/cadtrack.py",
                "lib/models/layers/head.py", "lib/models/layers/patch_embed.py",
                "lib/config/cadtrack/config.py", "experiments/cadtrack/cadtrack.yaml"}
    assert required <= common
    mismatches = sorted(name for name in common
                        if route["source_sha256"][name] != plan["source_sha256"][name])
    # The observed old-profile/R15 difference is the process wrapper, not model math.
    assert set(mismatches) <= {"scripts/run_task_four_gpu.py"}
    shape = dict(templates=[1, 2, 6, 128, 128], search=[1, 6, 256, 256])
    assert all(route["inputs"][key] == value for key, value in shape.items())
    assert candidate["inputs"] == shape
    assert all(row["independent_formula_match"] for row in route["routes"])
    assert route["m2_initial_anchor"]["independent_formula_match"]
    primary = candidate["primary"]
    assert len(primary) == 2 and {row["modality"] for row in primary} == {0, 1}
    assert all(row["independent_formula_match"] for row in primary)
    assert candidate["queries_per_invocation"] == 12
    assert candidate["primary_forwards_per_invocation"] == 24
    assert candidate["candidate_primary_macs_per_invocation"] == (
        12 * sum(row["macs"] for row in primary))
    return sorted(common), mismatches


def compute_totals(route, candidate, counts, invocations, sequences):
    routes = {(row["modality"], row["action"]): row for row in route["routes"]}
    assert set(routes) == {(m, a) for m in range(2) for a in range(5)}
    assert all(value >= 0 for value in counts.values())
    head = [row for row in route["m2_identity_head"] if row["candidates"] == 24]
    assert len(head) == 1 and head[0]["independent_formula_match"]
    base = sum(routes[key]["macs"] * count for key, count in counts.items())
    primary = invocations * candidate["candidate_primary_macs_per_invocation"]
    identity = invocations * head[0]["macs"]
    anchor = sequences * route["m2_initial_anchor"]["macs"]
    return dict(e0_tracked_frame_macs=base,
                m2_candidate_primary_macs=primary,
                m2_identity_head_macs=identity,
                m2_initial_anchor_macs=anchor,
                tracked_frame_macs=base + primary + identity,
                all_frame_macs=base + primary + identity + anchor)


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "plan", "plan-sha256", "score", "score-sha256",
                 "route-profile", "route-profile-sha256",
                 "candidate-profile", "candidate-profile-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    loaded = {}
    for key in ("plan", "score", "route_profile", "candidate_profile"):
        path = root / getattr(args, key)
        assert sha(path) == getattr(args, key + "_sha256")
        loaded[key] = json.loads(path.read_bytes())
    plan, score = loaded["plan"], loaded["score"]
    route, candidate = loaded["route_profile"], loaded["candidate_profile"]
    run = root / plan["run"]
    output = root / args.output
    assert not output.exists() and run.resolve() not in output.resolve().parents
    assert plan["target_root"] == root.as_posix()
    assert plan["stage"] == "E0_M2_online_integration"
    assert plan["phase"] == "full" and plan["split"] == "test"
    assert plan["launch_allowed"] is True
    assert plan["exploratory_test"] is True and plan["test_eligible"] is False
    assert Path(args.score) == Path(plan["score_output"])
    assert score["variant"] == "E0+M2" and score["split"] == "test"
    assert score["exploratory_test"] is True and score["test_eligible"] is False
    assert score["training_allowed"] is False
    assert score["plan_sha256"] == args.plan_sha256
    assert all(sha(root / name) == value for name, value in plan["source_sha256"].items())
    assert all(sha(root / name) == value for name, value in plan["input_sha256"].items())
    config = json.loads((run / "config.json").read_bytes())
    complete = json.loads((run / "COMPLETE.json").read_bytes())
    native_path = root / "runs/pipeline_tasks" / plan["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == config["task"] == complete["task"] == plan["task"]
    assert native["command"] == config["native_command"] == plan["evaluator_argv"]
    assert native["physical_gpu"] == config["physical_gpu"] == plan["assigned_gpu"]
    assert native["hostname"] == config["native_hostname"] == plan["target_hostname"]
    assert config["plan_sha256"] == args.plan_sha256
    assert config["source_sha256"] == plan["source_sha256"]
    assert config["input_sha256"] == plan["input_sha256"]
    assert complete["state_unchanged"] is True and complete["smoke_only"] is False
    assert complete["optimizer_updates"] == complete["backward_calls"] == 0
    assert complete["exploratory_test"] is True and complete["test_eligible"] is False
    assert score["checkpoint_sha256"] == complete["parent_checkpoint_sha256"] == (
        config["parent_checkpoint_sha256"]) == plan["parent_checkpoint_sha256"]
    assert score["m2_checkpoint_sha256"] == complete["m2_checkpoint_sha256"] == (
        config["m2_checkpoint_sha256"]) == plan["m2_checkpoint_sha256"]
    artifacts = {path.relative_to(run).as_posix(): sha(path)
                 for path in run.rglob("*") if path.is_file() and path.name != "COMPLETE.json"}
    assert artifacts == complete["artifacts_sha256"] == score["run_artifacts_sha256"]
    manifest_path = root / plan["manifest"]
    assert sha(manifest_path) == score["manifest_sha256"]
    manifest = json.loads(manifest_path.read_bytes())
    benchmark = manifest["benchmark"]
    assert benchmark == plan["benchmark"] == score["benchmark"]
    assert manifest["split"] == "test"
    sequences = manifest["sequences"]
    names = [sequence["name"] for sequence in sequences]
    expected = {"LasHeR": 245, "RGBT210": 210, "RGBT234": 234}[benchmark]
    assert len(names) == len(set(names)) == manifest["sequence_count"] == expected
    assert complete["sequences"] == score["sequences"] == expected
    frames = sum(len(sequence["rgb"]) for sequence in sequences)
    assert frames == manifest["frame_count"] == complete["frames"] == score["frames"]
    assert config["sequences"] == complete["sequence_names"] == names
    assert {p.stem for p in (run / "traces").glob("*.csv")} == set(names)
    common, differences = bind_profiles(route, candidate, plan)
    routes = {(row["modality"], row["action"]): row for row in route["routes"]}
    assert len(routes) == len(route["routes"]) == 10
    counts = {key: 0 for key in routes}
    invocations, selections = 0, 0
    for sequence in sequences:
        selected, calls, matches = trace_counts(
            run / "traces" / (sequence["name"] + ".csv"), len(sequence["rgb"]), routes)
        for key, count in selected.items():
            counts[key] += count
        invocations += calls
        selections += matches
    assert sum(counts.values()) == frames - expected
    assert invocations == complete["m2_invocations"] == score["m2_invocations"]
    assert selections == complete["m2_candidate_selections"] == score["m2_candidate_selections"]
    parameters = route["parameter_counts"]["e0"] + route["parameter_counts"]["m2"]
    assert parameters == config["total_parameters"]

    import numpy as np
    timing = np.load(run / "frame_seconds.npy", allow_pickle=False)
    assert timing.shape == (frames,) and np.isfinite(timing).all() and (timing > 0).all()
    assert np.isclose(timing.sum(), complete["seconds"], rtol=1e-12)
    assert np.isclose(frames / timing.sum(), score["fps"], rtol=1e-12)
    assert np.isclose(1000 * timing.mean(), score["mean_ms"], rtol=1e-12)
    assert np.isclose(1000 * np.quantile(timing, .95), score["p95_ms"], rtol=1e-12)
    totals = compute_totals(route, candidate, counts, invocations, expected)
    result = dict(status="COMPLETE_FULL_M2_COMPUTE_SUMMARY",
                  task=plan["task"], benchmark=benchmark, frames=frames,
                  tracked_frames=frames - expected, initialization_frames=expected,
                  total_parameters=parameters, m2_invocations=invocations,
                  m2_candidate_selections=selections,
                  fps=score["fps"], mean_ms=score["mean_ms"], p95_ms=score["p95_ms"],
                  timing_scope=config["timing_scope"],
                  route_counts=[dict(modality=key[0], action=key[1], frames=count)
                                for key, count in sorted(counts.items())],
                  mean_tracked_frame_gmacs=totals["tracked_frame_macs"] / (frames - expected) / 1e9,
                  mean_all_frame_gmacs=totals["all_frame_macs"] / frames / 1e9,
                  common_profile_sources=common, profile_source_differences=differences,
                  plan_sha256=args.plan_sha256, score_sha256=args.score_sha256,
                  route_profile_sha256=args.route_profile_sha256,
                  candidate_profile_sha256=args.candidate_profile_sha256,
                  native_status_sha256=sha(native_path),
                  complete_sha256=sha(run / "COMPLETE.json"),
                  script_sha256=sha(__file__), exploratory_test=True, test_eligible=False,
                  scientific_result_audit_accepted=False, **totals,
                  scope="Complete actual traces weighted by CPU-executed, formula-checked "
                        "convolution/dense/attention MACs. Counts candidate generation on "
                        "all M2 calls including no-match, plus the identity head; per-sequence "
                        "anchor is included only in all-frame costs. Excludes bias adds, "
                        "normalization, nonlinearities, ROI/crops, box decoding, memory and "
                        "bookkeeping. FPS includes decode and initialization under concurrent "
                        "load, not isolated throughput. References the existing official "
                        "score unchanged; no scorer, model or tracker is executed.")
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
