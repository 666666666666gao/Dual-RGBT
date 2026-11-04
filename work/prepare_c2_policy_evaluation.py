"""Prepare disabled LasHeR evaluation plans from archived terminal C2 FIT evidence."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ENTRY = "scripts/eval_c2_policy_completed.py"
FIT_NAMES = {
    "plan", "config", "native", "complete", "precision", "review",
    "acceptance", "replay_native",
}
NEW_SOURCES = (
    ENTRY, "dual_rgbt/c2_policy_evaluation.py", "scripts/benchmark_inputs.py",
    "tools/score_c2_policy_completed.py", "work/score_c2_policy_completed_remote.py",
)
CPU = {
    "2025": "/home/gaob/Dual-RGBT/conda-envs/cpu-scoring/bin/python",
    "2026": "/data2/gb/Dual-RGBT/conda-envs/cpu-scoring/bin/python",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def binding(plan):
    return sha(json.dumps(
        {key: value for key, value in plan.items()
         if key not in ("launch_allowed", "review")},
        sort_keys=True, separators=(",", ":")).encode())


def prepare(documents, evidence, *, root, run_root, python, gpu, task_prefix,
            review_path, manifests, protocol, toolkit):
    """Check metadata readiness; actual weights/source reconstruction stays mandatory."""
    assert set(documents) == set(evidence) == FIT_NAMES
    for name, raw in documents.items():
        assert evidence[name]["sha256"] == sha(raw), name
        assert PurePosixPath(evidence[name]["path"]).is_absolute(), name
    docs = {name: json.loads(raw) for name, raw in documents.items()}
    fit, complete, native = docs["plan"], docs["complete"], docs["native"]
    assert fit == docs["config"]
    assert fit["stage"] == complete["stage"] == "P2_C2_policy_refresh_fit"
    assert fit["mode"] == "fit" and fit["launch_allowed"] is True
    assert fit["smoke"] is complete["smoke"] is False
    assert fit["test_enabled"] is fit["dev_enabled"] is False
    assert fit["host"] in CPU and fit["variant"] in ("zero", "film")
    assert (fit["epochs"], fit["batch_size"], fit["workers"]) == (5, 512, 4)
    assert (fit["seed"], fit["label_stride"], fit["cost_weight"]) == (2027, 5, .05)
    assert native["state"] == complete["status"] == "COMPLETE"
    assert native["exit_code"] == 0
    assert native["task"] == fit["task"] and native["physical_gpu"] == fit["gpu"]
    assert native["command"] == fit["command"]
    assert (complete["epochs"], complete["steps"], complete["records"]) == (5, 890, 91091)
    assert complete["retained_epochs"] == [1, 2, 3, 4, 5]
    assert set(complete["artifacts"]) == {"epoch_%03d.pth" % n for n in range(1, 6)}
    assert complete["plan_sha256"] == evidence["plan"]["sha256"]
    assert complete["executor_binding"] == fit["executor_binding"]
    for key in ("executor_preserved", "all_policy_modules_changed", "strict_reload"):
        assert complete[key] is True, key
    assert complete["benchmark_evidence"] is False
    assert docs["precision"] == dict(
        matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)
    assert fit["review"] == evidence["review"]
    assert fit["collection_acceptance"] == evidence["acceptance"]
    assert fit["collection_acceptance_native"] == evidence["replay_native"]
    review = docs["review"]
    assert review["stage"] == fit["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == fit["sources"]
    assert binding(fit) in review["admitted_plan_bindings"]
    accepted = docs["acceptance"]
    assert accepted["stage"] == "P2_C2_policy_refresh_full_collection_acceptance"
    assert accepted["status"] == "PASS"
    assert (accepted["sequences"], accepted["tracked_frames"], accepted["records"]) == (
        882, 458553, 91091)
    for key in ("host", "variant", "c2_task_id", "executor_binding"):
        assert accepted[key] == fit[key], key
    assert docs["replay_native"]["state"] == "COMPLETE"
    assert docs["replay_native"]["exit_code"] == 0
    for path in (root, run_root, python, review_path, toolkit, protocol["path"]):
        assert PurePosixPath(path).is_absolute()
    assert root != fit["root"]
    PurePosixPath(review_path).relative_to(root)
    assert gpu in ("0", "1", "2", "3")
    assert task_prefix and PurePosixPath(task_prefix).name == task_prefix
    assert set(manifests) == {"smoke", "lasher"}
    for mode, item in manifests.items():
        manifest = json.loads(item["raw"])
        assert manifest["benchmark"] == "LasHeR"
        assert manifest["split"] == ("train_smoke" if mode == "smoke" else "test")
        assert PurePosixPath(item["path"]).is_absolute()
        if mode == "smoke":
            assert len(manifest["sequences"][0]["rgb"]) >= 11
        else:
            assert (manifest["sequence_count"], manifest["frame_count"]) == (245, 220703)
    sources = dict(fit["sources"])
    # Preserve host-specific dependency pins; add only the separate evaluation entry.
    for name in NEW_SOURCES:
        digest = sha((ROOT / name).read_bytes())
        if name in sources:
            assert sources[name] == digest, name
        sources[name] = digest
    assert {"dual_rgbt/tracker.py", "dual_rgbt/c2_executor.py",
            "dual_rgbt/c2_policy_refresh.py", "dual_rgbt/model.py",
            "dual_rgbt/data.py", "scripts/run_task_four_gpu.py"}.issubset(sources)
    plans = {}
    for mode, item in manifests.items():
        task = task_prefix + "_" + mode
        plan_path = str(PurePosixPath(root) / "reports" / task / "plan.json")
        smoke = mode == "smoke"
        plans[mode] = dict(
            stage="P2_C2_policy_evaluation", host=fit["host"],
            variant=fit["variant"], c2_task_id=fit["c2_task_id"],
            root=root, run_root=run_root, run=str(PurePosixPath(run_root) / task),
            task=task, gpu=gpu, plan_path=plan_path, launch_allowed=False,
            smoke=smoke, smoke_frames=11 if smoke else 0,
            cost_weight=.05, exploratory_test=not smoke, test_eligible=False,
            dev_enabled=False, benchmark="LasHeR", fit_evidence=evidence,
            checkpoint=dict(path=str(PurePosixPath(fit["run"]) / "epoch_005.pth"),
                            sha256=complete["artifacts"]["epoch_005.pth"]),
            sources=sources, review=dict(path=review_path, sha256="PENDING"),
            manifest=dict(path=item["path"], sha256=sha(item["raw"])),
            protocol=protocol, toolkit=toolkit, scoring_interpreter=CPU[fit["host"]],
            score_output=str(PurePosixPath(root) / "reports" / task / "score.json"),
            command=[python, "-B", str(PurePosixPath(root) / ENTRY),
                     "--plan", plan_path],
        )
    return plans, dict(
        stage="P2_C2_policy_evaluation", status="PREPARED_DISABLED",
        admitted_plan_bindings=[binding(plan) for plan in plans.values()],
        sources=sources, fit_evidence=evidence,
        actual_cpu_reconstruction_required=True,
        own_train_smoke_required=True, runtime_admission_required=True,
        checkpoint_hash_verification_required=True,
        benchmark_evidence=False, launch_allowed=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True,
                        help="JSON map of eight names to local archived file and remote path")
    for name in ("root", "run-root", "python", "gpu", "task-prefix", "review-path",
                 "toolkit", "protocol", "protocol-path", "smoke-manifest",
                 "smoke-manifest-path", "lasher-manifest", "lasher-manifest-path",
                 "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    items = json.loads(Path(args.evidence).read_bytes())
    documents = {name: Path(item["local"]).read_bytes() for name, item in items.items()}
    evidence = {name: dict(path=item["path"], sha256=sha(documents[name]))
                for name, item in items.items()}
    plans, receipt = prepare(
        documents, evidence,
        **{key: getattr(args, key) for key in (
            "root", "run_root", "python", "gpu", "task_prefix", "review_path", "toolkit")},
        manifests={
            mode: dict(path=getattr(args, mode + "_manifest_path"),
                       raw=Path(getattr(args, mode + "_manifest")).read_bytes())
            for mode in ("smoke", "lasher")},
        protocol=dict(path=args.protocol_path, sha256=sha(Path(args.protocol).read_bytes())))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in {**plans, "binding": receipt}.items():
        (output / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
