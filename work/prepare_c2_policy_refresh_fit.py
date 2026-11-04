"""Prepare a disabled C2 FIT draft from a completed matching replay."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from work.launch_c2_policy_refresh_acceptance import collection_plan_binding

ENTRY = "scripts/train_c2_policy_refresh.py"
VERIFIER = "tools/verify_c2_policy_refresh_collection.py"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(acceptance_raw, native_raw, *, acceptance_path, native_path,
            root, run_root, task, gpu, python, review_path):
    accepted = json.loads(acceptance_raw)
    native = json.loads(native_raw)
    assert accepted["stage"] == "P2_C2_policy_refresh_full_collection_acceptance"
    assert accepted["status"] == "PASS"
    assert accepted["host"] in ("2025", "2026")
    assert accepted["variant"] in ("zero", "film")
    assert (accepted["sequences"], accepted["tracked_frames"],
            accepted["records"]) == (882, 458553, 91091)
    assert accepted["benchmark_evidence"] is False
    assert accepted["policy_fit_admitted"] is False
    assert accepted["cuda_initialized"] is True
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["physical_gpu"] in ("0", "1", "2", "3")
    replay_root = PurePosixPath(accepted["root"])
    assert native_path == str(
        replay_root / "runs/pipeline_tasks" / native["task"] / "status.json")
    assert accepted["verifier_sha256"] == accepted["sources"][VERIFIER]
    collections = accepted["collections"]
    evidence = accepted["native_and_artifact_evidence"]
    assert len(collections) == len(evidence) == 4
    paths = [row["plan"]["command"][-1] for row in collections]
    assert len(set(paths)) == 4
    assert native["command"] == [
        python, "-B", str(replay_root / VERIFIER), "--plans",
        *paths, "--output", acceptance_path,
    ]
    first = collections[0]["plan"]
    totals = dict(sequences=0, tracked_frames=0, records=0)
    for shard, (collection, witness) in enumerate(zip(collections, evidence)):
        plan = collection["plan"]
        assert plan["shard"] == shard and plan["shards"] == 4
        assert plan["smoke"] is False
        for key in ("host", "root", "variant", "c2_task_id",
                    "executor_binding", "sources"):
            assert plan[key] == accepted[key], key
        for key in ("seed", "cost_weight", "label_stride"):
            assert plan[key] == first[key], key
        assert witness["plan_path"] == paths[shard]
        assert witness["plan_sha256"] == collection["plan_sha256"]
        for key in totals:
            totals[key] += witness[key]
    assert totals == {key: accepted[key] for key in totals}
    binding = accepted["executor_binding"]
    for key in ("host", "variant", "c2_task_id"):
        assert binding[key] == accepted[key], key
    assert binding["stage"] == "P2_C2_policy_refresh"
    assert task and PurePosixPath(task).name == task
    assert gpu in ("0", "1", "2", "3")
    for path in (root, run_root, acceptance_path, native_path, python,
                 review_path):
        assert PurePosixPath(path).is_absolute()
    assert root != accepted["root"]
    PurePosixPath(review_path).relative_to(PurePosixPath(root))

    sources = dict(accepted["sources"])
    sources[ENTRY] = sha256((ROOT / ENTRY).read_bytes())
    plan_path = str(PurePosixPath(root) / "reports" / task / "plan.json")
    plan = dict(
        stage="P2_C2_policy_refresh_fit", mode="fit",
        host=accepted["host"], variant=accepted["variant"],
        c2_task_id=accepted["c2_task_id"],
        root=root, run_root=run_root,
        run=str(PurePosixPath(run_root) / task), task=task, gpu=gpu,
        launch_allowed=False, smoke=False,
        seed=first["seed"], label_stride=first["label_stride"],
        cost_weight=first["cost_weight"],
        epochs=5, batch_size=512, workers=4, preview_lr=4e-5,
        quality_gain_lr=4e-4, weight_decay=1e-4, gradient_clip=1.0,
        test_enabled=False, dev_enabled=False,
        executor_binding=binding, sources=sources,
        collection_acceptance=dict(
            path=acceptance_path, sha256=sha256(acceptance_raw)),
        collection_acceptance_native=dict(
            path=native_path, sha256=sha256(native_raw)),
        review=dict(path=review_path, sha256="PENDING"),
        command=[python, "-B", str(PurePosixPath(root) / ENTRY),
                 "--plan", plan_path],
    )
    return plan, dict(
        stage=plan["stage"], status="PREPARED_DISABLED",
        plan_path=plan_path, task=task,
        admitted_plan_binding=collection_plan_binding(plan),
        sources=sources,
        collection_acceptance=plan["collection_acceptance"],
        collection_acceptance_native=plan["collection_acceptance_native"],
        executor_binding=binding,
        runtime_admission_required=True, launch_allowed=False,
        benchmark_evidence=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", required=True)
    parser.add_argument("--native", required=True)
    for name in ("acceptance-path", "native-path", "root", "run-root", "task",
                 "gpu", "python", "review-path", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    plan, binding = prepare(
        Path(args.acceptance).read_bytes(), Path(args.native).read_bytes(),
        **{key: getattr(args, key) for key in (
            "acceptance_path", "native_path", "root", "run_root", "task",
            "gpu", "python", "review_path")})
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (folder / "binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    print(json.dumps(binding, indent=2))


if __name__ == "__main__":
    main()
