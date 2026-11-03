"""Prepare a disabled five-epoch FIT draft from full collection acceptance."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(acceptance_path, target_acceptance_path, task, gpu, batch_size, workers):
    raw = Path(acceptance_path).read_bytes()
    accepted = json.loads(raw)
    assert accepted["status"] == "PASS"
    assert accepted["stage"] == "P1B_direct_E0_full_collection_acceptance"
    assert accepted["sequences"] == 882 and accepted["records"] > 0
    assert accepted["cuda_initialized"] is False
    assert accepted["full_fit_admitted"] is False
    assert accepted["benchmark_evidence"] is False
    assert gpu in ("0", "1", "2", "3") and batch_size > 0 and workers >= 0
    assert task and Path(task).name == task
    collections = accepted["collections"]
    assert len(collections) == len(accepted["native_and_artifact_evidence"]) == 4
    for total, field in (("sequences", "sequences"), ("records", "records"),
                         ("tracked_frames", "tracked_frames")):
        assert accepted[total] == sum(
            item[field] for item in accepted["native_and_artifact_evidence"])
    first = collections[0]["plan"]
    assert first["smoke"] is False and first["mode"] == "collect"
    assert first["host"] == accepted["host"] and first["root"] == accepted["root"]
    for key in ("sources", "inputs", "branch", "round", "label_stride"):
        assert first[key] == accepted[key], key
    assert [row["plan"]["shard"] for row in collections] == list(range(4))
    for collection, evidence in zip(collections, accepted["native_and_artifact_evidence"]):
        plan = collection["plan"]
        for key in ("host", "root", "sources", "inputs", "branch", "round",
                    "seed", "cost_weight", "label_stride"):
            assert plan[key] == first[key], key
        assert collection["plan_sha256"] == evidence["plan_sha256"]
        assert collection["path"] == (Path(plan["run"]) / "COMPLETE.json").as_posix()
        assert plan["mode"] == "collect" and plan["smoke"] is False and plan["shards"] == 4
    relative = "reports/" + task + "/plan.json"
    review_relative = "reports/" + task + "_ADMISSION.json"
    plan = dict(first)
    plan.update(
        mode="fit", launch_allowed=False, gpu=gpu, task=task, epochs=5,
        batch_size=batch_size, workers=workers,
        run=(Path(first["run"]).parent / task).as_posix(),
        review=dict(path=first["root"] + "/" + review_relative, sha256=None),
        collections=collections,
        collection_acceptance=dict(
            path=target_acceptance_path, sha256=hashlib.sha256(raw).hexdigest()),
        command=[first["command"][0],
                 first["root"] + "/scripts/train_e0_policy_refresh.py",
                 "--plan", first["root"] + "/" + relative])
    del plan["shard"], plan["shards"]
    binding = {key: value for key, value in plan.items()
               if key not in ("review", "launch_allowed")}
    digest = hashlib.sha256(json.dumps(
        binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata = dict(plan_binding_sha256=digest, plan_relative_path=relative,
                    review_relative_path=review_relative, launch_allowed=False,
                    expected_epochs=5, records=accepted["records"],
                    expected_updates=((accepted["records"] + batch_size - 1) // batch_size) * 5,
                    source_count=len(plan["sources"]))
    return plan, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", required=True)
    parser.add_argument("--target-acceptance", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan, binding = prepare(args.acceptance, args.target_acceptance, args.task,
                            args.gpu, args.batch_size, args.workers)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    print(json.dumps(binding))


if __name__ == "__main__":
    main()
