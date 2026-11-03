"""Generate a disabled fit-smoke draft from actual collector receipts."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt.e0_policy_refresh import file_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt_path = Path(args.collection)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    collection_plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    native = json.loads(Path(args.native).read_text(encoding="utf-8"))
    assert receipt["status"] == "COMPLETE" and receipt["smoke"] is True
    assert receipt["stage"] == "P1B_direct_E0_collection"
    assert receipt["tracked_frames"] == receipt["records"] == 10
    assert receipt["plan_sha256"] == file_sha(args.plan)
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == collection_plan["task"]
    assert native["command"] == collection_plan["command"]
    assert native["physical_gpu"] == collection_plan["gpu"]
    for name, digest in receipt["artifacts"].items():
        assert file_sha(receipt_path.parent / name) == digest
    for name, digest in collection_plan["sources"].items():
        assert file_sha(ROOT / name) == digest
    branch, target = collection_plan["branch"], collection_plan["root"]
    task = "fit_p1b_{}_smoke_r1_p26_20261102".format(branch)
    plan_rel = "reports/p1b_{}_fit_smoke_r1_20261102/plan.json".format(branch)
    review_rel = "reports/E0_EXECUTOR_V2_P1B_{}_FIT_SMOKE_ADMISSION_R1_20261102.json".format(branch.upper())
    plan = dict(collection_plan)
    plan.update(
        mode="fit", launch_allowed=False, task=task, epochs=5,
        batch_size=8, workers=2,
        run="/data2/gb/Dual-RGBT/p1b_runs/" + task,
        review=dict(path=target + "/" + review_rel, sha256=None),
        collections=[dict(
            path=collection_plan["run"] + "/COMPLETE.json",
            sha256=file_sha(receipt_path),
            plan_sha256=file_sha(args.plan), plan=collection_plan)],
        command=[collection_plan["command"][0],
                 target + "/scripts/train_e0_policy_refresh.py",
                 "--plan", target + "/" + plan_rel])
    del plan["shard"], plan["shards"]
    binding = {key: value for key, value in plan.items()
               if key not in ("review", "launch_allowed")}
    digest = hashlib.sha256(json.dumps(
        binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "binding.json").write_text(json.dumps(dict(
        plan_binding_sha256=digest, launch_allowed=False,
        plan_relative_path=plan_rel, review_relative_path=review_rel,
        collection_receipt_sha256=file_sha(receipt_path),
        expected_real_updates=2, batch_size=8, records=10), indent=2) + "\n")
    print(json.dumps(dict(plan_binding_sha256=digest, launch_allowed=False)))


if __name__ == "__main__":
    main()
