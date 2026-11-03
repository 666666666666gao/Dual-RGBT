"""Prepare disabled full-TRAIN shard plans after a real two-update fit smoke."""
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
    parser.add_argument("--collection-plan", required=True)
    parser.add_argument("--fit-plan", required=True)
    parser.add_argument("--fit-complete", required=True)
    parser.add_argument("--fit-native", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-stride", type=int, required=True)
    args = parser.parse_args()
    assert args.label_stride > 0
    collection = json.loads(Path(args.collection_plan).read_text(encoding="utf-8"))
    fit = json.loads(Path(args.fit_plan).read_text(encoding="utf-8"))
    receipt = json.loads(Path(args.fit_complete).read_text(encoding="utf-8"))
    native = json.loads(Path(args.fit_native).read_text(encoding="utf-8"))
    assert collection["mode"] == "collect" and collection["smoke"] is True
    assert fit["mode"] == "fit" and fit["smoke"] is True
    assert receipt["status"] == "COMPLETE" and receipt["smoke"] is True
    assert receipt["stage"] == "P1B_direct_E0_fit" and receipt["steps"] == 2
    assert receipt["executor_preserved"] and receipt["all_policy_modules_changed"]
    assert receipt["strict_reload"] and receipt["plan_sha256"] == file_sha(args.fit_plan)
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == fit["task"] and native["command"] == fit["command"]
    for key in ("inputs", "sources", "branch", "round", "seed", "cost_weight"):
        assert collection[key] == fit[key], key
    assert receipt["sources"] == fit["sources"]
    assert receipt["executor_sha256"] == fit["inputs"]["executor"]["sha256"]
    assert receipt["closed_loop_round"] == fit["round"] == 1
    for name, digest in fit["sources"].items():
        assert file_sha(ROOT / name) == digest
    branch, target = fit["branch"], fit["root"]
    host_tag = {"2025": "p25", "2026": "p26"}[fit["host"]]
    review_rel = "reports/E0_EXECUTOR_V2_P1B_{}_FULL_COLLECTION_ADMISSION_R1_20261102.json".format(branch.upper())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    bindings = []
    for shard in range(4):
        task = "collect_p1b_{}_full_s{}_r1_{}_20261102".format(branch, shard, host_tag)
        plan_rel = "reports/p1b_{}_full_collection_r1_20261102/s{}/plan.json".format(branch, shard)
        plan = dict(collection)
        plan.update(
            launch_allowed=False, smoke=False, gpu=str(shard),
            shard=shard, shards=4, task=task, label_stride=args.label_stride,
            run=(Path(collection["run"]).parent / task).as_posix(),
            review=dict(path=target + "/" + review_rel, sha256=None),
            smoke_acceptance=dict(path=fit["run"] + "/COMPLETE.json",
                                  sha256=file_sha(args.fit_complete)),
            command=[collection["command"][0],
                     target + "/scripts/collect_e0_policy_refresh.py",
                     "--plan", target + "/" + plan_rel])
        binding = {key: value for key, value in plan.items()
                   if key not in ("review", "launch_allowed")}
        digest = hashlib.sha256(json.dumps(
            binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        folder = output / ("s{}".format(shard))
        folder.mkdir()
        (folder / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        bindings.append(dict(shard=shard, gpu=str(shard), task=task,
                             plan_relative_path=plan_rel,
                             plan_binding_sha256=digest))
    (output / "binding.json").write_text(json.dumps(dict(
        launch_allowed=False, review_relative_path=review_rel,
        full_train_sequences=882, all_frames_tracked=True,
        label_stride=args.label_stride, shards=bindings), indent=2) + "\n")
    print(json.dumps(dict(launch_allowed=False, shards=bindings)))


if __name__ == "__main__":
    main()
