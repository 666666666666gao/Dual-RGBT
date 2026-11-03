"""Prepare a disabled, source-bound P1B smoke draft; never launch or deploy."""
import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt.e0_policy_refresh import file_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-root", required=True)
    args = parser.parse_args()
    inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    branch = inputs["branch"]
    assert branch in ("sot", "dropmoe")
    source_review = ROOT / "reports/E0_EXECUTOR_V2_P1B_SOURCE_REVIEW_R3_20261102.json"
    reviewed = json.loads(source_review.read_text(encoding="utf-8"))
    assert reviewed["status"] == "SOURCE_PASS" and reviewed["blocking_issues"] == 0
    for name, digest in reviewed["sources"].items():
        assert file_sha(ROOT / name) == digest
    files = set(reviewed["sources"])
    files.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "lib").rglob("*.py"))
    files.update([
        "dual_rgbt/__init__.py", "experiments/cadtrack/cadtrack.yaml",
        "tools/prepare_e0_policy_refresh_plan.py"])
    sources = {name: file_sha(ROOT / name) for name in sorted(files)}
    target = Path(args.target_root).as_posix()
    task = "collect_p1b_{}_smoke_r1_p26_20261102".format(branch)
    plan_rel = "reports/p1b_{}_smoke_r1_20261102/plan.json".format(branch)
    review_rel = "reports/E0_EXECUTOR_V2_P1B_{}_SMOKE_ADMISSION_R1_20261102.json".format(branch.upper())
    interpreter = "/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python"
    plan = dict(
        stage="P1B_direct_E0", mode="collect", host="2026", root=target,
        launch_allowed=False, branch=branch, round=1, smoke=True,
        seed=2027, gpu="0", shard=0, shards=4, label_stride=1,
        cost_weight=.05, task=task, p1a_task_id=inputs["p1a_task_id"],
        inputs=inputs["inputs"], sources=sources,
        run="/data2/gb/Dual-RGBT/p1b_runs/" + task,
        review=dict(path=target + "/" + review_rel, sha256=None),
        command=[interpreter, target + "/scripts/collect_e0_policy_refresh.py",
                 "--plan", target + "/" + plan_rel])
    binding = {key: value for key, value in plan.items()
               if key not in ("review", "launch_allowed")}
    digest = hashlib.sha256(json.dumps(
        binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "binding.json").write_text(json.dumps(dict(
        plan_binding_sha256=digest, plan_relative_path=plan_rel,
        review_relative_path=review_rel, source_count=len(sources),
        source_review_sha256=file_sha(source_review),
        launch_allowed=False), indent=2) + "\n")
    with tarfile.open(str(output / "sources.tar.gz"), "w:gz") as archive:
        for name in sources:
            archive.add(str(ROOT / name), arcname=name, recursive=False)
    print(json.dumps(dict(source_count=len(sources),
                          plan_binding_sha256=digest, launch_allowed=False)))


if __name__ == "__main__":
    main()
