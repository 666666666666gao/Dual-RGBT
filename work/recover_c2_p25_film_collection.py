"""Recollect only FiLM shards whose original source-plan bytes were lost."""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from launch_c2_policy_collection_full import REMOTE
from prepare_c2_policy_collection import binding_sha

ROOT = Path(__file__).resolve().parents[1]
FOLDER = "reports/c2_policy_collection_recovery_r2_20261104/dropmoe/film"
REVIEW = "P2_C2_P25_FILM_COLLECTION_RECOVERY_REVIEW_R2_20261104.json"
REMOTE_ROOT = "/data2/gaob/Dual-RGBT/c2_policy_collection_dropmoe_r1_20261103"
OPERATIONS = (
    "work/recover_c2_p25_film_collection.py",
    "work/launch_c2_policy_collection_full.py",
    "work/prepare_c2_policy_collection.py",
    "tests/test_recover_c2_p25_film_collection.py",
    "reports/C2_P25_FILM_COLLECTION_RECOVERY_RUNBOOK_20261104.md",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def template(shard):
    return json.loads((ROOT / (
        "reports/c2_policy_collection_full_r1_20261103/dropmoe/film/s%d/plan.json"
        % shard)).read_bytes())


def make_plan(original, shard):
    assert shard in (0, 2)
    plan = copy.deepcopy(original)
    assert plan["host"] == "2025" and plan["variant"] == "film"
    assert plan["root"] == REMOTE_ROOT and plan["shard"] == shard
    assert (plan["seed"], plan["shards"], plan["label_stride"], plan["cost_weight"]) == (
        2027, 4, 5, .05)
    assert plan["smoke"] is False and plan["launch_allowed"] is False
    assert plan["c2_task_id"] == "train_c2_dropmoe_film_full_r1_p25_20261103"
    task = "collect_c2_policy_dropmoe_film_s%d_recovery_r2_p25_20261104" % shard
    plan.update(task=task, gpu={0: "2", 2: "3"}[shard],
                run=plan["run_root"] + "/" + task)
    plan["command"][-1] = REMOTE_ROOT + "/reports/" + task + "/plan.json"
    plan["review"] = dict(path=REMOTE_ROOT + "/reports/" + REVIEW,
                          sha256="PENDING_REVIEW_SHA256")
    return plan


def validate_review(plans, review):
    assert review["stage"] == "P2_C2_policy_refresh_collection"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["branch"] == "dropmoe" and review["host"] == "2025"
    assert set(review["operational_sources"]) == set(OPERATIONS)
    for name in OPERATIONS:
        assert sha(ROOT / name) == review["operational_sources"][name], name
    assert len(plans) == 2
    assert plans[0]["sources"] == plans[1]["sources"] == review["sources"]
    assert plans[0]["executor_binding"] == plans[1]["executor_binding"]
    assert plans[0]["smoke_evidence"] == plans[1]["smoke_evidence"]
    for plan, shard in zip(plans, (0, 2)):
        assert plan == make_plan(template(shard), shard)
        name = FOLDER + "/s%d/plan.json" % shard
        digest = binding_sha(plan)
        assert digest in review["admitted_plan_bindings"]
        row = next(row for row in review["reviewed_drafts"]["film"] if row["shard"] == shard)
        assert row["plan_path"] == name and row["plan_sha256"] == sha(ROOT / name)
        assert row["task"] == plan["task"] and row["gpu"] == plan["gpu"]
        assert row["admitted_plan_binding"] == digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "launch"))
    args = parser.parse_args()
    folder = ROOT / FOLDER
    if args.action == "prepare":
        assert not folder.exists()
        for shard in (0, 2):
            plan = make_plan(template(shard), shard)
            destination = folder / ("s%d" % shard)
            destination.mkdir(parents=True, exist_ok=False)
            (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
            (destination / "binding.json").write_text(json.dumps(dict(
                task=plan["task"], binding_sha256=binding_sha(plan),
                launch_allowed=False), indent=2) + "\n")
        print("PREPARED_DISABLED: new s0/GPU2 and s2/GPU3; old evidence preserved")
        return
    log = folder / "launch.log"
    assert not log.exists()
    plans = [json.loads((folder / ("s%d/plan.json" % shard)).read_bytes()) for shard in (0, 2)]
    review_raw = (ROOT / "reports" / REVIEW).read_bytes()
    validate_review(plans, json.loads(review_raw))
    for plan in plans:
        plan["launch_allowed"] = True
        plan["review"]["sha256"] = hashlib.sha256(review_raw).hexdigest()
    response = subprocess.run([
        "ssh", "-T", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
        "2025", plans[0]["command"][0], "-B", "-c", shlex.quote(REMOTE),
    ], input=json.dumps(dict(
        plans=plans, review_raw=base64.b64encode(review_raw).decode())),
        encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with log.open("x") as handle:
        handle.write(response.stdout)
    print(response.stdout, flush=True)
    response.check_returncode()


if __name__ == "__main__":
    main()
