"""Contracts for preparing FIT only from the matching terminal replay."""
import copy
import json

import pytest

from work.launch_c2_policy_refresh_acceptance import collection_plan_binding
from work.prepare_c2_policy_refresh_fit import ENTRY, VERIFIER, prepare, sha256


def fixture():
    root = "/collection"
    python = "/env/bin/python"
    task = "verify_film"
    acceptance_path = root + "/reports/acceptance.json"
    native_path = root + "/runs/pipeline_tasks/" + task + "/status.json"
    binding = dict(stage="P2_C2_policy_refresh", host="2026",
                   variant="film", c2_task_id="train_film", inputs={})
    accepted = dict(
        status="PASS", stage="P2_C2_policy_refresh_full_collection_acceptance",
        host="2026", root=root, variant="film", c2_task_id="train_film",
        executor_binding=binding, sources={VERIFIER: "verifier-hash"},
        verifier_sha256="verifier-hash", cuda_initialized=True,
        sequences=882, tracked_frames=458553, records=91091,
        benchmark_evidence=False, policy_fit_admitted=False,
        collections=[], native_and_artifact_evidence=[])
    for shard, sequences in enumerate((221, 221, 220, 220)):
        plan = {key: copy.deepcopy(accepted[key]) for key in (
            "host", "root", "variant", "c2_task_id", "executor_binding",
            "sources")}
        path = root + "/reports/collect_" + str(shard) + "/plan.json"
        plan.update(
            shard=shard, shards=4, smoke=False,
            seed=2027, label_stride=5, cost_weight=.05,
            command=[python, "-B", root + "/collector.py", "--plan", path])
        accepted["collections"].append(dict(
            plan=plan, plan_sha256="plan-" + str(shard)))
        accepted["native_and_artifact_evidence"].append(dict(
            plan_path=path, plan_sha256="plan-" + str(shard),
            sequences=sequences,
            tracked_frames=(98613, 103761, 128483, 127696)[shard],
            records=(19574, 20578, 25541, 25398)[shard]))
    native = dict(
        state="COMPLETE", exit_code=0, physical_gpu="2", task=task,
        command=[python, "-B", root + "/" + VERIFIER, "--plans",
                 *[row["plan"]["command"][-1]
                   for row in accepted["collections"]],
                 "--output", acceptance_path])
    kwargs = dict(
        acceptance_path=acceptance_path, native_path=native_path,
        root="/fit_snapshot", run_root="/fit_runs", task="fit_film",
        gpu="0", python=python, review_path="/fit_snapshot/reports/review.json")
    return accepted, native, kwargs


def encode(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def test_matching_terminal_replay_prepares_disabled_fixed_recipe():
    accepted, native, kwargs = fixture()
    raw = encode(accepted)
    plan, binding = prepare(raw, encode(native), **kwargs)
    assert plan["launch_allowed"] is False
    assert plan["review"]["sha256"] == "PENDING"
    assert plan["epochs"] == 5 and plan["batch_size"] == 512
    assert plan["preview_lr"] == 4e-5 and plan["quality_gain_lr"] == 4e-4
    assert plan["executor_binding"] == accepted["executor_binding"]
    assert plan["collection_acceptance"]["sha256"] == sha256(raw)
    assert ENTRY in plan["sources"]
    assert plan["command"][-3] == "/fit_snapshot/" + ENTRY
    assert binding["admitted_plan_binding"] == collection_plan_binding(plan)
    assert binding["runtime_admission_required"] is True
    assert binding["benchmark_evidence"] is False
    assert plan["test_enabled"] is plan["dev_enabled"] is False


@pytest.mark.parametrize("target,field,value", [
    ("native", "state", "RUNNING"),
    ("native", "exit_code", 1),
    ("accepted", "status", "FAIL"),
    ("accepted", "variant", "zero"),
    ("accepted", "records", 91090),
    ("accepted", "benchmark_evidence", True),
    ("accepted", "verifier_sha256", "changed"),
])
def test_incomplete_or_mismatched_replay_rejected(target, field, value):
    accepted, native, kwargs = fixture()
    (accepted if target == "accepted" else native)[field] = value
    with pytest.raises(AssertionError):
        prepare(encode(accepted), encode(native), **kwargs)


def test_wrong_native_output_and_changed_shard_recipe_rejected():
    accepted, native, kwargs = fixture()
    native["command"][-1] = "/other/acceptance.json"
    with pytest.raises(AssertionError):
        prepare(encode(accepted), encode(native), **kwargs)
    accepted, native, kwargs = fixture()
    accepted["collections"][2]["plan"]["cost_weight"] = .15
    with pytest.raises(AssertionError):
        prepare(encode(accepted), encode(native), **kwargs)
