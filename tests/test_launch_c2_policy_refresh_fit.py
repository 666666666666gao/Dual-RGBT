"""CPU admission contracts; these do not launch a real FIT."""
import copy
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from work.launch_c2_policy_refresh_acceptance import collection_plan_binding
from work.launch_c2_policy_refresh_fit import (
    REMOTE, validate_capacity, validate_review, validate_terminal_replay,
)
from work.prepare_c2_policy_refresh_fit import prepare

SPEC = importlib.util.spec_from_file_location(
    "fit_preparer_fixtures",
    Path(__file__).with_name("test_prepare_c2_policy_refresh_fit.py"))
FIXTURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURES)


def fixture():
    accepted, native, kwargs = FIXTURES.fixture()
    plan, _ = prepare(
        FIXTURES.encode(accepted), FIXTURES.encode(native), **kwargs)
    review = dict(
        stage=plan["stage"], status="SOURCE_PASS", blocking_issues=0,
        sources=copy.deepcopy(plan["sources"]),
        admitted_plan_bindings=[collection_plan_binding(plan)],
        operational_sources={"work/launch_c2_policy_refresh_fit.py": "launcher"},
        actual_model="UNATTESTED", scientific_result_audit_accepted=False)
    return plan, accepted, native, review


def test_exact_plan_review_and_matching_terminal_replay_pass():
    plan, accepted, native, review = fixture()
    validate_review(plan, review, "launcher")
    validate_terminal_replay(plan, accepted, native, plan["command"][0])


@pytest.mark.parametrize("change", ["recipe", "source", "launcher", "binding"])
def test_stale_plan_source_launcher_or_binding_review_rejected(change):
    plan, _, _, review = fixture()
    if change == "recipe":
        plan["preview_lr"] *= 2
    elif change == "source":
        review["sources"]["unreviewed.py"] = "changed"
    elif change == "launcher":
        review["operational_sources"]["work/launch_c2_policy_refresh_fit.py"] = "stale"
    else:
        review["admitted_plan_bindings"] = []
    with pytest.raises(AssertionError):
        validate_review(plan, review, "launcher")


@pytest.mark.parametrize("change", [
    "running", "exit", "variant", "input", "native_output", "count", "source",
])
def test_nonterminal_or_mismatched_replay_rejected(change):
    plan, accepted, native, _ = fixture()
    if change == "running":
        native["state"] = "RUNNING"
    elif change == "exit":
        native["exit_code"] = 1
    elif change == "variant":
        accepted["variant"] = "zero"
    elif change == "input":
        plan["executor_binding"] = dict(plan["executor_binding"], inputs={"wrong": {}})
    elif change == "native_output":
        native["command"][-1] = "/wrong/output.json"
    elif change == "count":
        accepted["native_and_artifact_evidence"][2]["records"] += 1
    else:
        plan["sources"]["unexpected.py"] = "extra"
    with pytest.raises(AssertionError):
        validate_terminal_replay(plan, accepted, native, plan["command"][0])


def test_target_gpu_free_other_gpu_occupied_is_allowed():
    plan, _, _, _ = fixture()
    rows = ["0, 24000, 100, 350, 40, target",
            "3, 22000, 2000, 350, 75, other"]
    assert validate_capacity(plan, rows, ["other, 123"], 20 * 2**30)[5] == "target"
    plan["host"] = "2025"
    rows[0] = "0, 24000, 100, 250, 40, target"
    validate_capacity(plan, rows, [], 20 * 2**30)


@pytest.mark.parametrize("gpu_row,processes,disk", [
    ("0, 19000, 100, 350, 40, target", [], 20),
    ("0, 24000, 1000, 350, 40, target", [], 20),
    ("0, 24000, 100, 250, 40, target", [], 20),
    ("0, 24000, 100, 350, 85, target", [], 20),
    ("0, 24000, 100, 350, 40, target", ["target, 123"], 20),
    ("0, 24000, 100, 350, 40, target", [], 11),
])
def test_capacity_rejects_target_conflicts(gpu_row, processes, disk):
    plan, _, _, _ = fixture()
    with pytest.raises(AssertionError):
        validate_capacity(plan, [gpu_row], processes, disk * 2**30)


def test_remote_preflight_compiles_and_exits_before_deployment_or_launch():
    source = REMOTE.replace("__FUNCTIONS__", "\n".join(
        inspect.getsource(function) for function in (
            collection_plan_binding, validate_review,
            validate_terminal_replay, validate_capacity,
        ))).replace("__REQUEST__", json.dumps({}).encode().hex())
    compile(source, "<remote-fit-admission>", "exec")
    assert (source.index('if request["preflight_only"]:') <
            source.index("sys.exit(0)") <
            source.index("root.mkdir(exist_ok=False)") <
            source.index("process = subprocess.Popen"))
    assert 'entry.load_plan(plan_path)' in source
    assert 'completed_executor_state(plan["executor_binding"])' in source
