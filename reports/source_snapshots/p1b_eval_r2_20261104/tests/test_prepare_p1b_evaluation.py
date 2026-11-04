"""Fixed P1B checkpoint and host-specific evaluator deployment contracts."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p1b_eval_preparer", ROOT / "work/prepare_p1b_evaluation.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs(branch):
    original = json.loads((ROOT / "reports/p1a_evaluation_r1_20261102" /
                           branch / "draft.json").read_bytes())
    audit = json.loads((ROOT / "reports/p1b_epoch005_evaluation_audit_r1_20261104" /
                        branch / "audit.json").read_bytes())
    return original, audit


@pytest.mark.parametrize("branch", ["dropmoe", "sot"])
def test_fixed_epoch005_preserves_host_map_and_lasher_first(branch):
    original, audit = inputs(branch)
    plan = MODULE.prepare(original, audit, "audit")
    assert plan["launch_allowed"] is False
    assert plan["checkpoint"] == audit["checkpoint"]
    assert len(plan["source_sha256"]) == 50
    assert {key: value for key, value in plan["source_sha256"].items()
            if key != "scripts/eval_p1b_completed.py"} == original["source_sha256"]
    assert plan["root"] != original["root"]
    assert len(plan["jobs"]) == 1
    for job in [plan["smoke"]] + plan["jobs"]:
        assert job["command"][0] == original["smoke"]["command"][0]
        assert "-B" not in job["command"]
        assert job["command"][1] == plan["root"] + "/scripts/eval_p1b_completed.py"
        assert job["command"][job["command"].index("--cost-weight") + 1] == "0.05"
        assert job["test_eligible"] is False
    assert (plan["jobs"][0]["sequences"], plan["jobs"][0]["frames"]) == (245, 220703)
    assert plan["smoke"]["frames"] == 11


@pytest.mark.parametrize("key,value", [
    ("status", "RUNNING"), ("cuda_initialized", True),
    ("executor_preserved", False), ("all_policy_modules_changed", False),
])
def test_incomplete_or_invalid_audit_rejected(key, value):
    original, audit = inputs("sot")
    audit[key] = value
    with pytest.raises(AssertionError):
        MODULE.prepare(original, audit, "audit")


def test_host_map_cannot_be_substituted():
    original, audit = inputs("sot")
    audit = copy.deepcopy(audit)
    audit["source_sha256"] = inputs("dropmoe")[1]["source_sha256"]
    with pytest.raises(AssertionError):
        MODULE.prepare(original, audit, "audit")


def test_remote_scripts_compile_without_running():
    for filename in ("audit_p1b_completed_for_evaluation.py", "operate_p1b_evaluation.py"):
        spec = importlib.util.spec_from_file_location(filename, ROOT / "work" / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        compile(module.REMOTE, filename + ":REMOTE", "exec")
