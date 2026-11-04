"""Fixed epoch005 and exact full datasets survive the separate RGBT schedule."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))
SPEC = importlib.util.spec_from_file_location(
    "p1b_rgbt_preparer", ROOT / "work/prepare_p1b_rgbt_evaluation.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("branch,benchmark,gpu,count,frames", [
    ("dropmoe", "rgbt210", "2", 210, 104706),
    ("dropmoe", "rgbt234", "3", 234, 116650),
    ("sot", "rgbt210", "1", 210, 104706),
    ("sot", "rgbt234", "1", 234, 116650),
])
def test_exact_fixed_checkpoint_host_sources_and_complete_manifest(
        branch, benchmark, gpu, count, frames):
    original = json.loads((ROOT / "reports/p1a_evaluation_r1_20261102" /
                           branch / "draft.json").read_bytes())
    audit = json.loads((ROOT / "reports/p1b_epoch005_evaluation_audit_r1_20261104" /
                        branch / "audit.json").read_bytes())
    dependencies = json.loads((ROOT / "reports/p1b_evaluation_dependencies_r3_20261104" /
                               branch / "audit.json").read_bytes())
    expected = next(row for row in original["jobs"]
                    if row["benchmark"].lower() == benchmark)
    plan = MODULE.prepare(original, audit, "audit", dependencies, benchmark)
    assert plan["launch_allowed"] is False and plan["checkpoint"] == audit["checkpoint"]
    assert len(plan["source_sha256"]) == 58
    assert (plan["jobs"][0]["sequences"], plan["jobs"][0]["frames"]) == (count, frames)
    assert plan["jobs"][0]["manifest"] == expected["manifest"]
    assert plan["smoke"]["manifest"] == original["smoke"]["manifest"]
    assert (plan["smoke"]["sequences"], plan["smoke"]["frames"]) == (1, 11)
    assert plan["source_root"] == audit["root"]
    for job in [plan["smoke"]] + plan["jobs"]:
        command = job["command"]
        assert job["gpu"] == gpu
        assert "-B" not in command and command[1].startswith(plan["root"] + "/")
        assert command[command.index("--checkpoint") + 1] == audit["checkpoint"]["path"]
        assert command[command.index("--cost-weight") + 1] == "0.05"
        assert command[command.index("--source-review") + 1] == plan["review"]["path"]
        assert command[command.index("--manifest") + 1] == job["manifest"]["path"]
        assert job["test_eligible"] is False and benchmark in job["task"]
        assert job["run"].endswith(job["task"])


def test_rgbt_operator_uses_the_same_reviewed_runtime_gate():
    spec = importlib.util.spec_from_file_location(
        "rgbt_operator", ROOT / "work/operate_p1b_rgbt_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from operate_p1b_evaluation import REMOTE
    assert module.REMOTE == REMOTE
    compile(module.REMOTE, "unchanged_runtime", "exec")
