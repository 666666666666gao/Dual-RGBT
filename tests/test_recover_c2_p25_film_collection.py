"""New identities repair lineage by collecting fresh data, never editing receipts."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))
SPEC = importlib.util.spec_from_file_location(
    "film_recovery", ROOT / "work/recover_c2_p25_film_collection.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


@pytest.fixture
def ready(tmp_path, monkeypatch):
    originals = {shard: recovery.template(shard) for shard in (0, 2)}
    plans = [recovery.make_plan(originals[shard], shard) for shard in (0, 2)]
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "template", lambda shard: copy.deepcopy(originals[shard]))
    for name in recovery.OPERATIONS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    rows = []
    for plan in plans:
        name = recovery.FOLDER + "/s%d/plan.json" % plan["shard"]
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(plan, indent=2) + "\n")
        rows.append(dict(
            shard=plan["shard"], task=plan["task"], gpu=plan["gpu"],
            plan_path=name, plan_sha256=recovery.sha(path),
            admitted_plan_binding=recovery.binding_sha(plan)))
    review = dict(
        stage="P2_C2_policy_refresh_collection", status="SOURCE_PASS",
        blocking_issues=0, branch="dropmoe", host="2025",
        sources=plans[0]["sources"],
        operational_sources={name: recovery.sha(tmp_path / name) for name in recovery.OPERATIONS},
        admitted_plan_bindings=[row["admitted_plan_binding"] for row in rows],
        reviewed_drafts={"film": rows})
    return plans, review, originals


def test_new_identity_preserves_executor_sources_smoke_and_original(ready):
    plans, review, originals = ready
    before = copy.deepcopy(originals)
    recovery.validate_review(plans, review)
    assert [p["gpu"] for p in plans] == ["2", "3"]
    assert [p["shard"] for p in plans] == [0, 2]
    for plan in plans:
        old = originals[plan["shard"]]
        assert recovery.make_plan(old, plan["shard"]) == plan
        assert plan["task"] != old["task"] and plan["run"] != old["run"]
        assert plan["command"][-1] != old["command"][-1]
        assert plan["executor_binding"] == old["executor_binding"]
        assert plan["sources"] == old["sources"]
        assert plan["smoke_evidence"] == old["smoke_evidence"]
    assert originals == before


@pytest.mark.parametrize("field,value", [
    ("gpu", "0"), ("task", "old"), ("seed", 2028), ("cost_weight", 0.),
    ("label_stride", 10), ("variant", "zero"), ("smoke", True),
    ("launch_allowed", True), ("root", "/other"),
])
def test_unreviewed_plan_changes_rejected(ready, field, value):
    plans, review, _ = ready
    plans[0][field] = value
    with pytest.raises(AssertionError):
        recovery.validate_review(plans, review)


@pytest.mark.parametrize("change", ["binding", "raw_sha", "source", "operations", "status"])
def test_changed_review_rejected(ready, change):
    plans, review, _ = ready
    if change == "binding":
        review["admitted_plan_bindings"] = []
    elif change == "raw_sha":
        review["reviewed_drafts"]["film"][0]["plan_sha256"] = "changed"
    elif change == "source":
        review["sources"] = {}
    elif change == "operations":
        review["operational_sources"][recovery.OPERATIONS[0]] = "changed"
    else:
        review["status"] = "SOURCE_FAIL"
    with pytest.raises(AssertionError):
        recovery.validate_review(plans, review)


def test_only_missing_shards_can_be_recollected(ready):
    _, _, originals = ready
    with pytest.raises(AssertionError):
        recovery.make_plan(originals[0], 1)
