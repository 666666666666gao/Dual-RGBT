import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from work.launch_c2_policy_refresh_acceptance import (
    REMOTE, collection_plan_binding, validate_collection_source_pin,
    validate_expected_counts, validate_review,
)


def source_fixture(rebound=False):
    source = {
        "root": "/snapshot", "task": "collect", "gpu": "3",
        "variant": "zero", "shard": 3, "sources": {"collector.py": "sha"},
        "launch_allowed": False,
        "review": {"path": "/snapshot/old.json", "sha256": "PENDING"},
    }
    files = {"source.json": json.dumps(source).encode()}
    pin = {
        "source_plan": "source.json",
        "source_plan_sha256": hashlib.sha256(files["source.json"]).hexdigest(),
        "binding_sha256": collection_plan_binding(source),
        "path": "reports/review.json", "sha256": "review-sha",
    }
    final = copy.deepcopy(source)
    final.update(launch_allowed=True, review={
        "path": "/snapshot/reports/review.json", "sha256": "review-sha",
    })
    draft = {
        "shard": 3, "plan_path": pin["source_plan"],
        "plan_sha256": pin["source_plan_sha256"],
        "admitted_plan_binding": pin["binding_sha256"],
        "task": source["task"], "gpu": source["gpu"],
    }
    if rebound:
        original = copy.deepcopy(source)
        original.update(task="original", gpu="0")
        files["original.json"] = json.dumps(original).encode()
        pin.update(
            original_source_plan="original.json",
            original_source_plan_sha256=hashlib.sha256(
                files["original.json"]).hexdigest(),
            original_binding_sha256=collection_plan_binding(original),
        )
        draft.update(
            original_source_plan_path=pin["original_source_plan"],
            original_source_plan_sha256=pin["original_source_plan_sha256"],
            original_source_binding=pin["original_binding_sha256"],
        )
    review = {
        "sources": source["sources"], "reviewed_drafts": {"zero": [draft]},
    }
    return files, pin, final, review


def check_source(fixture):
    files, pin, final, review = fixture
    validate_collection_source_pin(
        files.__getitem__, pin, {"review_sha256": "review-sha"}, final, review)


@pytest.mark.parametrize("rebound", [False, True])
def test_finalized_plan_and_original_lineage_pass(rebound):
    check_source(source_fixture(rebound))


@pytest.mark.parametrize("field", [
    "source_plan_sha256", "binding_sha256", "sha256",
    "original_source_plan_sha256", "original_binding_sha256",
])
def test_wrong_pin_rejected(field):
    fixture = source_fixture(True)
    fixture[1][field] = "wrong"
    with pytest.raises(AssertionError):
        check_source(fixture)


def test_finalized_recipe_mutation_rejected():
    fixture = source_fixture()
    fixture[2]["cost_weight"] = 0.1
    with pytest.raises(AssertionError):
        check_source(fixture)


def test_original_raw_bytes_mutation_rejected():
    fixture = source_fixture(True)
    fixture[0]["original.json"] += b" "
    with pytest.raises(AssertionError):
        check_source(fixture)


@pytest.mark.parametrize("field", [
    "original_source_plan_path", "original_source_plan_sha256",
    "original_source_binding", "task", "gpu",
])
def test_wrong_reviewed_draft_rejected(field):
    fixture = source_fixture(True)
    fixture[3]["reviewed_drafts"]["zero"][0][field] = "wrong"
    with pytest.raises(AssertionError):
        check_source(fixture)


def test_expected_counts_and_false_claim_flags():
    totals = {"sequences": 882, "tracked_frames": 458553, "records": 91091}
    expected = dict(totals, benchmark_evidence=False, policy_fit_admitted=False)
    validate_expected_counts(totals, expected)
    with pytest.raises(AssertionError):
        validate_expected_counts(dict(totals, records=1), expected)
    for field in ("benchmark_evidence", "policy_fit_admitted"):
        with pytest.raises(AssertionError):
            validate_expected_counts(totals, dict(expected, **{field: True}))


def test_review_rejects_failure_or_stale_binding():
    plan = {"status": "DISABLED_PENDING_EXACT_REVIEW", "launch_allowed": False}
    review = {
        "status": "SOURCE_PASS", "blocking_issues": 0,
        "target": {"plan_sha256": "plan", "launcher_sha256": "launcher"},
        "scientific_result_audit_accepted": False, "actual_model": "UNATTESTED",
    }
    validate_review(plan, review, "plan", "launcher")
    with pytest.raises(AssertionError):
        validate_review(plan, dict(review, status="SOURCE_FAIL"), "plan", "launcher")
    with pytest.raises(AssertionError):
        validate_review(plan, review, "changed", "launcher")


def test_remote_compile_and_preflight_exits_before_mutation():
    source = REMOTE.replace("__VALIDATION_FUNCTIONS__", "\n".join(
        inspect.getsource(f) for f in (
            collection_plan_binding, validate_collection_source_pin,
            validate_expected_counts,
        ))).replace("__PAYLOAD_HEX__", "{}".encode().hex())
    compile(source, "<remote>", "exec")
    assert (source.index('if request["preflight_only"]:') <
        source.index("sys.exit(0)") < source.index("remote_witness.write_text")
        < source.index("process = subprocess.Popen"))
    assert 'float(gpu_row[3])' in source
    assert 'plan["python"], "-B"' in source


def test_exact_planned_command_matches_parent_environment():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / (
        "reports/C2_P26_SOT_ZERO_REPLAY_ACCEPTANCE_PLAN_R6_20261104.json"
    )).read_text())
    from pathlib import PurePosixPath
    remote_root = PurePosixPath(plan["root"])
    python = "/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python"
    assert plan["native_command"] == [
        python, "-B", str(remote_root / "scripts/run_task_four_gpu.py"),
        "--task", plan["task"], "--gpu", plan["gpu"], "--", python, "-B",
        str(remote_root / plan["verifier"]["path"]), "--plans",
    ] + [entry["path"] for entry in plan["collection_plans"]] + [
        "--output", plan["output"],
    ]
    assert plan["preflight_witness"] != plan["local_witness"]
