"""Readiness and provenance gates for the disabled C2 evaluation preparer."""
import json

import pytest

from work import score_c2_policy_completed_remote as remote
from work.prepare_c2_policy_evaluation import NEW_SOURCES, binding, prepare, sha


def encode(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def fixture(host="2025", variant="zero"):
    sources = {name: "original-host-pin" for name in (
        "dual_rgbt/tracker.py", "dual_rgbt/c2_executor.py",
        "dual_rgbt/c2_policy_refresh.py", "dual_rgbt/model.py",
        "dual_rgbt/data.py", "scripts/run_task_four_gpu.py")}
    fit = dict(
        stage="P2_C2_policy_refresh_fit", mode="fit", launch_allowed=True,
        smoke=False, test_enabled=False, dev_enabled=False,
        host=host, variant=variant, c2_task_id="c2_" + variant,
        epochs=5, batch_size=512, workers=4, seed=2027, label_stride=5,
        cost_weight=.05, task="fit_" + variant, gpu="1",
        command=["/python", "-B", "/fit/train.py", "--plan", "/fit/plan.json"],
        executor_binding={"executor": "original"}, root="/fit", run="/fit_runs/fit",
        sources=sources)
    accepted = {key: fit[key] for key in (
        "host", "variant", "c2_task_id", "executor_binding")}
    accepted.update(
        stage="P2_C2_policy_refresh_full_collection_acceptance", status="PASS",
        sequences=882, tracked_frames=458553, records=91091)
    replay = dict(state="COMPLETE", exit_code=0)
    evidence = {}
    docs = {"acceptance": accepted, "replay_native": replay}
    for name, value in docs.items():
        evidence[name] = dict(path="/evidence/" + name + ".json", sha256=sha(encode(value)))
    fit["collection_acceptance"] = evidence["acceptance"]
    fit["collection_acceptance_native"] = evidence["replay_native"]
    review = dict(stage=fit["stage"], status="SOURCE_PASS", blocking_issues=0,
                  sources=sources, admitted_plan_bindings=[binding(fit)])
    evidence["review"] = dict(path="/evidence/review.json", sha256=sha(encode(review)))
    fit["review"] = evidence["review"]
    complete = dict(
        stage=fit["stage"], status="COMPLETE", smoke=False,
        epochs=5, steps=890, records=91091, retained_epochs=[1, 2, 3, 4, 5],
        artifacts={"epoch_%03d.pth" % n: str(n) * 64 for n in range(1, 6)},
        plan_sha256=sha(encode(fit)), executor_binding=fit["executor_binding"],
        executor_preserved=True, all_policy_modules_changed=True,
        strict_reload=True, benchmark_evidence=False)
    docs.update(
        plan=fit, config=fit, review=review, complete=complete,
        native=dict(state="COMPLETE", exit_code=0, task=fit["task"],
                    physical_gpu=fit["gpu"], command=fit["command"]),
        precision=dict(matmul_allow_tf32=False, cudnn_allow_tf32=False,
                       cudnn_benchmark=False))
    kwargs = dict(
        root="/evaluation", run_root="/evaluation_runs", python="/env/python",
        gpu="0", task_prefix="eval_fixed", review_path="/evaluation/reports/review.json",
        manifests={
            "smoke": dict(path="/manifests/train.json", raw=encode(dict(
                benchmark="LasHeR", split="train_smoke", sequences=[dict(rgb=list(range(11)))]))),
            "lasher": dict(path="/manifests/test.json", raw=encode(dict(
                benchmark="LasHeR", split="test", sequence_count=245, frame_count=220703)))},
        protocol=dict(path="/protocol.json", sha256="protocol-pin"), toolkit="/toolkit")
    return docs, kwargs


def run(docs, kwargs):
    raw = {name: encode(value) for name, value in docs.items()}
    evidence = {name: dict(path="/evidence/" + name + ".json", sha256=sha(value))
                for name, value in raw.items()}
    return prepare(raw, evidence, **kwargs)


@pytest.mark.parametrize("host,variant", [
    ("2025", "zero"), ("2025", "film"), ("2026", "zero"), ("2026", "film")])
def test_fixed_final_lasher_first_preserves_host_sources(host, variant):
    docs, kwargs = fixture(host, variant)
    plans, receipt = run(docs, kwargs)
    assert set(plans) == {"smoke", "lasher"}
    for mode, plan in plans.items():
        assert plan["launch_allowed"] is False
        assert plan["review"]["sha256"] == "PENDING"
        assert plan["checkpoint"]["path"] == "/fit_runs/fit/epoch_005.pth"
        assert plan["checkpoint"]["sha256"] == "5" * 64
        assert plan["cost_weight"] == .05 and plan["test_eligible"] is False
        assert plan["sources"]["dual_rgbt/model.py"] == "original-host-pin"
        assert set(NEW_SOURCES).issubset(plan["sources"])
        assert plan["smoke_frames"] == (11 if mode == "smoke" else 0)
        assert plan["exploratory_test"] is (mode == "lasher")
    assert receipt["admitted_plan_bindings"] == [binding(p) for p in plans.values()]
    assert receipt["actual_cpu_reconstruction_required"] is True
    assert receipt["benchmark_evidence"] is receipt["launch_allowed"] is False


@pytest.mark.parametrize("document,key,value", [
    ("native", "state", "RUNNING"), ("native", "exit_code", 1),
    ("native", "command", ["wrong"]), ("complete", "steps", 889),
    ("complete", "epochs", 4), ("complete", "retained_epochs", [2, 3, 4, 5]),
    ("complete", "strict_reload", False), ("complete", "executor_preserved", False),
    ("complete", "all_policy_modules_changed", False),
    ("acceptance", "status", "FAIL"), ("replay_native", "state", "RUNNING"),
    ("precision", "matmul_allow_tf32", True), ("review", "status", "SOURCE_FAIL"),
])
def test_incomplete_or_changed_evidence_rejected(document, key, value):
    docs, kwargs = fixture()
    docs[document][key] = value
    with pytest.raises(AssertionError):
        run(docs, kwargs)


def test_changed_archived_bytes_rejected():
    docs, kwargs = fixture()
    raw = {name: encode(value) for name, value in docs.items()}
    evidence = {name: dict(path="/evidence/" + name + ".json", sha256=sha(value))
                for name, value in raw.items()}
    raw["complete"] += b" "
    with pytest.raises(AssertionError):
        prepare(raw, evidence, **kwargs)


def test_partial_lasher_manifest_rejected():
    docs, kwargs = fixture()
    kwargs["manifests"]["lasher"]["raw"] = encode(dict(
        benchmark="LasHeR", split="test", sequence_count=244, frame_count=220703))
    with pytest.raises(AssertionError):
        run(docs, kwargs)


def test_generated_full_plan_passes_actual_scoring_request(tmp_path):
    docs, kwargs = fixture()
    plans, _ = run(docs, kwargs)
    plan = plans["lasher"]
    assert plan["plan_path"] == plan["command"][-1]
    review = dict(stage=plan["stage"], status="SOURCE_PASS", blocking_issues=0,
                  sources=plan["sources"], admitted_plan_bindings=[binding(plan)])
    review_path = tmp_path / "review.json"
    review_path.write_bytes(encode(review))
    plan["review"] = dict(path=str(review_path), sha256=sha(encode(review)))
    plan["launch_allowed"] = True
    local_plan = tmp_path / "plan.json"
    local_plan.write_bytes(encode(plan))
    actual, request = remote.prepare_request(local_plan, review_path)
    assert actual == plan
    assert request["plan_path"] == plan["command"][-1]
    assert request["plan_sha256"] == sha(local_plan.read_bytes())
