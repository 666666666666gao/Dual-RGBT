"""Check terminal ownership before full caches may become training inputs."""
import importlib.util
import json
from pathlib import Path
import socket

import pytest
import torch

from dual_rgbt.e0_policy_refresh import file_sha


def fixture(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "collection_verifier_test", root / "tools/verify_e0_policy_refresh_collection.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    entry = tmp_path / "scripts/train_e0_policy_refresh.py"
    entry.parent.mkdir()
    entry.write_bytes((root / "scripts/train_e0_policy_refresh.py").read_bytes())
    sequences = [dict(name="sequence_%d" % i, boxes=[[0, 0, 1, 1]] * 4,
                      valid_frames=[0, 1, 2, 3]) for i in range(4)]
    plans = {}
    for shard in range(4):
        run = tmp_path / ("run%d" % shard)
        run.mkdir()
        task = "task%d" % shard
        plan = dict(stage="P1B_direct_E0", mode="collect", host="2025",
                    root=str(tmp_path), inputs={"executor": {"sha256": "fixture"},
                                               "manifest": {"path": "fixture"}},
                    sources={"scripts/train_e0_policy_refresh.py": file_sha(entry)},
                    branch="fixture", round=1, smoke=False, seed=2027,
                    cost_weight=.05, label_stride=1, shards=4, shard=shard,
                    gpu=str(shard), task=task, command=["fixture", task], run=str(run))
        path = tmp_path / ("plan%d.json" % shard)
        path.write_text(json.dumps(plan))
        (run / "config.json").write_text(json.dumps(plan))
        artifact = run / (sequences[shard]["name"] + ".pth")
        torch.save(dict(sequence=sequences[shard]["name"],
                        records=[dict(frame=f) for f in (1, 2, 3)],
                        tracked_frames=3, plan_sha256=file_sha(path)), artifact)
        receipt = dict(stage="P1B_direct_E0_collection", status="COMPLETE",
                       smoke=False, benchmark_evidence=False, tracked_frames=3, records=3,
                       sequence_names=[sequences[shard]["name"]],
                       executor_sha256="fixture", plan_sha256=file_sha(path),
                       artifacts={artifact.name: file_sha(artifact)})
        (run / "COMPLETE.json").write_text(json.dumps(receipt))
        (run / "precision.json").write_text(json.dumps(dict(
            matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)))
        native = tmp_path / "runs/pipeline_tasks" / task / "status.json"
        native.parent.mkdir(parents=True)
        native.write_text(json.dumps(dict(
            state="COMPLETE", exit_code=0, task=task, physical_gpu=str(shard),
            hostname=socket.gethostname(), command=plan["command"])))
        plans[str(path)] = plan
    monkeypatch.setattr(verifier, "load_plan", lambda path, mode: plans[str(path)])
    monkeypatch.setattr(verifier, "train_sequences", lambda path: sequences)
    return verifier, list(plans), tmp_path


def test_accepts_complete_full_population(tmp_path, monkeypatch):
    verifier, paths, _ = fixture(tmp_path, monkeypatch)
    result = verifier.verify(paths)
    assert (result["sequences"], result["tracked_frames"], result["records"]) == (4, 12, 12)
    assert not result["full_fit_admitted"]


@pytest.mark.parametrize("field,value", [
    ("state", "RUNNING"), ("exit_code", 1),
    ("task", "other"), ("physical_gpu", "3"), ("command", ["other"])])
def test_rejects_nonterminal_or_wrong_native_task(tmp_path, monkeypatch, field, value):
    verifier, paths, folder = fixture(tmp_path, monkeypatch)
    path = folder / "runs/pipeline_tasks/task0/status.json"
    native = json.loads(path.read_text())
    native[field] = value
    path.write_text(json.dumps(native))
    with pytest.raises(AssertionError):
        verifier.verify(paths)


def test_rejects_precision_drift(tmp_path, monkeypatch):
    verifier, paths, folder = fixture(tmp_path, monkeypatch)
    path = folder / "run0/precision.json"
    precision = json.loads(path.read_text())
    precision["cudnn_allow_tf32"] = True
    path.write_text(json.dumps(precision))
    with pytest.raises(AssertionError):
        verifier.verify(paths)


def test_full_fit_draft_preserves_host_sources_and_actual_collection_binding(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "full_fit_preparer_test", root / "tools/prepare_e0_policy_refresh_full_fit.py")
    preparer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparer)
    collections, evidence = [], []
    for shard, count in enumerate((221, 221, 220, 220)):
        plan = dict(host="2025", root="/snapshot", sources={"entry": "fixture"},
                    inputs={"executor": {"sha256": "fixture"}}, branch="dropmoe",
                    round=1, label_stride=5, seed=2027, cost_weight=.05,
                    mode="collect", smoke=False, shards=4, shard=shard,
                    gpu=str(shard), task="collect%d" % shard,
                    run="/runs/collect%d" % shard, command=["/python", "/collector"])
        collections.append(dict(path=plan["run"] + "/COMPLETE.json",
                                sha256="receipt%d" % shard,
                                plan_sha256="plan%d" % shard, plan=plan))
        evidence.append(dict(sequences=count, tracked_frames=1000, records=200,
                             plan_sha256="plan%d" % shard))
    first = collections[0]["plan"]
    acceptance = dict(status="PASS", stage="P1B_direct_E0_full_collection_acceptance",
                      host=first["host"], root=first["root"], sources=first["sources"],
                      inputs=first["inputs"], branch=first["branch"], round=1,
                      label_stride=5, sequences=882, records=800, tracked_frames=4000,
                      cuda_initialized=False, full_fit_admitted=False, benchmark_evidence=False,
                      collections=collections, native_and_artifact_evidence=evidence)
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(acceptance))
    fit, metadata = preparer.prepare(str(path), "/acceptance.json", "fit_fixture", "2", 32, 4)
    assert fit["host"] == "2025" and fit["run"] == "/runs/fit_fixture"
    assert fit["sources"] == first["sources"] and fit["collections"] == collections
    assert fit["collection_acceptance"]["sha256"] == file_sha(path)
    assert fit["command"] == ["/python", "/snapshot/scripts/train_e0_policy_refresh.py",
                              "--plan", "/snapshot/reports/fit_fixture/plan.json"]
    assert not fit["launch_allowed"] and fit["epochs"] == 5
    assert metadata["expected_updates"] == 125
    acceptance["records"] = 801
    path.write_text(json.dumps(acceptance))
    with pytest.raises(AssertionError):
        preparer.prepare(str(path), "/acceptance.json", "fit_fixture", "2", 32, 4)
