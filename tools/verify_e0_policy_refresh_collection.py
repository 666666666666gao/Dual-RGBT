"""Verify four completed full-TRAIN collectors without inference or training."""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt.e0_policy_refresh import file_sha, load_plan, train_sequences


def verify(plan_paths):
    assert len(plan_paths) == 4
    plans = [load_plan(path, "collect") for path in plan_paths]
    first = plans[0]
    assert first["smoke"] is False
    assert [plan["shard"] for plan in plans] == list(range(4))
    sequences = train_sequences(first["inputs"]["manifest"]["path"])
    collections, evidence = [], []
    tracked_frames = 0
    for path, plan in zip(plan_paths, plans):
        assert plan["shards"] == 4 and plan["smoke"] is False
        for key in ("host", "root", "inputs", "sources", "branch", "round",
                    "seed", "cost_weight", "label_stride"):
            assert plan[key] == first[key], key
        run = Path(plan["run"])
        receipt_path = run / "COMPLETE.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["stage"] == "P1B_direct_E0_collection"
        assert receipt["status"] == "COMPLETE" and receipt["smoke"] is False
        assert receipt["benchmark_evidence"] is False
        assert receipt["plan_sha256"] == file_sha(path)
        assert json.loads((run / "config.json").read_text()) == plan
        expected = sequences[plan["shard"]::4]
        assert receipt["sequence_names"] == [row["name"] for row in expected]
        native_path = Path(plan["root"]) / "runs/pipeline_tasks" / plan["task"] / "status.json"
        native = json.loads(native_path.read_text())
        assert native["state"] == "COMPLETE" and native["exit_code"] == 0
        assert native["task"] == plan["task"] and native["command"] == plan["command"]
        assert native["physical_gpu"] == plan["gpu"]
        assert native["hostname"] == socket.gethostname()
        precision_path = run / "precision.json"
        assert json.loads(precision_path.read_text()) == dict(
            matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)
        collections.append(dict(
            path=str(receipt_path), sha256=file_sha(receipt_path),
            plan_sha256=file_sha(path), plan=plan))
        evidence.append(dict(
            plan_path=str(path), plan_sha256=file_sha(path),
            native_path=str(native_path), native_sha256=file_sha(native_path),
            precision_path=str(precision_path), precision_sha256=file_sha(precision_path),
            sequences=len(expected), tracked_frames=receipt["tracked_frames"],
            records=receipt["records"], artifact_hashes=receipt["artifacts"]))
        tracked_frames += receipt["tracked_frames"]
    entry = Path(first["root"]) / "scripts/train_e0_policy_refresh.py"
    assert file_sha(entry) == first["sources"]["scripts/train_e0_policy_refresh.py"]
    spec = importlib.util.spec_from_file_location("verified_policy_refresh_trainer", entry)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    dataset_plan = dict(first, collections=collections)
    dataset = trainer.StateDataset(dataset_plan, sequences)
    assert tracked_frames == sum(len(row["boxes"]) - 1 for row in sequences)
    assert len(dataset) == sum(sum(
        frame > 0 and frame % first["label_stride"] == 0
        for frame in row["valid_frames"]) for row in sequences)
    return dict(
        status="PASS", stage="P1B_direct_E0_full_collection_acceptance",
        host=first["host"], root=first["root"], branch=first["branch"],
        round=first["round"], sources=first["sources"], inputs=first["inputs"],
        sequences=len(sequences), tracked_frames=tracked_frames, records=len(dataset),
        label_stride=first["label_stride"], collections=collections,
        native_and_artifact_evidence=evidence,
        verifier_sha256=file_sha(__file__), cuda_initialized=trainer.torch.cuda.is_initialized(),
        benchmark_evidence=False, full_fit_admitted=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", nargs=4, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    result = verify(args.plans)
    assert result["cuda_initialized"] is False
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], sequences=result["sequences"],
                          tracked_frames=result["tracked_frames"], records=result["records"])))


if __name__ == "__main__":
    main()
