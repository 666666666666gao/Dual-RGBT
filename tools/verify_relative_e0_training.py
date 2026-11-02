"""Verify completed relative TRAIN artifacts and replay retained heads on CPU."""
import argparse
import json
import math
from pathlib import Path
import socket
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from current_e0_supplement_contract import sha
from dual_rgbt.e0_relative_identity import E0RelativeIdentity
from relative_e0_full_contract import admission, native_command
from relative_e0_training import acceptance_rank, evaluate


def compare_metrics(actual, recorded):
    assert set(actual) == set(recorded)
    for key, value in actual.items():
        if isinstance(value, float):
            assert math.isfinite(value) and math.isfinite(recorded[key]), key
            if key == "loss":
                assert math.isclose(value, recorded[key], rel_tol=1e-6, abs_tol=1e-7), key
            else:
                assert value == recorded[key], key
        else:
            assert value == recorded[key], key


def verify_metric_populations(metrics, data):
    counts = {
        "sets", "sequences", "reliable_sets", "lost_sets", "keep", "replacements",
        "reliable_to_below_half", "lost_to_at_least_half", "selected_original_unknown",
        "supervised_sets"}
    floats = {"mean_iou_delta", "sequence_macro_iou_delta",
              "sequence_macro_reliable_iou_delta", "loss"}
    assert set(metrics) == counts | floats
    size = len(data["current_iou"])
    for key in counts:
        assert type(metrics[key]) is int and 0 <= metrics[key] <= size, key
    for key in floats:
        assert math.isfinite(metrics[key]), key
    assert metrics["sets"] == size
    assert metrics["sequences"] == len(data["sequence_index"].unique())
    assert metrics["reliable_sets"] == int((data["current_iou"] >= .5).sum())
    assert metrics["lost_sets"] == int((data["current_iou"] < .1).sum())
    assert metrics["supervised_sets"] == int(data["current_coverage"].eq(1).any(1).sum())
    assert metrics["keep"] + metrics["replacements"] == size
    assert metrics["reliable_to_below_half"] <= min(
        metrics["reliable_sets"], metrics["replacements"])
    assert metrics["lost_to_at_least_half"] <= min(
        metrics["lost_sets"], metrics["replacements"])
    assert metrics["selected_original_unknown"] <= metrics["replacements"]
    assert metrics["loss"] >= 0
    for key in floats - {"loss"}:
        assert -1 <= metrics[key] <= 1, key


def verify_history(history, data, batch):
    assert len(history) == 12
    supervised = int(data["fit"]["current_coverage"].eq(1).any(1).sum())
    updates = (supervised + batch - 1) // batch
    ranked = []
    for epoch, row in enumerate(history, 1):
        assert row["epoch"] == epoch and row["optimizer_updates"] == epoch * updates
        assert row["training"]["optimizer_updates"] == updates
        assert row["training"]["supervised_sets"] == supervised
        assert math.isfinite(row["training"]["loss"]) and row["training"]["loss"] >= 0
        assert math.isfinite(row["epoch_seconds"]) and row["epoch_seconds"] > 0
        for part, size in (("fit", 14762), ("holdout", 3949)):
            metrics = row[part]
            assert metrics["sets"] == size
            verify_metric_populations(metrics, data[part])
        rank = acceptance_rank(row["holdout"], epoch)
        assert row["qualifies"] is (rank is not None)
        if rank is not None:
            ranked.append((rank, epoch))
    best = max(ranked)[1] if ranked else None
    return best, updates * 12


def verify_checkpoint(path, epoch, metrics, provenance, initial, data, batch):
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["epoch"] == epoch and checkpoint["metrics"] == metrics
    assert checkpoint["provenance"] == provenance
    assert checkpoint["selection_partition"] == "train_component_holdout"
    assert checkpoint["test_eligible"] is False
    state = checkpoint["net"]
    assert set(state) == set(initial)
    assert all(value.dtype == initial[key].dtype and value.shape == initial[key].shape
               and torch.isfinite(value).all() for key, value in state.items())
    assert torch.equal(state["no_match"], initial["no_match"])
    assert any(not torch.equal(value, initial[key])
               for key, value in state.items() if key != "no_match")
    head = E0RelativeIdentity().cpu().eval()
    head.load_state_dict(state, strict=True)
    assert checkpoint["total_parameters"] == sum(p.numel() for p in head.parameters()) == 116482
    replay = {part: evaluate(head, fields, batch) for part, fields in data.items()}
    for part in replay:
        compare_metrics(replay[part], metrics[part])
    return state, replay


def verify_completed(root, plan_path):
    root = Path(root)
    plan = json.loads((root / plan_path).read_bytes())
    native_path = root / "runs/pipeline_tasks" / plan["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == plan["task"]
    assert native["hostname"] == plan["target_hostname"] == socket.gethostname()
    assert native["physical_gpu"] == plan["assigned_gpu"]
    assert native["command"] == native_command(root, plan_path)
    assert native["completed"] >= native["started"]
    data = admission(root, plan)
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    directory = root / plan["run"]
    config = json.loads((directory / "config.json").read_bytes())
    receipt = json.loads((directory / "COMPLETE.json").read_bytes())
    history = json.loads((directory / "history.json").read_bytes())
    control = json.loads((directory / "frozen_control.json").read_bytes())
    provenance = {key: plan[key] for key in (
        "source_review_sha256", "collection_plan_sha256", "collection_acceptance_sha256",
        "initialization_sha256", "update_smoke_plan_sha256", "update_smoke_receipt_sha256",
        "update_smoke_native_sha256")}
    provenance.update(plan_path=plan_path, plan_sha256=sha(root / plan_path))
    assert config == dict(plan=plan, provenance=provenance)
    assert receipt["provenance"] == provenance
    assert receipt["status"] == "RELATIVE_CURRENT_E0_OFFLINE_TRAINING_COMPLETE"
    assert receipt["split"] == "train" and receipt["epochs"] == 12
    for key in ("test_eligible", "scientific_result_audit_accepted", "online_policy_changed",
                "protected_e0_loaded"):
        assert receipt[key] is False
    assert receipt["strict_reload_bit_equal"] is True
    assert receipt["frozen_no_match_unchanged"] is True
    assert math.isfinite(receipt["wall_seconds"]) and receipt["wall_seconds"] > 0
    assert 0 < receipt["peak_allocated_bytes"] <= receipt["peak_reserved_bytes"]
    assert receipt["total_parameters"] == 116482
    best, updates = verify_history(history, data, plan["batch_sets"])
    assert receipt["optimizer_updates"] == updates
    assert receipt["best_qualifying_epoch"] == best
    assert receipt["retain_protected_e0"] is (best is None)
    names = {"config.json", "COMPLETE.json", "history.json", "frozen_control.json",
             "latest.pth"} | {"epoch_%03d.pth" % epoch for epoch in range(8, 13)}
    if best is not None:
        names.add("best.pth")
    assert {p.name for p in directory.iterdir()} == names
    before = {name: sha(directory / name) for name in names}
    initial = torch.load(root / plan["initialization"], map_location="cpu")["net"]
    head = E0RelativeIdentity().cpu().eval()
    head.load_state_dict(initial, strict=True)
    for part, fields in data.items():
        compare_metrics(evaluate(head, fields, plan["batch_sets"]), control[part])
    replayed, replay_ranks = {}, []
    for epoch in range(8, 13):
        state, replayed[str(epoch)] = verify_checkpoint(
            directory / ("epoch_%03d.pth" % epoch), epoch,
            {part: history[epoch - 1][part] for part in data},
            provenance, initial, data, plan["batch_sets"])
        rank = acceptance_rank(replayed[str(epoch)]["holdout"], epoch)
        assert (rank is not None) is history[epoch - 1]["qualifies"]
        if rank is not None:
            replay_ranks.append((rank, epoch))
    latest, _ = verify_checkpoint(
        directory / "latest.pth", 12, {part: history[-1][part] for part in data},
        provenance, initial, data, plan["batch_sets"])
    assert all(torch.equal(latest[key], value) for key, value in state.items())
    if best is not None:
        _, replayed["best"] = verify_checkpoint(
            directory / "best.pth", best,
            {part: history[best - 1][part] for part in data},
            provenance, initial, data, plan["batch_sets"])
        rank = acceptance_rank(replayed["best"]["holdout"], best)
        assert rank is not None
        assert rank == acceptance_rank(history[best - 1]["holdout"], best)
        replay_ranks.append((rank, best))
        assert max(replay_ranks)[1] == best
    else:
        assert not replay_ranks
    assert before == {name: sha(directory / name) for name in names}
    for name, expected in plan["source_sha256"].items():
        assert sha(root / name) == expected, name
    assert not torch.cuda.is_initialized()
    return dict(
        status="RELATIVE_CURRENT_E0_FULL_TRAINING_VERIFIED",
        task=plan["task"], plan_sha256=sha(root / plan_path),
        native_sha256=sha(native_path), artifacts_sha256=before,
        best_qualifying_epoch=best, retain_protected_e0=best is None,
        optimizer_updates=updates, total_parameters=116482,
        retained_cpu_replay=replayed, replay_device="cpu",
        cuda_initialized=False, online_allowed=False, test_eligible=False,
        scientific_result_audit_accepted=False,
        scope="Terminal native/artifact verification; frozen control, retained/latest/best heads replayed on accepted TRAIN states with exact decision metrics. Deleted early epochs have structurally checked history only. CPU/GPU decision differences reject acceptance. Not closed-loop or benchmark evidence.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    verified = verify_completed(Path(args.root), args.plan)
    with output.open("x") as handle:
        json.dump(verified, handle, indent=2)
    print(json.dumps(verified), flush=True)


if __name__ == "__main__":
    main()
