"""Verify completed M1 loss experiment and replay both heads on identical caches."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from benchmark_inputs import sha
from dual_rgbt.e0_reliability import E0Reliability
from train_e0_reliability import load_collection, metrics


def verify_run(directory):
    receipt = json.loads((directory / "COMPLETE.json").read_bytes())
    assert receipt["artifacts_sha256"] == {
        p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != "COMPLETE.json"}
    assert receipt["optimizer_updates"] == 216 and receipt["strict_reload_bit_equal"]
    assert receipt["test_eligible"] is False
    return receipt


@torch.no_grad()
def evaluate(checkpoint, data, batch):
    assert checkpoint["selection_partition"] == "train_component_holdout"
    model = E0Reliability().eval()
    model.load_state_dict(checkpoint["net"], strict=True)
    return {name: metrics(model, (part["features"] - checkpoint["feature_mean"]) /
                          checkpoint["feature_scale"], part["iou"], batch)
            for name, part in data.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_bytes())
    new = ROOT / plan["run"]
    old = ROOT / "runs/protected_e0_offline_m1_20261031"
    receipts = {"old": verify_run(old), "balanced": verify_run(new)}
    native_path = ROOT / "runs/pipeline_tasks" / plan["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["task"] == receipts["balanced"]["task"] == plan["task"]
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["physical_gpu"] == "1"
    assert receipts["balanced"]["plan_sha256"] == sha(plan_path)
    config = json.loads((new / "config.json").read_bytes())
    assert config["plan"] == plan and config["native_command"] == native["command"]
    draft_path = ROOT / plan["reviewed_draft"]
    draft = json.loads(draft_path.read_bytes())
    assert draft["launch_allowed"] is False and plan["launch_allowed"] is True
    assert set(draft) == set(plan)
    assert all(plan[k] == v for k, v in draft.items() if k != "launch_allowed")
    review_path = ROOT / "reports/E0_M1_BALANCED_TRAINING_SOURCE_REVIEW_R1_20261101.json"
    assert sha(review_path) == config["source_review_sha256"]
    review = json.loads(review_path.read_bytes())
    assert review["stage"] == plan["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(draft_path)
    for kind in ("source_sha256", "input_sha256"):
        assert review[kind] == plan[kind]
        assert all(sha(ROOT / name) == value for name, value in plan[kind].items())
    data, used = load_collection(ROOT, plan["baseline_plan"], plan["historical_source_snapshots"])
    assert used == config["cache_inputs_sha256"]
    results = {}
    for name, directory in (("old", old), ("balanced", new)):
        checkpoint = torch.load(directory / "best.pth", map_location="cpu")
        replay = evaluate(checkpoint, data, plan["batch_states"])
        # CPU and CUDA reduction roundoff may affect CE, but class counts must agree.
        for part in replay:
            for key in ("states", "class_counts", "confusion_true_rows_predicted_columns",
                        "macro_recall", "lost_recall", "reliable_interrupted_fraction"):
                assert replay[part][key] == checkpoint["metrics"][part][key]
        results[name] = dict(epoch=checkpoint["epoch"], best_sha256=sha(directory / "best.pth"),
                             receipt_sha256=sha(directory / "COMPLETE.json"), replay=replay)
    old_h, new_h = [results[name]["replay"]["holdout"] for name in ("old", "balanced")]
    history = json.loads((new / "history.json").read_bytes())
    summary = dict(
        verification="PASS", native_sha256=sha(native_path), models=results,
        held_delta={key: new_h[key] - old_h[key] for key in
                    ("macro_recall", "lost_recall", "reliable_interrupted_fraction")},
        class_weights=config["class_weights"], optimizer_updates=216,
        last_five=[p.name for p in sorted(new.glob("epoch_*.pth"))],
        epoch_seconds=[entry["seconds"] for entry in history],
        actual_cached_input_count=len(used), test_eligible=False,
        scope="Same accepted TRAIN caches and head architecture; class-balanced loss control. "
              "Component holdout seen by upstream E0. No closed-loop/TEST gain or promotion.")
    assert summary["last_five"] == ["epoch_%03d.pth" % epoch for epoch in range(8, 13)]
    assert results["balanced"]["epoch"] == max(history, key=lambda e:e["holdout"]["macro_recall"])["epoch"]
    with args.output.open("x") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
