"""Prepare one non-launchable same-architecture M1 class-balancing run."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = "reports/e0_m1_balanced_training_r1_20261101"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original_local = ROOT / "reports/e0_m1_m2_train_diagnostic_r1_20261101/inputs/original_m1_config.json"
    baseline = json.loads(original_local.read_bytes())["plan"]
    witness = json.loads((ROOT / "reports/e0_m1_m2_train_smoke_r1_20261101/plan.json").read_bytes())
    source = dict(witness["source_sha256"])
    for name in (
            "scripts/train_e0_reliability.py", "scripts/train_e0_reliability_balanced.py",
            "tests/test_train_e0_reliability_balanced.py", "tests/test_train_e0_reliability.py",
            "tools/prepare_e0_m1_balanced_training.py",
            "reports/source_snapshots/m1_balanced_train_r1_20261101/scripts/run_task_four_gpu.py"):
        source[name] = digest(ROOT / name)
    original_remote = "runs/protected_e0_offline_m1_20261031/config.json"
    inputs = {original_remote: digest(original_local),
              "data/manifest.json": witness["input_sha256"]["data/manifest.json"],
              baseline["collection_plan"]: baseline["collection_plan_sha256"],
              baseline["collection_acceptance"]: baseline["collection_acceptance_sha256"]}
    for entry in baseline["caches"]:
        inputs["runs/" + entry["run"] + "/COMPLETE.json"] = entry["receipt_sha256"]
        inputs["runs/pipeline_tasks/" + entry["task"] + "/status.json"] = entry["native_sha256"]
    plan = dict(schema_version=1, stage="M1_fit_class_balanced_training", split="train",
                training_allowed=True, launch_allowed=False, test_eligible=False,
                exploratory_test=False, target_host="2026", target_root="/data2/gb/Dual-RGBT",
                target_hostname=witness["target_hostname"], assigned_gpu="1",
                task="train_e0_m1_balanced_r1_p26_20261101",
                run="runs/protected_e0_offline_m1_balanced_r1_p26_20261101",
                reviewed_draft=OUT + "/draft.json", original_config=original_remote,
                baseline_plan=baseline, source_sha256=source, input_sha256=inputs,
                historical_source_snapshots={"scripts/run_task_four_gpu.py":
                    "reports/source_snapshots/m1_balanced_train_r1_20261101/scripts/run_task_four_gpu.py"},
                loss="fit_inverse_class_weighted_cross_entropy",
                selection="train_component_holdout_macro_recall",
                scope="Same 914-128-3 M1, same seed/optimizer/data/epochs; only fit inverse-class "
                      "loss weighting changes. Reuses accepted882TRAIN caches. Original wrapper "
                      "bytes pinned at historical snapshot, current wrapper separately reviewed. "
                      "No threshold search, E0 loading, recollection or TEST/closed-loop claim.")
    for key in ("seed", "epochs", "batch_states", "lr", "weight_decay", "device"):
        plan[key] = baseline[key]
    output = ROOT / plan["reviewed_draft"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(path=str(output), sha256=digest(output),
                          sources=len(source), inputs=len(inputs))))


if __name__ == "__main__":
    main()
