"""Predeclare full relative TRAIN bindings before observing fitted head results."""
import json
from pathlib import Path

from relative_e0_full_contract import NEW_SOURCES, STAGE
from relative_e0_smoke_contract import sha


ROOT = Path(__file__).resolve().parents[1]
FOLDER = "reports/e0_relative_full_training_r1_20261102"


def main():
    smoke_path = "reports/e0_relative_update_smoke_r1_20261102/draft.json"
    smoke = json.loads((ROOT / smoke_path).read_bytes())
    assert smoke["launch_allowed"] is False and smoke["full_training_allowed"] is False
    sources = dict(smoke["source_sha256"])
    for name in NEW_SOURCES:
        assert name not in sources
        sources[name] = sha(ROOT / name)
    plan = dict(
        stage=STAGE, phase="full", target_host="2026", target_root=smoke["target_root"],
        target_hostname=smoke["target_hostname"], launch_allowed=False, online_allowed=False,
        split="train", test_eligible=False, scientific_result_audit_accepted=False,
        device="cuda", assigned_gpu="0", seed=2027, epochs=12, batch_sets=512,
        lr=.0003, weight_decay=.0001, initialization=smoke["initialization"],
        initialization_sha256=smoke["initialization_sha256"],
        task="train_e0_relative_full_r1_p26_20261102",
        run="runs/e0_relative_full_r1_p26_20261102",
        collection_plan=smoke["full_supplement_plan"],
        collection_plan_sha256=smoke["full_supplement_plan_sha256"],
        collection_acceptance="reports/E0_M2_CURRENT_FULL_ACCEPTANCE_R1_20261102.json",
        collection_acceptance_sha256=None,
        update_smoke_draft=smoke_path, update_smoke_draft_sha256=sha(ROOT / smoke_path),
        update_smoke_plan="reports/e0_relative_update_smoke_r1_20261102/plan.json",
        update_smoke_plan_sha256=None, update_smoke_task=smoke["task"],
        update_smoke_run=smoke["run"], update_smoke_receipt_sha256=None,
        update_smoke_native_sha256=None,
        reviewed_draft=FOLDER + "/draft.json",
        source_review="reports/E0_M2_RELATIVE_FULL_TRAINING_SOURCE_REVIEW_R1_20261102.json",
        source_review_sha256=None, source_sha256=sources)
    directory = ROOT / FOLDER
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "draft.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(path=plan["reviewed_draft"], sha256=sha(directory / "draft.json"),
                          sources=len(sources), launch_allowed=False)))


if __name__ == "__main__":
    main()
