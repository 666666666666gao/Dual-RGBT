"""Freeze a disabled relative update-smoke proposal against a real host audit."""
import argparse
import json
from pathlib import Path

from relative_e0_smoke_contract import INITIAL_SHA, NEW_SOURCES, STAGE, sha


ROOT = Path(__file__).resolve().parents[1]
FOLDER = "reports/e0_relative_update_smoke_r1_20261102"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_bytes())
    assert audit["initialization_sha256"] == INITIAL_SHA
    assert audit["task_absent"] is True and audit["run_absent"] is True
    full_path = "reports/e0_current_full_r1_20261102/plan.json"
    full = json.loads((ROOT / full_path).read_bytes())
    assert audit["source_sha256"] == full["source_sha256"]
    assert audit["full_supplement_plan_sha256"] == sha(ROOT / full_path)
    sources = dict(full["source_sha256"])
    for name in NEW_SOURCES:
        sources[name] = sha(ROOT / name)
    plan = dict(
        stage=STAGE, phase="smoke", target_host="2026",
        target_root=audit["target_root"], target_hostname=audit["target_hostname"],
        launch_allowed=False, full_training_allowed=False, online_allowed=False,
        split="train", test_eligible=False, scientific_result_audit_accepted=False,
        device="cuda", assigned_gpu="0", seed=2027, optimizer_updates=5,
        batch_sets=6, lr=.0003, weight_decay=.0001,
        task=audit["task"], run=audit["run"],
        initialization=audit["initialization"], initialization_sha256=INITIAL_SHA,
        full_supplement_plan=full_path, full_supplement_plan_sha256=sha(ROOT / full_path),
        smoke_plan=full["smoke_plan"], smoke_plan_sha256=audit["smoke_plan_sha256"],
        smoke_run=full["smoke_run"], smoke_receipt_sha256=audit["smoke_verified"]["receipt_sha256"],
        smoke_native_sha256=audit["smoke_verified"]["native_sha256"],
        host_audit=args.audit.resolve().relative_to(ROOT).as_posix(), host_audit_sha256=sha(args.audit),
        reviewed_draft=FOLDER + "/draft.json",
        source_review="reports/E0_M2_RELATIVE_UPDATE_SMOKE_SOURCE_REVIEW_R1_20261102.json",
        source_review_sha256=None, source_sha256=sources)
    directory = ROOT / FOLDER
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "draft.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(path=plan["reviewed_draft"], sha256=sha(directory / "draft.json"),
                          sources=len(sources), launch_allowed=False)))


if __name__ == "__main__":
    main()
