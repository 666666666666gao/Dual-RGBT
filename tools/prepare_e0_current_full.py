"""Bind a disabled full supplement draft to a fresh host audit and real smoke."""
import argparse
import json
from pathlib import Path

from current_e0_supplement_contract import sha
from full_current_e0_supplement_contract import STAGE


ROOT = Path(__file__).resolve().parents[1]
NEW_SOURCES = [
    "scripts/collect_e0_current_full.py",
    "tools/full_current_e0_supplement_contract.py",
    "tools/verify_e0_current_full.py",
    "tools/prepare_e0_current_full.py",
    "tests/test_e0_current_full.py",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_bytes())
    assert audit["target_host"] == "2026" and audit["target_root"] == "/data2/gb/Dual-RGBT"
    assert audit["missing_images"] == 0 and audit["image_paths_checked"] == 918870
    assert len(audit["source_sha256"]) == 134 and len(audit["cache_sha256"]) == 882
    assert audit["smoke_verified"]["status"] == "CURRENT_E0_SUPPLEMENT_SMOKE_VERIFIED"
    sources = dict(audit["source_sha256"])
    for name in NEW_SOURCES:
        assert name not in sources
        sources[name] = sha(ROOT / name)
    directory = args.output_directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    draft_relative = (directory / "draft.json").relative_to(ROOT).as_posix()
    names = audit["sequence_names"]
    jobs = [dict(shard=i, assigned_gpu=str(i % 2),
                 sequence_names=names[i::8],
                 task="collect_e0_current_full_r1_p26_20261102_s%d" % i,
                 run="runs/e0_current_full_r1_p26_20261102_s%d" % i) for i in range(8)]
    plan = dict(
        stage=STAGE, phase="full", target_host="2026",
        target_root=audit["target_root"], target_hostname=audit["target_hostname"],
        launch_allowed=False, training_allowed=False, split="train", test_eligible=False,
        scientific_result_audit_accepted=False, cost_weight=0.,
        source_review="reports/E0_M2_CURRENT_FULL_SOURCE_REVIEW_R1_20261102.json",
        source_review_sha256=None, reviewed_draft=draft_relative,
        source_sha256=sources, input_sha256=audit["input_sha256"],
        acceptance=audit["acceptance"], acceptance_sha256=audit["acceptance_sha256"],
        checkpoint=audit["checkpoint"], checkpoint_sha256=audit["checkpoint_sha256"],
        manifest=audit["manifest"], cache_sha256=audit["cache_sha256"],
        sequence_names=names, fit_sequences=audit["fit_sequences"],
        holdout_sequences=audit["holdout_sequences"], shards=8, jobs=jobs,
        expected_frames=459435, expected_candidate_sets=18711,
        smoke_plan=audit["smoke_plan"], smoke_run=audit["smoke_run"],
        smoke_acceptance=audit["smoke_acceptance"],
        smoke_acceptance_sha256=audit["smoke_acceptance_sha256"],
        host_audit_sha256=sha(args.audit),
        runtime_contract="Original cost0 parent and fixed initial templates; one observer, "
                         "paired E0 paths and original cached predictions bit-exact every "
                         "frame; only current ROI/914-D observations added. Full original "
                         "882TRAIN sequences; no24candidate/anchor reruns, GT decisions, "
                         "training, TEST, or online commits.")
    (directory / "draft.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(draft=draft_relative, sha256=sha(directory / "draft.json"),
                          sources=len(sources), shards=8, launch_allowed=False)))


if __name__ == "__main__":
    main()
