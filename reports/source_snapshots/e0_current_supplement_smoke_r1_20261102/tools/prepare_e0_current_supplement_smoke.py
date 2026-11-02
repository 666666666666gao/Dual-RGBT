"""Prepare a disabled current-E0 smoke draft from the actual host26 inventory."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_bytes())
    assert audit["target_host"] == "2026" and audit["target_root"] == "/data2/gb/Dual-RGBT"
    assert not audit["missing_smoke_images"]
    assert audit["sequence_names"] == ["boysumbrella", "boyshorts", "whiteboyback"]
    assert len(audit["cache_sha256"]) == 3
    sources = dict(audit["source_sha256"])
    new_sources = [
        "dual_rgbt/e0_current_evidence.py",
        "tests/test_e0_current_evidence.py",
        "scripts/collect_e0_current_supplement.py",
        "tools/current_e0_supplement_contract.py",
        "tools/verify_e0_current_supplement.py",
        "tests/test_e0_current_supplement.py",
        "tools/prepare_e0_current_supplement_smoke.py",
    ]
    for name in new_sources:
        assert name not in sources
        sources[name] = sha(ROOT / name)
    directory = args.output_directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    relative = (directory / "draft.json").relative_to(ROOT).as_posix()
    plan = dict(
        stage="M2_current_E0_evidence_supplement", phase="smoke",
        target_host="2026", target_root=audit["target_root"],
        launch_allowed=False, training_allowed=False, split="train", test_eligible=False,
        scientific_result_audit_accepted=False,
        source_review="reports/E0_M2_CURRENT_SUPPLEMENT_SMOKE_SOURCE_REVIEW_R1_20261102.json",
        source_review_sha256=None, reviewed_draft=relative,
        source_sha256=sources, input_sha256=audit["input_sha256"],
        acceptance=audit["acceptance"], acceptance_sha256=audit["acceptance_sha256"],
        checkpoint=audit["checkpoint"], checkpoint_sha256=audit["checkpoint_sha256"],
        manifest=audit["manifest"], cache_sha256=audit["cache_sha256"],
        sequence_names=audit["sequence_names"], frames_per_sequence=31, cost_weight=0.,
        task="collect_e0_current_supplement_smoke_r1_p26_20261102",
        run="runs/e0_current_supplement_smoke_r1_p26_20261102",
        assigned_gpu="1", smoke_only=True, host_audit_sha256=sha(args.audit),
        runtime_contract="Original cost0 fixed parent; one observer, paired E0 states, "
                         "bit-exact cached predictions at every frame; original crop "
                         "binding before tracker import; two fixed-initial-template "
                         "encode/ROI calls per sampled frame. No24candidate re-execution, "
                         "anchor encoders, GT-derived decisions, training or online commits.")
    (directory / "draft.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(draft=relative, sha256=sha(directory / "draft.json"),
                          sources=len(sources), inputs=len(plan["input_sha256"]),
                          cache_inputs=len(plan["cache_sha256"]), launch_allowed=False)))


if __name__ == "__main__":
    main()
