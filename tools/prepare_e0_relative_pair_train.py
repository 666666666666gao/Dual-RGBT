"""Write a disabled fixed paired-TRAIN draft; trained evidence binds only later."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from relative_e0_pair_contract import (
    CORE_REVIEW, CORE_SHA, FOLDER, FULL, NAMES, NEW_SOURCES, RUN, STAGE, TASK, recipe, sha)


def main():
    full_path = ROOT / FULL / "draft.json"
    assert sha(full_path) == "499d301bab553c0b9384bb86b9af8c214292247fe29a71d057e0e267de84ec46"
    full = json.loads(full_path.read_bytes())
    assert sha(ROOT / CORE_REVIEW) == CORE_SHA
    core = json.loads((ROOT / CORE_REVIEW).read_bytes())
    old_path = ROOT / "reports/e0_m1_m2_train_diagnostic_r1_20261101/draft.json"
    old = json.loads(old_path.read_bytes())
    assert old["sequence_names"] == list(NAMES) and old["frames"] == 3076
    source = dict(full["source_sha256"])
    for mapping in (core["source_sha256"], core["read_only_dependency_sha256"]):
        for name, digest in mapping.items():
            if name in source:
                assert source[name] == digest, name
            source[name] = digest
    for name in NEW_SOURCES:
        source[name] = sha(ROOT / name)
    paths = ("parent_checkpoint", "factory_checkpoint", "m1_checkpoint", "manifest")
    inputs = {old[key]: old["input_sha256"][old[key]] for key in paths}
    inputs[FULL + "/draft.json"] = sha(full_path)
    inputs[CORE_REVIEW] = CORE_SHA
    draft = dict(
        stage=STAGE, phase="paired_train", split="train", launch_allowed=False,
        training_allowed=False, exploratory_test=False, test_eligible=False,
        benchmark_allowed=False, scientific_result_audit_accepted=False,
        target_host="2026", target_root="/data2/gb/Dual-RGBT",
        target_hostname=full["target_hostname"], assigned_gpu="0", seed=2027, cost_weight=0.,
        task=TASK, run=RUN, reviewed_draft=FOLDER + "/draft.json",
        source_review="reports/E0_M2_RELATIVE_PAIR_TRAIN_SOURCE_REVIEW_R1_20261102.json",
        source_review_sha256=None, training_draft=FULL + "/draft.json",
        training_plan=FULL + "/plan.json", training_plan_sha256=None,
        training_acceptance="reports/E0_M2_RELATIVE_FULL_TRAIN_ACCEPTANCE_R1_20261102.json",
        training_acceptance_sha256=None,
        relative_checkpoint="runs/e0_relative_full_r1_p26_20261102/best.pth",
        relative_checkpoint_sha256=None, qualifying_epoch=None,
        sequence_names=list(NAMES), sequence_frames=old["sequence_frames"], frames=3076,
        arms=["e0_m1_control", "e0_m1_relative_m2"],
        source_sha256=source, input_sha256=inputs, **{key: old[key] for key in paths})
    recipe(draft)
    path = ROOT / FOLDER / "draft.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(draft, indent=2) + "\n")
    print(json.dumps(dict(path=str(path), sha256=sha(path), sources=len(source))))


if __name__ == "__main__":
    main()
