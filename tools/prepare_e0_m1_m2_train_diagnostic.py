"""Bind a fixed eight-sequence TRAIN diagnostic to the completed witness."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "reports/e0_m1_m2_train_diagnostic_r1_20261101"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    witness_path = ROOT / "reports/e0_m1_m2_train_smoke_r1_20261101/plan.json"
    assert digest(witness_path) == "0cc3a7c099722dccb1988f836b74ec405e1f972884ebabca3563f09a0b59eb51"
    witness = json.loads(witness_path.read_bytes())
    evidence = ROOT / OUTPUT / "inputs"
    collection_path = evidence / "collection_plan.json"
    manifest_path = evidence / "manifest.json"
    assert digest(collection_path) == "e0768c6adbda9fce190bd3eb8a1e4053bbf321c2c2fce03ec83c3cc1a56b23f8"
    assert digest(manifest_path) == witness["input_sha256"]["data/manifest.json"]
    collection = json.loads(collection_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    names = collection["fit_sequences"][:4] + collection["holdout_sequences"][:4]
    rows = {row["name"]: row for row in manifest["train"]}
    assert len(names) == len(set(names)) == 8
    source = dict(witness["source_sha256"])
    for name in ("scripts/diagnose_e0_m1_m2_train.py",
                 "tests/test_diagnose_e0_m1_m2_train.py",
                 "tools/prepare_e0_m1_m2_train_diagnostic.py"):
        source[name] = digest(ROOT / name)
    inputs = dict(witness["input_sha256"])
    collection_remote = "reports/E0_M2_IDENTITY_FULL_PLAN_P25_R1_20261101.json"
    inputs[collection_remote] = digest(collection_path)
    receipt = "runs/e0_m1_m2_gated_train_smoke_r1_p26_20261101/SMOKE.json"
    native = "runs/pipeline_tasks/smoke_e0_m1_m2_gated_r1_p26_20261101/status.json"
    inputs[receipt] = digest(ROOT / "reports/e0_m1_m2_train_smoke_r1_20261101/runtime/SMOKE.json")
    inputs[native] = digest(ROOT / "reports/e0_m1_m2_train_smoke_r1_20261101/native_status.json")
    inputs[witness_path.relative_to(ROOT).as_posix()] = digest(witness_path)
    plan = dict(witness)
    plan.pop("frames_per_sequence")
    plan.update(stage="E0_M1_M2_fixed_train_diagnostic", phase="diagnostic",
                launch_allowed=False, task="diagnose_e0_m1_m2_fixed_r1_p26_20261101",
                run="runs/e0_m1_m2_fixed_train_diagnostic_r1_p26_20261101",
                reviewed_draft=OUTPUT + "/draft.json",
                collection_plan=collection_remote, witness_receipt=receipt,
                witness_native=native, witness_plan=witness_path.relative_to(ROOT).as_posix(),
                witness_task=witness["task"], sequence_names=names,
                sequence_frames={name: len(rows[name]["rgb"]) for name in names},
                frames=sum(len(rows[name]["rgb"]) for name in names),
                source_sha256=source, input_sha256=inputs,
                scope="First four fit and first four component holdout TRAIN sequences in accepted "
                      "collection order, full lengths. Frozen weights, no threshold/seed search. "
                      "GT after decisions only. Original proposal shares intervened history; "
                      "not independent E0 control or TEST admission.")
    output = ROOT / plan["reviewed_draft"]
    with output.open("x") as handle:
        handle.write(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(path=str(output), sha256=digest(output), names=names,
                          frames=plan["frames"], source=len(source), inputs=len(inputs))))


if __name__ == "__main__":
    main()
