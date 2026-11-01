"""Generate a non-launchable TRAIN witness draft from archived R15 bindings."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    old_path = ROOT / "reports/e0_m2_online_scores_r15_20261101/runtime/rgbt210/plan.json"
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == (
        "079d269c03c8f407ebe9220de4953fbf75c4e51f4b64845727655d8be20bfff9")
    old = json.loads(old_path.read_bytes())
    source = dict(old["source_sha256"])
    for name in ("dual_rgbt/e0_m1_m2_online.py", "dual_rgbt/e0_m1_guard.py",
                 "dual_rgbt/e0_reliability.py", "dual_rgbt/e0_observation.py",
                 "scripts/check_e0_m1_m2_train.py", "tests/test_e0_m1_m2_online.py",
                 "tests/test_check_e0_m1_m2_train.py",
                 "tools/prepare_e0_m1_m2_train_smoke.py"):
        source[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    m1 = "runs/protected_e0_offline_m1_20261031/best.pth"
    paths = [old["parent_checkpoint"], old["m2_checkpoint"],
             old["factory_checkpoint"], "data/manifest.json"]
    inputs = {name: old["input_sha256"][name] for name in paths}
    inputs[m1] = "658c84275ad1e1624d4b453fee89dd312273633664024bf0b42ae78dcc189ebf"
    relative = "reports/e0_m1_m2_train_smoke_r1_20261101/draft.json"
    draft = dict(schema_version=1, stage="E0_M1_M2_train_witness",
                 phase="smoke", split="train", launch_allowed=False,
                 training_allowed=False, exploratory_test=False, test_eligible=False,
                 target_host="2026", target_root="/data2/gb/Dual-RGBT",
                 target_hostname=old["target_hostname"], assigned_gpu="1",
                 task="smoke_e0_m1_m2_gated_r1_p26_20261101",
                 run="runs/e0_m1_m2_gated_train_smoke_r1_p26_20261101",
                 reviewed_draft=relative, manifest="data/manifest.json",
                 parent_checkpoint=old["parent_checkpoint"],
                 factory_checkpoint=old["factory_checkpoint"],
                 m1_checkpoint=m1, m2_checkpoint=old["m2_checkpoint"],
                 frames_per_sequence=31, sequence_names=["2up", "motocross", "carturn"],
                 source_sha256=source, input_sha256=inputs,
                 scope="Frozen TRAIN execution witness only. No policy acceptance, "
                       "training, TEST admission or scientific result.")
    output = ROOT / relative
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(draft, indent=2) + "\n")
    print(relative, hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
