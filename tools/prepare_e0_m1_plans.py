"""Read host26 identities and generate exact review drafts for the fixed experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    previous = json.loads((ROOT / "reports/E0_FINETUNED_EVALUATION_SOURCE_REVIEW_20261101.json").read_bytes())
    new = ["dual_rgbt/e0_m1_guard.py", "scripts/eval_e0_m1.py", "tools/score_e0_m1.py",
           "tests/test_e0_m1_guard.py", "tests/test_e0_observation.py"]
    existing = [name for name in previous["source_sha256"] if name not in (
        "scripts/eval_e0_finetuned.py", "tools/score_e0_finetuned.py")]
    existing += ["dual_rgbt/e0_observation.py", "dual_rgbt/e0_reliability.py"]
    inputs = ["checkpoints/CADTrack_ep0014.pth.tar",
              "runs/stage_c_all882_20261013/latest.pth",
              "runs/protected_e0_offline_m1_20261031/best.pth",
              "runs/protected_e0_offline_m1_20261031/COMPLETE.json",
              "runs/protected_e0_offline_m1_20261031/config.json",
              "runs/protected_e0_offline_m1_20261031/history.json",
              "runs/pipeline_tasks/train_protected_e0_offline_m1_r5_20261101/status.json",
              "data/manifest.json", "data/benchmarks/server26/lasher_test.json",
              "data/benchmarks/server26/rgbt210_test.json", "data/benchmarks/server26/rgbt234_test.json",
              "reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json"]
    payload = json.dumps(dict(source=existing, inputs=inputs))
    code = """
import json,hashlib
from pathlib import Path
r=Path('/data2/gb/Dual-RGBT')
spec=json.loads(__PAYLOAD__)
print(json.dumps({kind:{name:hashlib.sha256((r/name).read_bytes()).hexdigest()
                       for name in names} for kind,names in spec.items()}))
""".replace("__PAYLOAD__", repr(payload))
    output = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "2026",
         "cd /data2/gb/Dual-RGBT && conda-envs/dual_rgbt/bin/python -"],
        input=code.encode(), stdout=subprocess.PIPE, check=True)
    identities = json.loads(output.stdout)
    sources = dict(identities["source"])
    sources.update({name: digest(ROOT / name) for name in new})
    hashes = identities["inputs"]
    folder = ROOT / args.output
    folder.mkdir(exist_ok=False)
    common = dict(stage="E0_M1_template_guard_evaluation", target_host="2026",
                  target_root="/data2/gb/Dual-RGBT", training_allowed=False, test_eligible=False,
                  checkpoint=inputs[1], checkpoint_sha256=hashes[inputs[1]],
                  m1=inputs[2], m1_sha256=hashes[inputs[2]],
                  factory=inputs[0], factory_sha256=hashes[inputs[0]],
                  source_sha256=sources, input_sha256=hashes,
                  cost_weight=0.0, matmul_tf32=False, cudnn_tf32=False,
                  toolkit_protocol_sha256=hashes[inputs[-1]],
                  fixed_rule="Argmax reliable allows original commit; uncertain/lost undoes dynamic slot write only.",
                  scope="M1-only diagnostic ablation, not M2/M3 integration; upstream E0 saw component holdout.")
    dependencies = [dict(mode=mode, task="smoke_e0_m1_" + mode + "_r1_p26_20261101",
                         run="runs/e0_m1_" + mode + "_smoke_r1_p26_20261101")
                    for mode in ("observe", "guard")]
    for mode in ("observe", "guard"):
        plan = dict(common, mode=mode, smoke_frames=31, exploratory_test=False,
                    manifest="data/manifest.json", manifest_sha256=hashes["data/manifest.json"],
                    sequence_names=["2up", "motocross", "carturn"], smoke_dependencies=[])
        (folder / ("smoke_" + mode + ".json")).write_text(json.dumps(plan, indent=2) + "\n")
    for benchmark in ("lasher", "rgbt210", "rgbt234"):
        manifest = "data/benchmarks/server26/" + benchmark + "_test.json"
        for mode in ("control", "guard"):
            plan = dict(common, mode=mode, smoke_frames=0, exploratory_test=True,
                        manifest=manifest, manifest_sha256=hashes[manifest],
                        smoke_dependencies=dependencies)
            (folder / (benchmark + "_" + mode + ".json")).write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(drafts={p.name: digest(p) for p in sorted(folder.glob("*.json"))},
                         source_count=len(sources), input_count=len(hashes)), indent=2))


if __name__ == "__main__":
    main()
