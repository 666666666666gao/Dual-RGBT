"""Prepare one TRAIN-only weight-decay control using the admitted M1 trainer."""
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main():
    code = """
import hashlib,json
from pathlib import Path
r=Path('/data2/gb/Dual-RGBT')
d=r/'runs/protected_e0_offline_m1_20261031'
c=json.loads((d/'config.json').read_bytes())
p=c['plan']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(r/p['source_review'])==p['source_review_sha256']
assert all(sha(r/n)==h for n,h in p['source_sha256'].items())
assert sha(r/p['collection_acceptance'])==p['collection_acceptance_sha256']
native=json.loads((r/'runs/pipeline_tasks/train_protected_e0_offline_m1_r5_20261101/status.json').read_bytes())
assert native['state']=='COMPLETE' and native['exit_code']==0
h=json.loads((d/'history.json').read_bytes())
assert len(h)==12 and h[-1]['holdout']['cross_entropy']>h[4]['holdout']['cross_entropy']
print(json.dumps(dict(plan=p,history=h,
 original_config_sha256=sha(d/'config.json'),
 original_native_sha256=sha(r/'runs/pipeline_tasks/train_protected_e0_offline_m1_r5_20261101/status.json'),
 original_complete_sha256=sha(d/'COMPLETE.json'))))
"""
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "2026",
         "/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python -"],
        input=code.encode(), stdout=subprocess.PIPE, check=True)
    evidence = json.loads(result.stdout)
    original = evidence["plan"]
    assert original["weight_decay"] == .0001
    assert original["epochs"] == 12 and original["batch_states"] == 4096
    plan = dict(original, launch_allowed=False, weight_decay=.001,
        planned_output="runs/protected_e0_offline_m1_wd1e3_r1_20261101")
    path = ROOT / "reports/E0_M1_WD1E3_CONTROL_DRAFT_R1_20261101.json"
    assert not path.exists()
    path.write_text(json.dumps(plan, indent=2) + "\n")
    evidence["plan"] = {"baseline_weight_decay": original["weight_decay"],
                        "control_weight_decay": plan["weight_decay"]}
    evidence.update(stage="M1_weight_decay_control",
        draft=path.relative_to(ROOT).as_posix(),
        draft_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        intended_task="train_protected_e0_m1_wd1e3_r1_p26_20261101",
        intended_gpu="3", launch_allowed=False,
        benchmark_selection_allowed=False, test_eligible=False,
        scientific_result_audit_accepted=False,
        scope="One TRAIN-only control prompted by fit/held loss divergence. "
              "Unchanged trainer/head/cache/split/seed/epochs/batch/lr. "
              "No TEST outcomes used. Existing fixed epoch5 evaluations continue. "
              "Review is for plan/source/input bindings, not scientific acceptance.")
    output = ROOT / "reports/E0_M1_WD1E3_CONTROL_EVIDENCE_R1_20261101.json"
    assert not output.exists()
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(dict(draft=str(path), sha256=evidence["draft_sha256"],
                          launched=False)), flush=True)


if __name__ == "__main__":
    main()
