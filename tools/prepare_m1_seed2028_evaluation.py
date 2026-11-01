"""Bind fixed TRAIN-selected seed2028 weights to three exploratory replications."""
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FOLDER = "reports/e0_m1_seed2028_evaluation_r1_20261101"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    code = """
import json,hashlib
from pathlib import Path
r=Path('/data2/gb/Dual-RGBT')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((r/'reports/e0_m1_template_guard_r3_20261101/lasher_guard.json').read_bytes())
assert all(sha(r/n)==h for n,h in p['source_sha256'].items())
assert all(sha(r/n)==h for n,h in p['input_sha256'].items())
new='runs/protected_e0_offline_m1_seed2028_r1_20261101/'
task='runs/pipeline_tasks/train_protected_e0_m1_seed2028_r1_p26_20261101/status.json'
n=json.loads((r/task).read_bytes());assert n['state']=='COMPLETE' and n['exit_code']==0
c=json.loads((r/(new+'COMPLETE.json')).read_bytes())
assert all(sha(r/(new+k))==v for k,v in c['artifacts_sha256'].items())
history=json.loads((r/(new+'history.json')).read_bytes())
assert len(history)==12
best=max(history,key=lambda x:x['holdout']['macro_recall'])
assert best['epoch']==3
assert sha(r/(new+'best.pth'))=='d747dc25f9f8acf0abfe137b268b4a6766a3d8785ea0193865297e116351c20e'
inputs={k:v for k,v in p['input_sha256'].items()
        if not k.startswith('runs/protected_e0_offline_m1_20261031/')
        and not k.startswith('runs/pipeline_tasks/train_protected_e0_offline_m1_r5_20261101/')}
for k in [new+x for x in ('best.pth','COMPLETE.json','config.json','history.json')]+[task]:
 inputs[k]=sha(r/k)
print(json.dumps(dict(template=p,inputs=inputs,training_best=best)))
"""
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         "2026", "/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python -"],
        input=code.encode(), stdout=subprocess.PIPE, check=True)
    evidence = json.loads(result.stdout)
    common = evidence["template"]
    common.update(m1="runs/protected_e0_offline_m1_seed2028_r1_20261101/best.pth",
        m1_sha256="d747dc25f9f8acf0abfe137b268b4a6766a3d8785ea0193865297e116351c20e",
        input_sha256=evidence["inputs"],
        scope="Fixed seed2028 replication selected only on TRAIN component "
              "holdout. Prior TEST outcomes were observed; exploratory only. "
              "Report all three outcomes without TEST checkpoint/model "
              "selection, threshold changes or replacement of original head.")
    common["source_sha256"]["scripts/eval_e0_m1_seed2028.py"] = sha(
        ROOT / "scripts/eval_e0_m1_seed2028.py")
    folder = ROOT / FOLDER
    folder.mkdir(exist_ok=False)
    dependencies = [dict(mode=mode,
        task="smoke_e0_m1_seed2028_" + mode + "_r1_p26_20261101",
        run="runs/e0_m1_seed2028_" + mode + "_smoke_r1_p26_20261101")
        for mode in ("observe", "guard")]
    for mode in ("observe", "guard"):
        manifest = "data/manifest.json"
        plan = dict(common, mode=mode, smoke_frames=31, exploratory_test=False,
            manifest=manifest, manifest_sha256=evidence["inputs"][manifest],
            sequence_names=["2up", "motocross", "carturn"], smoke_dependencies=[])
        (folder / ("smoke_" + mode + ".json")).write_text(
            json.dumps(plan, indent=2) + "\n")
    for benchmark in ("lasher", "rgbt210", "rgbt234"):
        manifest = "data/benchmarks/server26/" + benchmark + "_test.json"
        plan = dict(common, mode="guard", smoke_frames=0, exploratory_test=True,
            manifest=manifest, manifest_sha256=evidence["inputs"][manifest],
            smoke_dependencies=dependencies)
        (folder / (benchmark + "_guard.json")).write_text(
            json.dumps(plan, indent=2) + "\n")
    evidence.update(stage="E0_M1_template_guard_evaluation",
        test_outcomes_already_observed=True, test_eligible=False,
        scientific_result_audit_accepted=False,
        reviewed_drafts={p.name: sha(p) for p in sorted(folder.glob("*.json"))},
        entry="scripts/eval_e0_m1_seed2028.py",
        entry_sha256=sha(ROOT / "scripts/eval_e0_m1_seed2028.py"),
        evaluator_original_bytes_unchanged=True,
        benchmark_selection_allowed=False, fixed_replication_seed=2028)
    (folder / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence["reviewed_drafts"], indent=2))


if __name__ == "__main__":
    main()
