"""Bind existing E0 evaluation to host26 and the completed final-five training."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/e0_ep20_p26_evaluation_r1_20261102"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remote(host, interpreter, source):
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         "-o", "ConnectTimeout=15", host, interpreter + " -"],
        input=source.encode(), stdout=subprocess.PIPE, check=True)
    return json.loads(result.stdout)


def main():
    assert not FOLDER.exists()
    old = json.loads((ROOT / "reports/E0_FINETUNED_EVALUATION_SOURCE_REVIEW_20261101.json").read_bytes())
    entries = ["scripts/eval_e0_finetuned.py", "tools/score_e0_finetuned.py"]
    names = [name for name in old["source_sha256"] if name not in entries]
    inputs = ["checkpoints/CADTrack_ep0014.pth.tar",
              "data/benchmarks/server26/lasher_train_smoke.json",
              "data/benchmarks/server26/lasher_test.json",
              "data/benchmarks/server26/rgbt210_test.json",
              "data/benchmarks/server26/rgbt234_test.json",
              "reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json"]
    target = remote("2026", "/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python", """
import hashlib,json
from pathlib import Path
r=Path('/data2/gb/Dual-RGBT')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
names=%r
inputs=%r
metadata={}
for n in inputs:
 if '/server26/' in n:
  m=json.loads((r/n).read_bytes())
  metadata[n]={k:m[k] for k in ('benchmark','split','sequence_count','frame_count')}
print(json.dumps(dict(source_sha256={n:sha(r/n) for n in names},
 input_sha256={n:sha(r/n) for n in inputs},
 manifests=metadata,
 entries_present={n:(r/n).exists() for n in %r})))
""" % (names, inputs, entries))
    training = remote("2025", "/home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python", """
import hashlib,json
from pathlib import Path
r=Path('/data2/gaob/Dual-RGBT/e0_executor_ep20_lasher_r1_20261101')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
native=Path('/home/gaob/Dual-RGBT/runs/pipeline_tasks/train_e0_executor_ep20_r1_20261101/status.json')
config=json.loads((r/'config.json').read_bytes())
assert sha(Path('/home/gaob/Dual-RGBT/scripts/finetune_e0_ep20.py'))==config['hashes']['entry']
print(json.dumps(dict(config=config,complete=json.loads((r/'COMPLETE.json').read_bytes()),
 native=json.loads(native.read_bytes()),native_sha256=sha(native),
 hashes={n:sha(r/n) for n in ['config.json','COMPLETE.json','steps.csv']+
 ['epoch_%03d.pth'%e for e in range(36,41)]})))
""")
    assert training["native"]["state"] == "COMPLETE" and training["native"]["exit_code"] == 0
    assert training["complete"]["epochs"] == 40 and training["complete"]["steps"] == 100000
    assert training["complete"]["retained_epochs"] == [36, 37, 38, 39, 40]
    assert training["config"]["stage"] == "E0_executor_ep20_finetune"
    source = dict(target["source_sha256"])
    source.update({name: sha(ROOT / name) for name in entries})
    assert len(source) == 49
    # Preserve actual target identities; never replace dependencies of the running M2 task.
    changes = {name: {"old_p25": old["source_sha256"][name], "p26": digest}
               for name, digest in source.items() if old["source_sha256"][name] != digest}
    destination = "checkpoints/e0_ep20_p25_completed_20261102"
    jobs = []
    for epoch in (40, 39, 38, 37, 36):
        for benchmark in ("lasher", "rgbt210", "rgbt234"):
            manifest = "data/benchmarks/server26/" + benchmark + "_test.json"
            metadata = target["manifests"][manifest]
            task = "eval_e0_ep20_e%03d_%s_r1_p26_20261102" % (epoch, benchmark)
            jobs.append(dict(epoch=epoch, benchmark=metadata["benchmark"], task=task,
                             run="runs/" + task, manifest=manifest,
                             manifest_sha256=target["input_sha256"][manifest],
                             checkpoint=destination + "/epoch_%03d.pth" % epoch,
                             checkpoint_sha256=training["hashes"]["epoch_%03d.pth" % epoch],
                             sequences=metadata["sequence_count"], frames=metadata["frame_count"],
                             score_output="reports/e0_ep20_p26_evaluation_r1_20261102/" + task + ".json"))
    draft = dict(stage="E0_finetuned_evaluation", target_host="2026",
                 target_root="/data2/gb/Dual-RGBT", launch_allowed=False,
                 allowed_gpus=["1"], max_concurrent=3, retained_epochs=[36, 37, 38, 39, 40],
                 epoch_order=[40, 39, 38, 37, 36], jobs=jobs, cost_weight=0.05,
                 factory=inputs[0], factory_sha256=target["input_sha256"][inputs[0]],
                 smoke_manifest=inputs[1], smoke_frames=11,
                 smoke_task="smoke_e0_ep20_e040_r1_p26_20261102",
                 smoke_run="runs/smoke_e0_ep20_e040_r1_p26_20261102",
                 source_sha256=source, input_sha256=target["input_sha256"],
                 training_hashes=training["hashes"], training_native_sha256=training["native_sha256"],
                 exploratory_test=True, test_eligible=False, optimizer_updates=0,
                 scope="All final-five E0 checkpoints, all three full benchmarks; LasHeR first. "
                       "Fixed final epoch40 is first by schedule, not selected on TEST. "
                       "Report all results descriptively. No TEST checkpoint/threshold/seed selection. "
                       "No M1/M2/M3 in this E0-only evaluation. Require successful TRAIN smoke "
                       "and exact target source/input/checkpoint/review bindings before full inference.")
    FOLDER.mkdir()
    for name, data in (("host_audit.json", dict(target=target, training=training,
                                               changed_from_p25_review=changes)),
                       ("draft.json", draft)):
        (FOLDER / name).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(dict(draft=str(FOLDER / "draft.json"), sha256=sha(FOLDER / "draft.json"),
                          source_count=len(source), source_differences=changes,
                          target_entries_present=target["entries_present"]), indent=2))


if __name__ == "__main__":
    main()
