"""Operational handoff: wait for admitted collectors, verify, then train M1."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/data2/gb/Dual-RGBT')
cpu = root / 'conda-envs/cpu-scoring/bin/python'
python = root / 'conda-envs/dual_rgbt/bin/python'
collection_relative = 'reports/E0_PAIRED_TRAIN_COLLECTION_PLAN_R1_20261031.json'
review_relative = 'reports/E0_M1_FULL_COLLECTION_VERIFIER_SOURCE_REVIEW_20261031.json'
final_relative = 'reports/E0_M1_OFFLINE_TRAINING_FINAL_20261031.json'
acceptance_relative = 'reports/E0_FULL_PAIRED_TRAIN_ACCEPTANCE_20261031.json'
output_relative = 'runs/protected_e0_offline_m1_20261031'
task = 'train_protected_e0_offline_m1_20261031'
status_path = root / 'reports/E0_M1_COLLECTION_HANDOFF_CONTROLLER_20261031.json'
assert not status_path.exists()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(root / collection_relative) == '411cb3d824c212cdd08303d6b7a6b0847904c230ae45bda73044fdbbf2ca3d9f'
assert sha(root / review_relative) == '1f00d3680f2f3bbcf57ca6f441b9b799162072a0e04b79851765cb3f0ee1e483'
collection = json.loads((root / collection_relative).read_bytes())
review = json.loads((root / review_relative).read_bytes())
for relative, digest in review['source_sha256'].items():
    assert sha(root / relative) == digest
record = dict(state='WAITING_COLLECTION', pid=os.getpid(), started=time.time(),
    poll_seconds=240, collection_plan=collection_relative,
    collection_plan_sha256=sha(root / collection_relative),
    verifier_review=review_relative, verifier_review_sha256=sha(root / review_relative),
    final_plan=final_relative, acceptance=acceptance_relative,
    training_task=task, output=output_relative, test_eligible=False)
status_path.write_text(json.dumps(record, indent=2) + '\n')
while True:
    statuses = [json.loads((root / 'runs/pipeline_tasks' / job['task'] / 'status.json').read_bytes())
                for job in collection['jobs']]
    failed = [row['task'] for row in statuses if row['state'] == 'FAILED']
    if failed:
        record.update(state='BLOCKED_COLLECTOR_FAILED', failed_tasks=failed, updated=time.time())
        status_path.write_text(json.dumps(record, indent=2) + '\n')
        raise SystemExit(1)
    if all(row['state'] == 'COMPLETE' and row['exit_code'] == 0 for row in statuses):
        break
    time.sleep(240)
record.update(state='VERIFYING_FULL_COLLECTION', updated=time.time())
status_path.write_text(json.dumps(record, indent=2) + '\n')
command = [str(python), '-B', str(root / 'tools/verify_e0_paired_train_collection.py'),
    '--draft', 'reports/E0_M1_OFFLINE_TRAINING_DRAFT_R1_20261031.json',
    '--review', review_relative, '--report', acceptance_relative, '--output-plan', final_relative]
result = subprocess.run(command, cwd=str(root), env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
if result.returncode:
    record.update(state='FAILED_ACCEPTANCE', exit_code=result.returncode, updated=time.time())
    status_path.write_text(json.dumps(record, indent=2) + '\n')
    raise SystemExit(result.returncode)
assert not (root / output_relative).exists()
assert not (root / 'runs/pipeline_tasks' / task).exists()
memory = [tuple(map(int, line.replace(',', '').split())) for line in
    subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'], universal_newlines=True).splitlines()]
gpu, used = min(memory, key=lambda row: row[1])
assert used < 14000
command = [str(python), str(root / 'scripts/run_task_four_gpu.py'),
    '--task', task, '--gpu', str(gpu), '--', str(python),
    str(root / 'scripts/train_e0_reliability.py'),
    '--plan', final_relative, '--output', output_relative]
record.update(state='TRAINING_M1', updated=time.time(), physical_gpu=str(gpu),
    gpu_memory_mib_before=used, training_command=command,
    final_plan_sha256=sha(root / final_relative),
    acceptance_sha256=sha(root / acceptance_relative))
status_path.write_text(json.dumps(record, indent=2) + '\n')
result = subprocess.run(command, cwd=str(root), env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'))
record.update(state='COMPLETE' if result.returncode == 0 else 'FAILED_M1_TRAINING',
              exit_code=result.returncode, completed=time.time())
status_path.write_text(json.dumps(record, indent=2) + '\n')
raise SystemExit(result.returncode)
