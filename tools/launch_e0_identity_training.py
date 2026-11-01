"""Bind the reviewed M2 training proposal to real receipts and submit one task."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/data2/gb/Dual-RGBT')
FOLDER = 'reports/e0_m2_training_p25_r2_20261101'
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]
from benchmark_inputs import sha
from train_e0_identity import verify_update_smoke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    relative = FOLDER + '/%s_draft.json' % args.phase
    draft = json.loads((ROOT / relative).read_bytes())
    assert draft['launch_allowed'] is False and draft['reviewed_draft'] == relative
    report_path = ROOT / draft['source_review']
    assert sha(report_path) == args.review_sha256
    review = json.loads(report_path.read_bytes())
    assert review['stage'] == 'M2_identity_training'
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['target_host'] == '2025' and review['target_root'] == str(ROOT)
    assert review['reviewed_drafts'][relative] == sha(ROOT / relative)
    assert review['source_sha256'] == draft['source_sha256']
    for name, digest in draft['source_sha256'].items():
        assert sha(ROOT / name) == digest
    assert sha(ROOT / draft['collection_plan']) == draft['collection_plan_sha256']
    plan = dict(draft, launch_allowed=True, source_review_sha256=args.review_sha256)
    if args.phase == 'full':
        plan.update(
            collection_acceptance_sha256=sha(ROOT / plan['collection_acceptance']),
            update_smoke_plan_sha256=sha(ROOT / plan['update_smoke_plan']),
            update_smoke_receipt_sha256=sha(ROOT / plan['update_smoke_output'] / 'COMPLETE.json'),
            update_smoke_native_sha256=sha(ROOT / 'runs/pipeline_tasks' /
                                          plan['update_smoke_task'] / 'status.json'))
        verify_update_smoke(ROOT, plan)
        acceptance = json.loads((ROOT / plan['collection_acceptance']).read_bytes())
        assert acceptance['status'] == 'FULL_M2_IDENTITY_TRAIN_COLLECTION_ACCEPTED'
        assert acceptance['collection_plan_sha256'] == plan['collection_plan_sha256']
        assert acceptance['sequences'] == 882 and acceptance['candidate_sets'] == 18711
        collection = json.loads((ROOT / plan['collection_plan']).read_bytes())
        for job in collection['jobs']:
            native = json.loads((ROOT / 'runs/pipeline_tasks' / job['task'] / 'status.json').read_bytes())
            assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert shutil.disk_usage(ROOT).free > 6 * 2**30
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in usage.strip().splitlines()}
    assert free[int(plan['assigned_gpu'])] > 6 * 1024
    output = FOLDER + '/%s_plan.json' % args.phase
    assert not (ROOT / output).exists() and not (ROOT / plan['output']).exists()
    assert not (ROOT / 'runs/pipeline_tasks' / plan['task']).exists()
    (ROOT / output).write_text(json.dumps(plan, indent=2) + '\n')
    python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
    command = [python, str(ROOT / 'scripts/run_task_four_gpu.py'),
        '--task', plan['task'], '--gpu', plan['assigned_gpu'], '--',
        python, str(ROOT / 'scripts/train_e0_identity.py'), '--plan', output,
        '--output', plan['output'], '--task', plan['task']]
    child = subprocess.Popen(command, cwd=str(ROOT),
        env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print(json.dumps(dict(task=plan['task'], output=plan['output'],
        wrapper_pid=child.pid, gpu=plan['assigned_gpu'], command=command,
        plan_sha256=sha(ROOT / output))), flush=True)


if __name__ == '__main__':
    main()
