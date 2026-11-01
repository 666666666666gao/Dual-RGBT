"""Submit one ready retained E0 checkpoint through the existing native wrapper."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import torch

ROOT = Path('/home/gaob/Dual-RGBT')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', required=True)
    parser.add_argument('--queue-sha256', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--gpu', required=True, choices=('0', '1', '2', '3'))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    path = ROOT / args.queue
    assert sha(path) == args.queue_sha256
    queue = json.loads(path.read_bytes())
    assert queue['stage'] == 'E0_ep20_final_five_evaluation_queue'
    assert queue['target_host'] == '2025' and queue['target_root'] == str(ROOT)
    assert queue['test_eligible'] is False and queue['exploratory_test'] is True
    matches = [job for job in queue['jobs'] if job['task'] == args.task]
    assert len(matches) == 1
    job = matches[0]
    assert queue['retained_epochs'] == [36, 37, 38, 39, 40]
    assert job['epoch'] in queue['retained_epochs']
    assert sha(queue['source_review']) == queue['review_sha256']
    review = json.loads(Path(queue['source_review']).read_bytes())
    assert review['stage'] == 'E0_finetuned_evaluation'
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['source_sha256'] == queue['source_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in queue['source_sha256'].items())
    assert sha(ROOT / 'scripts/finetune_e0_ep20.py') == queue['training_entry_sha256']
    training_run = Path(queue['training_output'])
    assert sha(training_run / 'config.json') == queue['training_config_sha256']
    config = json.loads((training_run / 'config.json').read_bytes())
    assert config['epochs'] == 40 and config['policy_frozen'] is True
    assert config['stage'] == 'E0_executor_ep20_finetune'
    log = (ROOT / 'runs/pipeline_tasks' / queue['training_task'] / 'run.log').read_text()
    completed = {int(value) for value in re.findall(r'EPOCH_COMPLETE\s+(\d+)\s+', log)}
    ready = job['epoch'] in completed
    if args.check_only:
        print(json.dumps(dict(task=args.task, epoch=job['epoch'], ready=ready,
            completed_training_epochs=sorted(completed),
            source_entries_verified=len(queue['source_sha256']), launched=False)), flush=True)
        return
    assert ready, 'The retained checkpoint epoch has not completed.'
    checkpoint = Path(job['checkpoint'])
    state = torch.load(checkpoint, map_location='cpu')
    assert state['epoch'] == job['epoch'] and state['config'] == config
    assert state['step'] == job['epoch'] * config['samples'] // config['batch_size']
    assert config['hashes']['cadtrack'] == queue['factory_sha256']
    assert config['hashes']['parent'] == queue['parent_sha256']
    del state
    assert sha(queue['factory']) == queue['factory_sha256']
    assert sha(job['manifest']) == job['manifest_sha256']
    assert not (ROOT / 'runs/pipeline_tasks' / args.task).exists()
    assert not Path(job['run']).exists() and not Path(job['score_output']).exists()
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                     '--format=csv,noheader,nounits'], text=True)
    free = {row.split(',')[0].strip(): int(row.split(',')[1])
            for row in memory.strip().splitlines()}
    assert free[args.gpu] > 4096
    python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
    checkpoint_sha256 = sha(checkpoint)
    argv = [python, str(ROOT / 'scripts/eval_e0_finetuned.py'),
        '--checkpoint', str(checkpoint), '--checkpoint-sha256', checkpoint_sha256,
        '--factory', queue['factory'], '--factory-sha256', queue['factory_sha256'],
        '--manifest', job['manifest'], '--manifest-sha256', job['manifest_sha256'],
        '--source-review', queue['source_review'], '--source-review-sha256', queue['review_sha256'],
        '--task', args.task, '--run', job['run'], '--cost-weight', str(queue['cost_weight']),
        '--smoke-frames', '0']
    command = [python, str(ROOT / 'scripts/run_task_four_gpu.py'),
               '--task', args.task, '--gpu', args.gpu, '--'] + argv
    child = subprocess.Popen(command, cwd=str(ROOT),
        env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print(json.dumps(dict(task=args.task, wrapper_pid=child.pid, gpu=args.gpu,
                          checkpoint_sha256=checkpoint_sha256, command=command)), flush=True)


if __name__ == '__main__':
    main()
