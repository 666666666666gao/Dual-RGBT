"""Register, smoke and launch the current parent's missing full-test control."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data2/gb/Dual-RGBT/snapshots/cached_parent_loss_20261030')
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
from joint_evaluation_contract import check_files, evaluation_plan

PLAN = 'reports/MATCHED_PARENT_FULL_TEST_PLAN_20261031.json'
REVIEW = 'reports/MATCHED_PARENT_FULL_TEST_SOURCE_REVIEW_20261031.json'
SMOKE = 'matched_parent_train_smoke_p26_20261031'
FULL = 'matched_parent_lasher_test_p26_exploratory_20261031'
SCORE = 'reports/matched_parent_test_p26_20261031/parent.json'


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'smoke', 'full', 'score'))
    args = parser.parse_args()
    if args.operation == 'prepare':
        parent_path = 'reports/joint_candidate_evaluation/joint_quality_k8_cost0p1_server25_20261027_formal/plan.json'
        parent, spec, _ = evaluation_plan(ROOT, parent_path)
        assert parent['checkpoint_sha256'] == '21e07cd1f645c6f7237e063c81cf5d60f6a14bca710e0180e8a3ca29dc827073'
        review = json.loads((ROOT / REVIEW).read_text())
        assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
        check_files(ROOT, review['source_sha256'])
        for config_path in (ROOT / 'runs').glob('*/config.json'):
            config = json.loads(config_path.read_text())
            if config.get('checkpoint_sha256') == parent['checkpoint_sha256'] and config.get('split') == 'test':
                assert not (config.get('cost_weight') == .15 and config.get('max_candidates') == 8), config_path
        mapping = 'reports/CACHED_PARENT_LOSS_SERVER26_TEST_MAPPING_20261030.json'
        train_mapping = 'reports/MATCHED_PARENT_TRAIN_SMOKE_MAPPING_20261031.json'
        write_new(ROOT / PLAN, dict(
            mode='matched_parent_full_test_control', parent_evaluation_plan=parent_path,
            checkpoint=parent['checkpoint'], checkpoint_sha256=parent['checkpoint_sha256'],
            parent_checkpoint_sha256=parent['checkpoint_sha256'], cost_weight=.15,
            full_evaluation_allowed=True, exploratory_test=True, test_eligible=False,
            runtime_provider=dict(ssh_alias='2026', project=str(ROOT)),
            source_sha256=review['source_sha256'], source_review=REVIEW, source_review_sha256=sha(ROOT / REVIEW),
            dataset_mapping_receipt=mapping, dataset_mapping_receipt_sha256=sha(ROOT / mapping),
            train_mapping_receipt=train_mapping, train_mapping_receipt_sha256=sha(ROOT / train_mapping),
            smoke_run=SMOKE, comparison_scope='Unchanged parent, K8/cost.15; exploratory full-test control.'))
        print(json.dumps(dict(status='REGISTERED', plan=PLAN, plan_sha256=sha(ROOT / PLAN))))
        return
    plan = json.loads((ROOT / PLAN).read_text())
    check_files(ROOT, plan['source_sha256'])
    if args.operation == 'score':
        native = json.loads((ROOT / 'runs/pipeline_tasks' / ('eval_' + FULL) / 'status.json').read_text())
        config = json.loads((ROOT / 'runs' / FULL / 'config.json').read_text())
        assert native['state'] == 'RUNNING' and Path('/proc', str(native['pid'])).exists()
        assert native['command'] == config['native_command']
        assert config['evaluation_class'] == plan['mode'] and config['plan_sha256'] == sha(ROOT / PLAN)
        command = [str(ROOT / 'conda-envs/cpu-scoring/bin/python'),
            str(ROOT / 'tools/score_cached_gain_test.py'), '--run', FULL, '--output', SCORE, '--wait']
        run = FULL
    else:
        run = SMOKE if args.operation == 'smoke' else FULL
        task = 'eval_' + run
        assert not (ROOT / 'runs' / run).exists()
        assert not (ROOT / 'runs/pipeline_tasks' / task).exists()
        memory = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            universal_newlines=True).splitlines()
        assert int(memory[0]) < 7000
        python = str(ROOT / 'conda-envs/dual_rgbt/bin/python')
        command = [python, str(ROOT / 'scripts/run_task_four_gpu.py'),
            '--task', task, '--gpu', '0', '--', python,
            str(ROOT / 'scripts/eval_matched_parent_test.py'),
            '--plan', PLAN, '--run', run, '--task', task,
            '--split', 'train_smoke' if args.operation == 'smoke' else 'test',
            '--smoke-frames', '11' if args.operation == 'smoke' else '0']
    receipt = ROOT / ('reports/' + run + '_' + args.operation + '_launch.json')
    assert not receipt.exists()
    with (ROOT / ('reports/' + run + '_' + args.operation + '.log')).open('x') as log:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, OPENCV_FOR_THREADS_NUM='1'), start_new_session=True)
    record = dict(pid=process.pid, command=command, run=run, plan=PLAN,
        plan_sha256=sha(ROOT / PLAN), launched=time.time(), operation=args.operation,
        exploratory_test=True, test_eligible=False, poll_seconds=180)
    write_new(receipt, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
