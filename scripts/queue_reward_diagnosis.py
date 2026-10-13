"""Run one frozen train-only diagnostic on GPU1 after the reference test ends."""
import argparse
import hashlib
import json
import os
import subprocess
import time

from complete_pipeline import ROOT, REMOTE, SCP, remote, launch, collect_training, wait_until


def save(record):
    (ROOT / 'reports/reward_diagnostic_status.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)


def wait_task(task, delay, record):
    deadline = time.time() + delay
    while True:
        record.update(waiting_for=task, next_observation_epoch=deadline)
        save(record)
        wait_until(deadline)
        state = remote("from pathlib import Path\nprint(Path(%r).read_text())" %
                       (REMOTE + '/runs/pipeline_tasks/' + task + '/status.json'))
        record['last_observation'] = state
        save(record)
        assert state['state'] in ('RUNNING', 'COMPLETE', 'FAILED'), state
        if state['state'] != 'RUNNING':
            assert state['state'] == 'COMPLETE' and state['exit_code'] == 0, state
            return state
        deadline = time.time() + 240


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
               for name, expected in plan['source_sha256'].items())
    assert not (ROOT / 'reports/reward_diagnostic_status.json').exists()
    record = dict(state='WAIT_REFERENCE_PIPELINE', local_pid=os.getpid(), tasks=[],
                  plan_sha256=hashlib.sha256((ROOT / args.plan).read_bytes()).hexdigest(),
                  optimizer_updates=0, scope='Train-only frozen action diagnosis; no production model changes.')
    deadline = plan['not_before_epoch']
    while True:
        record.update(waiting_for='original_reference_pipeline', next_observation_epoch=deadline)
        save(record)
        wait_until(deadline)
        prior = json.loads((ROOT / 'reports/full_pipeline_status.json').read_text(encoding='utf-8'))
        assert prior.get('last_observation', {}).get('state') != 'FAILED', prior
        if prior['state'] == 'COMPLETE_PIPELINE':
            break
        # This is the known local Windows owner, not an inference progress poll.
        observed = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command',
                                  'Get-Process -Id %d -ErrorAction Stop | Out-Null' % int(prior['local_pid'])],
                                 capture_output=True, text=True)
        assert observed.returncode == 0, observed.stderr
        deadline = time.time() + 240
    checked = remote("""import pathlib,hashlib,json
r=pathlib.Path(%r)
plan=%r
assert hashlib.sha256((r/'data/manifest.json').read_bytes()).hexdigest()==plan['manifest_sha256']
assert hashlib.sha256((r/plan['input_checkpoint']).read_bytes()).hexdigest()==plan['input_checkpoint_sha256']
assert all(hashlib.sha256((r/name).read_bytes()).hexdigest()==expected for name,expected in plan['source_sha256'].items())
print(json.dumps({'status':'PASS'}))
""" % (REMOTE, plan))
    assert checked['status'] == 'PASS'
    for sanity in (True, False):
        run = 'reward_diag_input_sanity_20261014' if sanity else 'reward_diag_train20_20261014'
        task = 'check_' + run if sanity else 'diagnose_' + run
        arguments = ['scripts/diagnose_policy_rewards.py', '--plan', args.plan, '--run', run]
        if sanity:
            arguments.append('--sanity')
        record['state'] = 'SANITY' if sanity else 'TRAIN_ACTION_DIAGNOSTIC'
        record['tasks'].append(launch(task, 1, arguments))
        wait_task(task, 60 if sanity else plan['formal_first_check_seconds'], record)
        name = 'SANITY.json' if sanity else 'COMPLETE.json'
        result = remote("from pathlib import Path\nprint(Path(%r).read_text())" %
                        (REMOTE + '/runs/' + run + '/' + name))
        assert result['status'] == 'PASS' and result['optimizer_updates'] == 0
        assert result['input_checkpoint_unchanged'] and result['frozen_state_tensors'] == 265
        assert result['actual_chosen_execution_bit_equal_checks'] == result['labeled_states']
        if sanity:
            assert result['tracked_frames'] == 10 and result['labeled_states'] == 2 and result['action_pairs'] == 16
            assert result['input_checkpoint_second_load_prediction_bit_equal']
        else:
            assert result['tracked_frames'] == plan['expected_tracked_frames']
            assert result['labeled_states'] == plan['expected_labeled_states']
            assert result['action_pairs'] == plan['expected_labeled_states'] * 8
        record['sanity' if sanity else 'result'] = result
        collect_training(run)
    record.update(state='COMPLETE_DIAGNOSTIC', waiting_for=None)
    save(record)


if __name__ == '__main__':
    main()
