"""Complete all 882 train trajectories, then full dev selection and test.

Wait for the existing reference pipeline to finish. It retains sole ownership
of its current jobs. This continuation owns different run/status directories.
Training settings are frozen before the reference test results are available.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from complete_pipeline import ROOT, REMOTE, SCP, remote, launch, collect, collect_training, score, wait_until


def save(campaign):
    (ROOT / 'reports/all_train_status.json').write_text(json.dumps(campaign, indent=2), encoding='utf-8')
    print(json.dumps(campaign), flush=True)


def wait_task(task, delay, campaign, running_interval=240):
    while True:
        deadline = time.time() + delay
        campaign.update(waiting_for=task, next_observation_epoch=deadline)
        save(campaign)
        wait_until(deadline)
        status = remote("from pathlib import Path\nprint(Path(%r).read_text())" %
                        (REMOTE + '/runs/pipeline_tasks/' + task + '/status.json'))
        campaign['last_observation'] = status
        save(campaign)
        assert status['state'] in ('RUNNING', 'COMPLETE', 'FAILED'), status
        if status['state'] != 'RUNNING':
            assert status['state'] == 'COMPLETE' and status['exit_code'] == 0, status
            return status
        if task == 'train_c_all882_20261013':
            campaign['training_diagnostic'] = remote("""import csv,json,math
from pathlib import Path
from collections import deque
p=Path(%r)
with p.open(newline='') as h:
 rows=list(deque(csv.DictReader(h),maxlen=100))
assert rows
fields=('loss','quality_loss','gain_loss','negative_fraction')
assert all(math.isfinite(float(r[k])) for r in rows for k in fields)
print(json.dumps(dict(last_row=rows[-1],window_rows=len(rows),mean_losses={k:sum(float(r[k]) for r in rows)/len(rows) for k in fields})))
""" % (REMOTE + '/runs/stage_c_all882_20261013/trajectory_steps.csv'))
            save(campaign)
        delay = running_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--toolkit', required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text(encoding='utf-8'))
    assert plan['sequences'] == 882 and plan['rounds'] == 2 and plan['label_stride'] == 5
    assert plan['seed'] == 2027 and plan['cost_weight'] == .05
    assert plan['dev_cost_candidates'] == [0., .05, .15]
    assert json.loads((ROOT / 'reports/crop_runtime_check.json').read_text(encoding='utf-8'))['status'] == 'PASS'
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha
               for p, sha in plan['source_sha256'].items())
    assert not (ROOT / 'reports/all_train_status.json').exists()
    campaign = dict(state='WAIT_REFERENCE_PIPELINE', local_pid=os.getpid(), tasks=[],
                    plan_sha256=hashlib.sha256((ROOT / args.plan).read_bytes()).hexdigest(),
                    goal='All 882 train trajectories and complete metrics; research goal remains unmet.')
    deadline = plan['prior_not_before']
    while True:
        campaign.update(waiting_for='original_reference_pipeline', next_observation_epoch=deadline)
        save(campaign)
        wait_until(deadline)
        prior = json.loads((ROOT / 'reports/full_pipeline_status.json').read_text(encoding='utf-8'))
        observation = prior.get('last_observation', {})
        assert observation.get('state') != 'FAILED', observation
        if prior['state'] == 'COMPLETE_PIPELINE':
            break
        deadline = time.time() + 240
    # Preserve the reference test freeze before this run eventually replaces the
    # remote canonical selection used by the unchanged tracking test entry.
    reference = ROOT / 'reports/reference_frozen_test_selection.json'
    assert not reference.exists()
    reference.write_bytes((ROOT / 'reports/frozen_test_selection.json').read_bytes())
    checked = remote("""import json,hashlib
from pathlib import Path
r=Path(%r)
plan=%r
assert hashlib.sha256((r/'data/manifest.json').read_bytes()).hexdigest()==plan['manifest_sha256']
assert hashlib.sha256((r/plan['input_checkpoint']).read_bytes()).hexdigest()==plan['input_checkpoint_sha256']
assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==sha for p,sha in plan['source_sha256'].items())
print(json.dumps({'status':'PASS'}))
""" % (REMOTE, plan))
    assert checked['status'] == 'PASS'
    input_args = ['scripts/run_with_crop_api.py', 'train_trajectory_policy.py', '--policy',
                  plan['input_checkpoint'], '--sequences', '882', '--rounds', '2',
                  '--label-stride', '5', '--cost-weight', '0.05', '--seed', '2027']
    campaign['state'] = 'CHECK_FULL_TRAIN_INPUT'
    task = 'check_c_all882_input_20261013'
    sanity_run = 'stage_c_all882_input_sanity_20261013'
    campaign['tasks'].append(launch(task, 0, input_args + ['--run', sanity_run, '--sanity']))
    wait_task(task, 60, campaign)
    sanity = remote("from pathlib import Path\nprint(Path(%r).read_text())" %
                    (REMOTE + '/runs/' + sanity_run + '/SANITY.json'))
    assert sanity['status'] == 'PASS' and sanity['steps'] == 2 and sanity['tracked_frames'] == 10
    assert sanity['saved_state_reload_bit_equal'] and sanity['frozen_state_tensors'] == 257
    campaign['sanity'] = sanity
    run = 'stage_c_all882_20261013'
    task = 'train_c_all882_20261013'
    campaign['state'] = 'TRAIN_ALL_882'
    campaign['tasks'].append(launch(task, 0, input_args + ['--run', run]))
    # Estimated 36--45 hours from the completed 20-sequence run; no 240-second
    # completion polling throughout two days of training.
    wait_task(task, plan['train_diagnostic_interval_seconds'], campaign,
              running_interval=plan['train_diagnostic_interval_seconds'])
    complete = remote("from pathlib import Path\nprint(Path(%r).read_text())" %
                      (REMOTE + '/runs/' + run + '/COMPLETE.json'))
    assert complete['rounds'] == 2 and complete['sequences_per_round'] == 882, complete
    assert complete['steps'] == plan['expected_updates'], complete
    assert complete['tracked_frames'] == plan['expected_tracked_frames'], complete
    campaign['training_completion'] = complete
    collect_training(run)
    task = 'audit_c_all882_20261013'
    campaign['state'] = 'AUDIT_ALL_882'
    campaign['tasks'].append(launch(task, 0, ['scripts/audit_all_trajectories.py', '--plan', args.plan]))
    wait_task(task, 60, campaign)
    audit = remote("from pathlib import Path\nprint(Path(%r).read_text())" %
                   (REMOTE + '/reports/all_train_integrity.json'))
    assert audit['status'] == 'PASS' and audit['exact_supervision_keys'], audit
    campaign['training_integrity'] = audit
    subprocess.run(SCP + ['2025:' + REMOTE + '/reports/all_train_integrity.json',
                          str(ROOT / 'reports/all_train_integrity.json')], check=True, timeout=120)
    checkpoint = 'runs/' + run + '/latest.pth'
    dev_results = []
    for group in ([(0, 0.), (1, .05)], [(0, .15)]):
        campaign['state'] = 'FULL_DEV_CALIBRATION'
        launched = []
        for gpu, cost in group:
            name = 'dev_all882_cost' + str(cost).replace('.', 'p')
            task = 'track_' + name
            argv = ['scripts/run_with_crop_api.py', 'run_tracking.py', '--checkpoint', checkpoint,
                    '--run', name, '--split', 'dev', '--cost-weight', str(cost)]
            campaign['tasks'].append(launch(task, gpu, argv))
            launched.append((task, name))
        for index, (task, name) in enumerate(launched):
            wait_task(task, 135 * 60 if index == 0 else 0, campaign)
            output = ROOT / 'reports/full_pipeline' / (name + '_metrics.json')
            dev_results.append(score([collect(name)], output, args.toolkit))
    selected = max(dev_results, key=lambda r: (r['SR'], r['PR'], r['NPR'], -r['mean_ms']))
    selection = dict(checkpoint_sha256=selected['checkpoint_sha256'], cost_weight=selected['cost_weight'],
                     criterion=plan['dev_selection'], split='dev', sequences=97, candidates=dev_results,
                     training_sequences=882, crop_entry='scripts/run_with_crop_api.py')
    selection_path = ROOT / 'reports/all_train_frozen_test_selection.json'
    assert not selection_path.exists()
    selection_path.write_text(json.dumps(selection, indent=2), encoding='utf-8')
    subprocess.run(SCP + [str(selection_path), '2025:' + REMOTE + '/reports/frozen_test_selection.json'],
                   check=True, timeout=300)
    campaign['state'] = 'FULL_TEST'
    launched = []
    for gpu in (0, 1):
        name = 'test_all882_shard' + str(gpu)
        task = 'track_' + name
        argv = ['scripts/run_with_crop_api.py', 'run_tracking.py', '--checkpoint', checkpoint,
                '--run', name, '--split', 'test', '--cost-weight', str(selected['cost_weight']),
                '--shard', str(gpu), '--shards', '2']
        campaign['tasks'].append(launch(task, gpu, argv))
        launched.append((task, name))
    runs = []
    for index, (task, name) in enumerate(launched):
        wait_task(task, 280 * 60 if index == 0 else 0, campaign)
        runs.append(collect(name))
    result = score(runs, ROOT / 'reports/full_pipeline/lasher_all882_full_metrics.json', args.toolkit)
    campaign.update(state='COMPLETE_ALL_TRAIN_PIPELINE', waiting_for=None,
                    full_test={k: v for k, v in result.items() if k != 'per_sequence'})
    save(campaign)


if __name__ == '__main__':
    main()
