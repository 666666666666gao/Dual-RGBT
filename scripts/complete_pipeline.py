"""Local continuation of the single B -> C -> complete dev -> complete test run.

Uses the existing local Python/unchanged toolkit and the existing remote conda.
Task launches are detached and recorded. Monitoring waits for estimated nodes,
then uses 240-second intervals. A failed task stops this controller.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/gaob/Dual-RGBT'
PYTHON = REMOTE + '/conda-envs/dual_rgbt/bin/python'
SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ClearAllForwardings=yes', '2025', shlex.quote(PYTHON) + ' -']
SCP = ['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ClearAllForwardings=yes']


def remote(code):
    result = subprocess.run(SSH, input=code, text=True, encoding='utf-8',
                            capture_output=True, check=True, timeout=120)
    return json.loads(result.stdout)


def wait_until(deadline):
    while time.time() < deadline:
        time.sleep(min(60, deadline - time.time()))


def wait_task(task, first_delay, campaign):
    deadline = time.time() + first_delay
    while True:
        campaign['waiting_for'] = task
        campaign['next_observation_epoch'] = deadline
        save(campaign)
        wait_until(deadline)
        status = remote("from pathlib import Path\nimport json\np=Path(%r)\nprint(p.read_text() if p.exists() else json.dumps({'state':'STARTING'}))" %
                        (REMOTE + '/runs/pipeline_tasks/' + task + '/status.json'))
        campaign['last_observation'] = status
        save(campaign)
        if status['state'] in ('COMPLETE', 'FAILED'):
            assert status['exit_code'] == 0, status
            return status
        deadline = time.time() + 240


def launch(task, gpu, argv):
    command = [PYTHON, 'scripts/run_task.py', '--task', task, '--gpu', str(gpu), '--', PYTHON] + argv
    code = """import pathlib,subprocess,json,time
r=pathlib.Path(%r)
used=int(subprocess.check_output(['nvidia-smi','-i',%r,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<500,used
assert not (r/'runs/pipeline_tasks'/%r).exists()
with (r/'reports'/(%r+'.launcher.log')).open('wb') as log:
 p=subprocess.Popen(%r,cwd=str(r),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'task':%r,'pid':p.pid,'gpu':%r,'launched':time.time()}))
""" % (REMOTE, str(gpu), task, task, command, task, gpu)
    return remote(code)


def save(campaign):
    (ROOT / 'reports/full_pipeline_status.json').write_text(json.dumps(campaign, indent=2), encoding='utf-8')
    print(json.dumps(campaign), flush=True)


def collect(run):
    destination = ROOT / 'reports/full_pipeline' / run
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    subprocess.run(SCP + ['-r', '2025:' + REMOTE + '/runs/' + run, str(destination)], check=True, timeout=300)
    return destination


def collect_training(run):
    destination = ROOT / 'reports/full_pipeline' / run
    destination.mkdir(parents=True, exist_ok=False)
    source = '2025:' + REMOTE + '/runs/' + run
    subprocess.run(SCP + [source + '/*.json', source + '/*.csv', str(destination)], check=True, timeout=300)


def score(runs, output, toolkit):
    subprocess.run([sys.executable, str(ROOT / 'scripts/score_tracking.py'), '--toolkit', toolkit,
                    '--runs'] + [str(run) for run in runs] + ['--output', str(output)], check=True)
    return json.loads(output.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b-not-before', type=float, required=True)
    parser.add_argument('--toolkit', required=True)
    args = parser.parse_args()
    assert json.loads((ROOT / 'reports/test_gt_identity.json').read_text())['toolkit_rounded_all_frame_equal']
    campaign = dict(state='WAIT_B', local_pid=os.getpid(),
                    goal='Complete training and full LasHeR metrics; targets are not yet achieved.', tasks=[])
    deadline = args.b_not_before
    while True:
        campaign.update(waiting_for='stage_b_refreshed_20261013', next_observation_epoch=deadline)
        save(campaign)
        wait_until(deadline)
        state = remote("""import pathlib,json
r=pathlib.Path(%r)
e=r/'runs/stage_b_refreshed_20261013.exit_code.txt'
c=r/'runs/stage_b_refreshed_20261013/COMPLETE.json'
print(json.dumps({'exit_code':int(e.read_text()) if e.exists() else None,'completion':json.loads(c.read_text()) if c.exists() else None}))
""" % REMOTE)
        if state['exit_code'] is not None:
            assert state['exit_code'] == 0 and state['completion'] == {'epochs': 5, 'steps': 6250}, state
            campaign['b_completion'] = state
            collect_training('stage_b_refreshed_20261013')
            break
        deadline = time.time() + 240
    policy = 'runs/stage_b_refreshed_20261013/latest.pth'
    stages = [
        ('check_b_complete', ['scripts/check_policy_checkpoint.py', '--executor',
                              'runs/stage_a_spatial_full_20261013/best.pth', '--policy', policy,
                              '--report', 'reports/b_complete_check.json'], 60),
        ('check_c_complete_input', ['scripts/train_trajectory_policy.py', '--policy', policy,
                                    '--run', 'stage_c_complete_input_sanity_20261013', '--sanity'], 60),
        ('train_c_complete', ['scripts/train_trajectory_policy.py', '--policy', policy,
                              '--run', 'stage_c_complete_20261013', '--sequences', '20', '--rounds', '2',
                              '--label-stride', '5', '--cost-weight', '0.05', '--seed', '2027'], 60 * 60)]
    for task, argv, delay in stages:
        campaign.update(state=task)
        campaign['tasks'].append(launch(task, 0, argv))
        wait_task(task, delay, campaign)
    complete = remote("from pathlib import Path\nprint((Path(%r)/'runs/stage_c_complete_20261013/COMPLETE.json').read_text())" % REMOTE)
    assert complete['steps'] == 3504 and complete['tracked_frames'] == 17708, complete
    campaign['c_completion'] = complete
    collect_training('stage_c_complete_20261013')
    checkpoint = 'runs/stage_c_complete_20261013/latest.pth'
    dev_results = []
    for group in ([('0', 0.0), ('1', 0.05)], [('0', 0.15)]):
        campaign['state'] = 'FULL_DEV_CALIBRATION'
        launched = []
        for gpu, cost in group:
            tag = str(cost).replace('.', 'p')
            run = 'dev_complete_c_cost' + tag
            task = 'track_' + run
            argv = ['scripts/run_tracking.py', '--checkpoint', checkpoint, '--run', run,
                    '--split', 'dev', '--cost-weight', str(cost)]
            campaign['tasks'].append(launch(task, gpu, argv))
            launched.append((task, run))
        first = True
        for task, run in launched:
            wait_task(task, 135 * 60 if first else 0, campaign)
            first = False
            output = ROOT / 'reports/full_pipeline' / (run + '_metrics.json')
            dev_results.append(score([collect(run)], output, args.toolkit))
    selected = max(dev_results, key=lambda row: (row['SR'], row['PR'], row['NPR'], -row['mean_ms']))
    selection = dict(checkpoint_sha256=selected['checkpoint_sha256'], cost_weight=selected['cost_weight'],
                     criterion='Highest full-dev SR, then PR, NPR, lower measured mean latency.',
                     split='dev', sequences=97, candidates=dev_results)
    selection_path = ROOT / 'reports/frozen_test_selection.json'
    selection_path.write_text(json.dumps(selection, indent=2))
    subprocess.run(SCP + [str(selection_path), '2025:' + REMOTE + '/reports/frozen_test_selection.json'], check=True, timeout=300)
    campaign['state'] = 'FULL_TEST'
    test_tasks = []
    for gpu in (0, 1):
        run = 'test_complete_c_shard' + str(gpu)
        task = 'track_' + run
        argv = ['scripts/run_tracking.py', '--checkpoint', checkpoint, '--run', run, '--split', 'test',
                '--cost-weight', str(selected['cost_weight']), '--shard', str(gpu), '--shards', '2']
        campaign['tasks'].append(launch(task, gpu, argv))
        test_tasks.append((task, run))
    test_runs = []
    first = True
    for task, run in test_tasks:
        wait_task(task, 280 * 60 if first else 0, campaign)
        first = False
        test_runs.append(collect(run))
    result = score(test_runs, ROOT / 'reports/full_pipeline/lasher_full_metrics.json', args.toolkit)
    campaign.update(state='COMPLETE_PIPELINE', full_test={key: value for key, value in result.items()
                                                        if key != 'per_sequence'}, waiting_for=None)
    save(campaign)


if __name__ == '__main__':
    main()
