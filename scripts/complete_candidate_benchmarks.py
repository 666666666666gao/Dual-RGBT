"""Wait for the existing owner, then complete E1 C, dev freeze and all3 tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from complete_pipeline import ROOT, REMOTE, SCP, launch, remote, wait_until
from benchmark_inputs import evaluation_command, sha


def save(path, state):
    path.write_bytes((json.dumps(state, indent=2) + '\n').encode())
    print(json.dumps(state), flush=True)


def wait_owned(task, first_delay, total, training, state, status_path, max_gap):
    deadline = time.time() + first_delay
    while True:
        state.update(waiting_for=task, next_observation_epoch=deadline)
        save(status_path, state)
        wait_until(deadline)
        code = '''import csv,json,math,time
from collections import deque
from pathlib import Path
r=Path(%r)
native=json.loads((r/'runs/pipeline_tasks'/%r/'status.json').read_bytes())
result=dict(native=native,observed=time.time())
if native['state']=='RUNNING':
 directory=r/'runs'/%r
 if %r:
  with (directory/'trajectory_steps.csv').open(newline='') as handle:
   rows=list(deque(csv.DictReader(handle),maxlen=100))
  assert rows and all(math.isfinite(float(row[key])) for row in rows for key in ('loss','quality_loss','gain_loss'))
  result.update(done=int(rows[-1]['step']),last_row=rows[-1],window=len(rows),
   mean_losses={key:sum(float(row[key]) for row in rows)/len(rows) for key in ('loss','quality_loss','gain_loss')})
 else:
  rows=json.loads((directory/'progress.json').read_bytes())
  result.update(done=sum(row['frames'] for row in rows),completed_sequences=len(rows))
print(json.dumps(result))
''' % (REMOTE, task['task'], task['run'], training)
        observed = remote(code)
        state['last_observation'] = observed
        native = observed['native']
        assert native['task'] == task['task'] and native['physical_gpu'] == str(task['gpu'])
        assert native['pid'] == task['pid']
        save(status_path, state)
        assert native['state'] in ('RUNNING', 'COMPLETE', 'FAILED')
        if native['state'] != 'RUNNING':
            assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
            return native
        assert 0 < observed['done'] <= total
        elapsed = observed['observed'] - native['started']
        remaining = elapsed * (total - observed['done']) / observed['done']
        deadline = time.time() + max(180, min(max_gap, remaining - 300))


def collect(run, native, private):
    code = '''import json
from pathlib import Path
r=Path(%r)
native=r/'runs/pipeline_tasks'/%r/'status.json'
assert json.loads(native.read_bytes())==%r
assert %r['state']=='COMPLETE' and %r['exit_code']==0
(r/'runs'/%r/'native_task.json').write_bytes(native.read_bytes())
print(json.dumps(dict(status='NATIVE_RECEIPT_COPIED')))
''' % (REMOTE, native['task'], native, native, native, run)
    remote(code)
    destination = ROOT / 'runs' / run
    assert not destination.exists()
    result = subprocess.run(SCP + ['-r', '2025:' + REMOTE + '/runs/' + run, str(destination)],
                            capture_output=True, timeout=600)
    (private / (run + '.scp.stdout.bin')).write_bytes(result.stdout)
    (private / (run + '.scp.stderr.bin')).write_bytes(result.stderr)
    (private / (run + '.scp.transport.json')).write_text(json.dumps(dict(exit_code=result.returncode, completed=time.time())))
    assert result.returncode == 0
    assert json.loads((destination / 'native_task.json').read_bytes()) == native
    return destination


def checked_cpu(command, private, label):
    result = subprocess.run(command, cwd=str(ROOT), capture_output=True)
    (private / (label + '.stdout.bin')).write_bytes(result.stdout)
    (private / (label + '.stderr.bin')).write_bytes(result.stderr)
    (private / (label + '.transport.json')).write_text(json.dumps(dict(exit_code=result.returncode, completed=time.time())))
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--toolkit', required=True)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_bytes())
    assert plan['status'] == 'READY'
    review = json.loads((ROOT / plan['source_review']).read_bytes())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(plan_path) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == digest for name, digest in plan['local_input_sha256'].items())
    evaluation = json.loads((ROOT / plan['evaluation_plan']).read_bytes())
    identity = evaluation['models']['e1']
    status_path = ROOT / plan['status_path']
    assert not status_path.exists()
    private = ROOT / plan['private_directory']
    private.mkdir(parents=True, exist_ok=False)
    outputs = ROOT / plan['metrics_directory']
    outputs.mkdir(parents=True, exist_ok=False)
    state = dict(state='WAIT_ORIGINAL_PIPELINE', local_pid=os.getpid(),
        plan_sha256=sha(plan_path), tasks=[], goal='Complete E1 882x2 C and full three-benchmark results; target remains unmet.')
    deadline = plan['first_dependency_observation_epoch']
    while True:
        state.update(waiting_for='original_all_train_pipeline', next_observation_epoch=deadline)
        save(status_path, state)
        wait_until(deadline)
        prior_bytes = (ROOT / plan['original_status']).read_bytes()
        prior = json.loads(prior_bytes)
        assert prior['local_pid'] == plan['original_owner_pid']
        assert prior['state'] != 'FAILED'
        observation = prior['last_observation']
        assert observation['state'] in ('RUNNING', 'COMPLETE')
        assert observation['state'] == 'RUNNING' or observation['exit_code'] == 0
        state['original_state'] = prior['state']
        if prior['state'] == 'COMPLETE_ALL_TRAIN_PIPELINE':
            assert prior['full_test']['sequences'] == 245 and prior['full_test']['frames'] == 220703
            (private / 'original_terminal.json').write_bytes(prior_bytes)
            break
        deadline = max(time.time() + 240, prior['next_observation_epoch'] + 120)
    dependency = remote('''import hashlib,json
from pathlib import Path
r=Path(%r)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tasks=%r
native=[json.loads((r/'runs/pipeline_tasks'/task['task']/'status.json').read_bytes()) for task in tasks]
assert all(row['state']=='COMPLETE' and row['exit_code']==0 and row['pid']==task['pid'] for row,task in zip(native,tasks))
integrity=json.loads((r/'reports/all_train_integrity.json').read_bytes())
assert integrity['status']=='PASS' and integrity['steps']==182182 and integrity['tracked_frames']==917106
assert integrity['checkpoint_sha256']==%r
assert all(sha(r/name)==digest for name,digest in %r.items())
assert all(sha(r/name)==digest for name,digest in %r.items())
assert sha(Path('/home/gaob/CADTrack_ep0014.pth.tar'))==%r
(r/%r).mkdir(parents=True,exist_ok=False)
print(json.dumps(dict(native=native,integrity=integrity,legacy_frozen_selection_sha256=sha(r/'reports/frozen_test_selection.json'))))
''' % (REMOTE, prior['tasks'], prior['full_test']['checkpoint_sha256'],
       evaluation['source_sha256'], evaluation['input_sha256'], evaluation['protected_checkpoint_sha256'],plan['metrics_directory']))
    (private / 'dependency_admission.json').write_bytes((json.dumps(dependency, indent=2) + '\n').encode())
    state['state'] = 'FULL_C882'
    run = str(Path(identity['formal_checkpoint']).parent.relative_to('runs')).replace('\\', '/')
    task_name = Path(identity['formal_native_status']).parent.name
    task = launch(task_name, 1, [REMOTE + '/scripts/train_candidate_trajectory_policy.py',
        '--plan', identity['training_plan'], '--run', run,
        '--sanity-complete-sha256', identity['sanity_complete_sha256']])
    task['run'] = run
    state['tasks'].append(task)
    native = wait_owned(task, plan['first_training_observation_seconds'], evaluation['formal_counts']['updates'],
                        True, state, status_path, plan['training_max_observation_gap_seconds'])
    trained = remote('''import hashlib,json
from pathlib import Path
r=Path(%r)
identity=%r
directory=r/Path(identity['formal_checkpoint']).parent
complete=json.loads((directory/'COMPLETE.json').read_bytes())
config=json.loads((directory/'config.json').read_bytes())
training=json.loads((r/identity['training_plan']).read_bytes())
digest=hashlib.sha256((r/identity['formal_checkpoint']).read_bytes()).hexdigest()
assert complete['status']=='COMPLETE' and not complete['sanity'] and not config['sanity']
assert {key:complete[key] for key in %r}==%r
assert complete['checkpoint_sha256']==digest
assert len(complete['seen_train_sequences'])==882 and set(complete['seen_train_sequences'])==set(training['sequence_names'])
assert config['source_sha256']==training['source_sha256'] and config['input_sha256']==training['input_sha256']
assert complete['frozen_state_tensors']==257 and len(complete['changed_state_tensors'])==8
assert complete['selected_execution_bit_equal'] and complete['teacher_preserves_tracker_state']
assert complete['sampled_query_labels']==2*complete['updates']
assert complete['labelled_queries']==complete['sampled_query_labels']+complete['extra_selected_query_labels']
assert sum(complete['route_frames'].values())==complete['tracked_frames']
assert complete['sampled_family_counts']=={'%%d:%%d'%%(role,kind):(complete['updates']+3-kind)//4 for role in range(2) for kind in range(4)}
print(json.dumps(dict(complete=complete,config=config,checkpoint_sha256=digest)))
''' % (REMOTE, identity, evaluation['formal_counts'], evaluation['formal_counts']))
    (private / 'training_complete.json').write_bytes((json.dumps(dict(native=native, **trained), indent=2) + '\n').encode())
    checkpoint_sha = trained['checkpoint_sha256']

    def track(benchmark, split, cost, gpu, shard=0, shards=1, selection_sha=None):
        name = 'candidate_e1_' + benchmark.lower() + '_' + split + '_cost' + str(cost).replace('.', 'p') + '_shard' + str(shard) + '_20261015'
        values = dict(plan=plan['evaluation_plan'], variant='e1', benchmark=benchmark, split=split,
            checkpoint_sha256=checkpoint_sha, task='track_' + name, run=name, cost_weight=cost,
            max_candidates=8, shard=shard, shards=shards, smoke_frames=0, selection_sha256=selection_sha)
        launched = launch(values['task'], gpu, evaluation_command(values)[1:])
        launched['run'] = name
        launched['values'] = values
        state['tasks'].append(launched)
        return launched

    def finish(tasks, benchmark, split, output, seconds_per_frame):
        paths = []
        for task in tasks:
            values = task['values']
            manifest = json.loads((ROOT / evaluation['datasets'][benchmark][split]).read_bytes())
            total = sum(len(row['rgb']) for row in manifest['sequences'][values['shard']::values['shards']])
            first = max(180, task['launched'] + total * seconds_per_frame - 300 - time.time())
            native = wait_owned(task, first, total, False, state, status_path, plan['inference_max_observation_gap_seconds'])
            paths.append(collect(task['run'], native, private))
        checked_cpu([sys.executable, str(ROOT / 'tools/score_benchmark_tracking.py'), '--plan', plan['evaluation_plan'],
            '--manifest', str(ROOT / evaluation['datasets'][benchmark][split]), '--toolkit', args.toolkit,
            '--runs'] + [str(path) for path in paths] + ['--output', str(output)], private, output.stem)
        return json.loads(output.read_bytes())

    state['state'] = 'FULL_DEV97'
    dev_paths, dev = [], []
    for group in ([(0, 0.), (1, .05)], [(0, .15)]):
        jobs = [(track('LasHeR', 'dev', cost, gpu), cost) for gpu, cost in group]
        for task, cost in jobs:
            output = outputs / ('e1_dev_cost' + str(cost).replace('.', 'p') + '.json')
            dev.append(finish([task], 'LasHeR', 'dev', output, plan['initial_inference_seconds_per_frame']))
            dev_paths.append(output)
    checked_cpu([sys.executable, str(ROOT / 'tools/freeze_benchmark_selection.py'), '--plan', plan['evaluation_plan'],
        '--variant', 'e1', '--manifest', str(ROOT / evaluation['datasets']['LasHeR']['dev']), '--toolkit', args.toolkit,
        '--dev-reports'] + [str(path) for path in dev_paths], private, 'freeze')
    selection_path = ROOT / identity['frozen_selection']
    selection_sha = sha(selection_path)
    selection = json.loads(selection_path.read_bytes())
    receipts = [selection_path] + dev_paths
    for path in receipts:
        result = subprocess.run(SCP + [str(path), '2025:' + REMOTE + '/' + str(path.relative_to(ROOT)).replace('\\', '/')],
                                capture_output=True, timeout=180)
        (private / (path.stem + '.freeze_scp.stdout.bin')).write_bytes(result.stdout)
        (private / (path.stem + '.freeze_scp.stderr.bin')).write_bytes(result.stderr)
        assert result.returncode == 0
    state.update(state='FULL_THREE_TESTS', selection_sha256=selection_sha)
    full = {}
    for benchmark in ('LasHeR', 'RGBT210', 'RGBT234'):
        cost = selection['configurations'][benchmark]['cost_weight']
        measured = next(row['mean_ms'] / 1000 for row in dev if row['cost_weight'] == cost)
        tasks = [track(benchmark, 'test', cost, gpu, gpu, 2, selection_sha) for gpu in (0, 1)]
        report = finish(tasks, benchmark, 'test', outputs / ('e1_' + benchmark.lower() + '_full_metrics.json'), measured)
        full[benchmark] = {key:value for key,value in report.items() if key not in ('per_sequence', 'run_artifacts_sha256')}
    assert sha(selection_path) == selection_sha
    state.update(state='COMPLETE_E1_THREE_BENCHMARKS', waiting_for=None, full_test=full)
    save(status_path, state)


if __name__ == '__main__':
    main()
