"""Adopt the existing E1 C task, then use the registered dev and three-test protocol."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from complete_pipeline import ROOT, REMOTE, SCP, launch, remote
from benchmark_inputs import evaluation_command, sha
from complete_candidate_benchmarks import save, wait_owned, collect, checked_cpu


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
    (private / 'previous_local_state.json').write_bytes((ROOT / plan['previous_local_status']).read_bytes())
    adopted = plan['adopt_existing_task']
    admission = remote("""import hashlib,json
from pathlib import Path
r=Path(%r);expected=%r;identity=%r;evaluation=%r
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
native=json.loads((r/identity['formal_native_status']).read_bytes())
assert native['task']==expected['task'] and native['pid']==expected['pid'] and native['physical_gpu']=='1'
assert native['state']=='RUNNING' and native['command']==expected['command']
python='/home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python'
process=(Path('/proc')/str(native['pid'])/'cmdline').read_bytes().decode().rstrip('\\0').split('\\0')
assert process==[python,'scripts/run_task.py','--task',native['task'],'--gpu','1','--']+native['command']
directory=r/Path(identity['formal_checkpoint']).parent
assert sha(directory/'config.json')==expected['config_sha256']
config=json.loads((directory/'config.json').read_bytes())
training=json.loads((r/identity['training_plan']).read_bytes())
assert not config['sanity'] and config['plan_sha256']==sha(r/identity['training_plan'])
assert config['source_sha256']==training['source_sha256'] and config['input_sha256']==training['input_sha256']
assert all(sha(r/name)==digest for name,digest in evaluation['source_sha256'].items())
assert all(sha(r/name)==digest for name,digest in evaluation['input_sha256'].items())
assert sha(Path('/home/gaob/CADTrack_ep0014.pth.tar'))==evaluation['protected_checkpoint_sha256']
(r/%r).mkdir(parents=True,exist_ok=False)
print(json.dumps(dict(native=native,config_sha256=expected['config_sha256'],
 wrapper_process_confirmed=True,wrapper_argv_matches=True,training_restarted=False)))
""" % (REMOTE, adopted, identity, evaluation, plan['metrics_directory']))
    (private / 'adoption_admission.json').write_bytes((json.dumps(admission, indent=2) + '\n').encode())
    run = str(Path(identity['formal_checkpoint']).parent.relative_to('runs')).replace('\\', '/')
    task = dict(task=adopted['task'], pid=adopted['pid'], gpu=1,
                launched=admission['native']['started'], run=run)
    state = dict(state='FULL_C882', local_pid=os.getpid(), plan_sha256=sha(plan_path),
        tasks=[task], adopted_existing_task=True, previous_owner_pid=plan['previous_owner_pid'],
        goal='Finish existing E1 C882 and complete all three benchmarks; research target remains unmet.')
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
