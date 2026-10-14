"""After E1 releases both cards, evaluate full-C E0 on all three benchmarks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from complete_pipeline import ROOT, REMOTE, SCP, launch, remote, wait_until
from benchmark_inputs import evaluation_command, sha
from complete_candidate_benchmarks import save, wait_owned, collect, checked_cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--toolkit', required=True)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_bytes())
    review = json.loads((ROOT / plan['source_review']).read_bytes())
    assert plan['status'] == 'READY'
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(plan_path) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == digest for name, digest in plan['local_input_sha256'].items())
    evaluation = json.loads((ROOT / plan['evaluation_plan']).read_bytes())
    identity = evaluation['models']['e0']
    status_path = ROOT / plan['status_path']
    assert not status_path.exists()
    private = ROOT / plan['private_directory']
    private.mkdir(parents=True, exist_ok=False)
    outputs = ROOT / plan['metrics_directory']
    outputs.mkdir(parents=True, exist_ok=False)
    state = dict(state='WAIT_E1_PIPELINE', local_pid=os.getpid(), plan_sha256=sha(plan_path), tasks=[])
    deadline = plan['first_dependency_observation_epoch']
    while True:
        state.update(waiting_for='candidate_completion_controller', next_observation_epoch=deadline)
        save(status_path, state)
        wait_until(deadline)
        prior_bytes = (ROOT / plan['prior_status']).read_bytes()
        prior = json.loads(prior_bytes)
        assert prior['local_pid'] == plan['prior_owner_pid']
        if prior['state'] == 'COMPLETE_E1_THREE_BENCHMARKS':
            for benchmark in evaluation['datasets']:
                assert prior['full_test'][benchmark]['sequences'] == evaluation['split_sizes'][benchmark]['test']['sequences']
                assert prior['full_test'][benchmark]['frames'] == evaluation['split_sizes'][benchmark]['test']['frames']
            (private / 'e1_terminal.json').write_bytes(prior_bytes)
            break
        owner = subprocess.run(['powershell', '-NoProfile', '-Command',
            'Get-CimInstance Win32_Process -Filter "ProcessId=%d" | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress' % plan['prior_owner_pid']],
            capture_output=True, check=True, timeout=30)
        (private / 'latest_dependency_owner.stdout.bin').write_bytes(owner.stdout)
        (private / 'latest_dependency_owner.stderr.bin').write_bytes(owner.stderr)
        owner = json.loads(owner.stdout.decode('utf-8-sig'))
        assert owner['ProcessId'] == plan['prior_owner_pid']
        assert 'scripts/complete_candidate_benchmarks.py' in owner['CommandLine']
        if 'last_observation' in prior:
            native = prior['last_observation']['native']
            assert native['state'] in ('RUNNING', 'COMPLETE')
            assert native['state'] == 'RUNNING' or native['exit_code'] == 0
        deadline = max(time.time() + 240, prior['next_observation_epoch'] + 120)

    dependency = remote('''import hashlib,json
from pathlib import Path
r=Path(%r)
tasks=%r
native=[json.loads((r/'runs/pipeline_tasks'/task['task']/'status.json').read_bytes()) for task in tasks]
assert all(row['state']=='COMPLETE' and row['exit_code']==0 and row['pid']==task['pid'] for row,task in zip(native,tasks))
integrity=json.loads((r/%r).read_bytes())
assert integrity['status']=='PASS' and integrity['steps']==182182 and integrity['tracked_frames']==917106
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=sha(r/%r)
assert integrity['checkpoint_sha256']==digest
assert all(sha(r/name)==value for name,value in %r.items())
assert all(sha(r/name)==value for name,value in %r.items())
assert sha(Path('/home/gaob/CADTrack_ep0014.pth.tar'))==%r
(r/%r).mkdir(parents=True,exist_ok=False)
print(json.dumps(dict(checkpoint_sha256=digest,native=native,integrity=integrity)))
''' % (REMOTE, prior['tasks'], identity['training_integrity'], identity['formal_checkpoint'],
       evaluation['source_sha256'], evaluation['input_sha256'], evaluation['protected_checkpoint_sha256'],plan['metrics_directory']))
    (private / 'dependency_admission.json').write_bytes((json.dumps(dependency,indent=2)+'\n').encode())
    checkpoint_sha = dependency['checkpoint_sha256']

    def track(benchmark, split, cost, gpu, shard=0, shards=1, selection_sha=None):
        name = 'reference_e0_' + benchmark.lower() + '_' + split + '_cost' + str(cost).replace('.', 'p') + '_shard' + str(shard) + '_20261015'
        values = dict(plan=plan['evaluation_plan'],variant='e0',benchmark=benchmark,split=split,
            checkpoint_sha256=checkpoint_sha,task='track_'+name,run=name,cost_weight=cost,
            max_candidates=0,shard=shard,shards=shards,smoke_frames=0,selection_sha256=selection_sha)
        task = launch(values['task'],gpu,evaluation_command(values)[1:])
        task.update(run=name,values=values)
        state['tasks'].append(task)
        return task

    def finish(tasks, benchmark, split, output, seconds_per_frame):
        paths = []
        for task in tasks:
            values = task['values']
            manifest = json.loads((ROOT / evaluation['datasets'][benchmark][split]).read_bytes())
            total = sum(len(row['rgb']) for row in manifest['sequences'][values['shard']::values['shards']])
            first = max(180,task['launched']+total*seconds_per_frame-300-time.time())
            native = wait_owned(task,first,total,False,state,status_path,plan['inference_max_observation_gap_seconds'])
            paths.append(collect(task['run'],native,private))
        checked_cpu([sys.executable,str(ROOT/'tools/score_benchmark_tracking.py'),'--plan',plan['evaluation_plan'],
            '--manifest',str(ROOT/evaluation['datasets'][benchmark][split]),'--toolkit',args.toolkit,'--runs']+
            [str(path) for path in paths]+['--output',str(output)],private,output.stem)
        return json.loads(output.read_bytes())

    state['state'] = 'FULL_E0_DEV97'
    dev_paths, dev = [], []
    for group in ([(0,0.),(1,.05)],[(0,.15)]):
        jobs = [(track('LasHeR','dev',cost,gpu),cost) for gpu,cost in group]
        for task,cost in jobs:
            output = outputs/('e0_dev_cost'+str(cost).replace('.','p')+'.json')
            dev.append(finish([task],'LasHeR','dev',output,plan['initial_inference_seconds_per_frame']))
            dev_paths.append(output)
    checked_cpu([sys.executable,str(ROOT/'tools/freeze_benchmark_selection.py'),'--plan',plan['evaluation_plan'],
        '--variant','e0','--manifest',str(ROOT/evaluation['datasets']['LasHeR']['dev']),'--toolkit',args.toolkit,
        '--dev-reports']+[str(path) for path in dev_paths],private,'freeze')
    selection_path = ROOT/identity['frozen_selection']
    selection_sha = sha(selection_path)
    selection = json.loads(selection_path.read_bytes())
    for path in [selection_path]+dev_paths:
        result = subprocess.run(SCP+[str(path),'2025:'+REMOTE+'/'+str(path.relative_to(ROOT)).replace('\\','/')],
            capture_output=True,timeout=180)
        (private/(path.stem+'.freeze_scp.stdout.bin')).write_bytes(result.stdout)
        (private/(path.stem+'.freeze_scp.stderr.bin')).write_bytes(result.stderr)
        assert result.returncode==0
    state.update(state='FULL_E0_THREE_TESTS',selection_sha256=selection_sha)
    full = {}
    for benchmark in ('LasHeR','RGBT210','RGBT234'):
        cost = selection['configurations'][benchmark]['cost_weight']
        measured = next(row['mean_ms']/1000 for row in dev if row['cost_weight']==cost)
        tasks = [track(benchmark,'test',cost,gpu,gpu,2,selection_sha) for gpu in (0,1)]
        report = finish(tasks,benchmark,'test',outputs/('e0_'+benchmark.lower()+'_full_metrics.json'),measured)
        full[benchmark] = {key:value for key,value in report.items() if key not in ('per_sequence','run_artifacts_sha256')}
    assert sha(selection_path)==selection_sha
    state.update(state='COMPLETE_E0_THREE_BENCHMARKS',waiting_for=None,full_test=full)
    save(status_path,state)


if __name__=='__main__':
    main()
