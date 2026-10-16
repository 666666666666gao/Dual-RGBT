"""Wait for both full three-benchmark pipelines before paired E1 timing."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from benchmark_inputs import REMOTE_PYTHON, sha
from complete_pipeline import ROOT, REMOTE, launch, remote, wait_until
from complete_candidate_benchmarks import save, wait_owned, collect


def timing_argv(values):
    argv = [REMOTE+'/scripts/compare_candidate_latency.py']
    for key in ('plan','checkpoint_sha256','selection_sha256','task','run','mode'):
        argv += ['--'+key.replace('_','-'),str(values[key])]
    return argv


def validate_result(directory, native, plan, evaluation, values, chosen):
    mode = values['mode']
    config = json.loads((directory / 'config.json').read_bytes())
    complete = json.loads((directory / 'COMPLETE.json').read_bytes())
    assert native['state'] == complete['status'] == 'COMPLETE' and native['exit_code'] == 0
    assert native['task'] == complete['task'] == config['task'] and native['physical_gpu'] == config['physical_gpu'] == '0'
    assert native['command'] == config['native_command'] == [REMOTE_PYTHON]+timing_argv(values)
    assert all(config[key] == value for key,value in values.items())
    assert config['plan_sha256'] == sha(ROOT / values['plan'])
    assert config['manifest_sha256'] == plan['input_sha256'][evaluation['datasets']['LasHeR']['dev']]
    assert config['baseline_checkpoint_sha256'] == evaluation['protected_checkpoint_sha256']
    assert config['checkpoint'] == evaluation['models']['e1']['formal_checkpoint']
    assert config['selection'] == chosen and config['run'] == directory.name
    assert config['mode'] == mode and complete['smoke_only'] == (mode == 'smoke')
    assert config['source_sha256'] == plan['source_sha256']
    assert complete['config_sha256'] == sha(directory / 'config.json')
    assert complete['timings_sha256'] == sha(directory / 'timings.csv')
    assert complete['frozen_state_equal'] and not complete['accuracy_evaluated']
    assert complete['optimizer_updates'] == complete['backward_calls'] == 0
    manifest = json.loads((ROOT / evaluation['datasets']['LasHeR']['dev']).read_bytes())
    sequences = manifest['sequences'][:1] if mode == 'smoke' else manifest['sequences']
    counts = [16 if mode == 'smoke' else len(row['rgb']) for row in sequences]
    assert config['sequences'] == [row['name'] for row in sequences]
    assert config['frames_per_sequence'] == dict(zip(config['sequences'], counts))
    with (directory / 'timings.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == complete['frames'] == sum(counts)
    offset = 0
    for sequence_index, (sequence, count) in enumerate(zip(sequences, counts)):
        for frame, row in enumerate(rows[offset:offset + count]):
            assert row['sequence'] == sequence['name'] and int(row['frame']) == frame
            assert row['first'] == ('baseline' if (sequence_index + frame) % 2 == 0 else 'e1')
            for name in ('baseline', 'e1'):
                assert 0 < float(row[name + '_tracking_seconds']) <= float(row[name + '_service_seconds'])
        offset += count
    assert (complete['sequences'], complete['warm_frames'], complete['tracked_frames']) == (len(sequences), sum(count - 9 for count in counts), sum(count - 1 for count in counts))
    for scope, selected in (('full', rows), ('warm', [row for row in rows if int(row['frame']) >= 9])):
        for arm in ('baseline', 'e1'):
            for clock in ('service', 'tracking'):
                values = np.asarray([float(row[arm + '_' + clock + '_seconds']) for row in selected])
                assert np.isfinite(values).all()
                expected = dict(frames=len(values), seconds=float(values.sum()), mean_ms=1000*float(values.mean()),
                                p95_ms=1000*float(np.quantile(values,.95)), fps=len(values)/float(values.sum()))
                assert all(np.isclose(complete[scope][arm][clock][key], value, rtol=1e-12) for key,value in expected.items())
    online = [row for row in rows if int(row['frame']) > 0]
    assert complete['action_counts'] == [sum(int(row['action']) == a for row in online) for a in range(5)]
    assert complete['modality_counts'] == [sum(int(row['modality']) == m for row in online) for m in range(2)]
    assert sum(complete['action_counts']) == sum(complete['modality_counts']) == len(online)
    assert complete['query_frames'] == sum(int(row['action']) != 0 for row in online)
    assert complete['heavy_tokens'] == sum(int(row['primary_tokens']) + int(row['auxiliary_tokens']) for row in online)
    if mode == 'smoke':
        sanity = json.loads((directory / 'SANITY.json').read_bytes())
        assert sanity['status'] == 'PASS' and sanity['strict_reload_box_route_bit_equal'] and sanity['strict_reload_templates_history_bit_equal']
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_bytes())
    review = json.loads((ROOT / plan['source_review']).read_bytes())
    assert plan['status'] == 'READY' and review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['plan_sha256'] == sha(plan_path) and review['source_sha256'] == plan['source_sha256']
    assert all(sha(ROOT / name) == value for name,value in plan['source_sha256'].items())
    assert all(sha(ROOT / name) == value for name,value in plan['local_input_sha256'].items())
    evaluation = json.loads((ROOT / plan['evaluation_plan']).read_bytes())
    status_path = ROOT / plan['status_path']
    assert not status_path.exists()
    private = ROOT / plan['private_directory']
    private.mkdir(parents=True, exist_ok=False)
    state = dict(state='WAIT_E0_THREE_BENCHMARKS', local_pid=os.getpid(), tasks=[], plan_sha256=sha(plan_path))
    deadline = plan['first_dependency_observation_epoch']
    while True:
        state.update(waiting_for='reference_completion_controller', next_observation_epoch=deadline)
        save(status_path,state)
        wait_until(deadline)
        prior_bytes = (ROOT / plan['prior_status']).read_bytes()
        prior = json.loads(prior_bytes)
        assert prior['local_pid'] == plan['prior_owner_pid']
        if prior['state'] == 'COMPLETE_E0_THREE_BENCHMARKS':
            for benchmark in evaluation['datasets']:
                assert prior['full_test'][benchmark]['sequences'] == evaluation['split_sizes'][benchmark]['test']['sequences']
                assert prior['full_test'][benchmark]['frames'] == evaluation['split_sizes'][benchmark]['test']['frames']
            (private / 'e0_terminal.json').write_bytes(prior_bytes)
            break
        owner = subprocess.run(['powershell','-NoProfile','-Command',
            'Get-CimInstance Win32_Process -Filter "ProcessId=%d" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress' % plan['prior_owner_pid']],
            capture_output=True,check=True,timeout=30)
        (private/'latest_dependency_owner.stdout.bin').write_bytes(owner.stdout)
        (private/'latest_dependency_owner.stderr.bin').write_bytes(owner.stderr)
        owner = json.loads(owner.stdout.decode('utf-8-sig'))
        assert owner['ProcessId'] == plan['prior_owner_pid'] and 'scripts/complete_reference_benchmarks_adopted.py' in owner['CommandLine']
        if 'last_observation' in prior:
            native = prior['last_observation']['native']
            assert native['state'] == 'RUNNING' or (native['state'] == 'COMPLETE' and native['exit_code'] == 0)
        deadline = max(time.time()+240, prior['next_observation_epoch']+120)
    e1 = json.loads((ROOT/plan['e1_status']).read_bytes())
    assert e1['state'] == 'COMPLETE_E1_THREE_BENCHMARKS' and e1['local_pid'] == plan['e1_owner_pid']
    dependency = remote('''import hashlib,json
from pathlib import Path
r=Path(%r)
tasks=%r
native=[json.loads((r/'runs/pipeline_tasks'/task['task']/'status.json').read_bytes()) for task in tasks]
assert all(row['state']=='COMPLETE' and row['exit_code']==0 and row['pid']==task['pid'] for row,task in zip(native,tasks))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(r/name)==value for name,value in %r.items())
assert all(sha(r/name)==value for name,value in %r.items())
print(json.dumps(dict(native=native,checkpoint_sha256=sha(r/%r),selection_sha256=sha(r/%r))))
''' % (REMOTE, prior['tasks']+e1['tasks'], plan['source_sha256'], plan['input_sha256'],
       evaluation['models']['e1']['formal_checkpoint'],evaluation['models']['e1']['frozen_selection']))
    (private/'dependency_admission.json').write_text(json.dumps(dependency,indent=2))
    checkpoint_sha,selection_sha = dependency['checkpoint_sha256'],dependency['selection_sha256']
    selection_path = ROOT/evaluation['models']['e1']['frozen_selection']
    assert sha(selection_path) == selection_sha
    selection = json.loads(selection_path.read_bytes())
    assert selection['checkpoint_sha256'] == checkpoint_sha
    chosen = selection['configurations']['LasHeR']
    smoke = None
    for mode in ('smoke','full'):
        name = 'candidate_e1_paired_latency_'+mode+'_20261015'
        values = dict(plan=args.plan,checkpoint_sha256=checkpoint_sha,selection_sha256=selection_sha,
                      task='measure_'+name,run=name,mode=mode)
        task = launch(values['task'],0,timing_argv(values))
        task.update(run=name)
        state.update(state='PAIRED_E1_'+mode.upper())
        state['tasks'].append(task)
        expected_frames = 16 if mode == 'smoke' else 54646
        projected = 180 if mode == 'smoke' else expected_frames*(smoke['full']['baseline']['service']['mean_ms']+smoke['full']['e1']['service']['mean_ms'])/1000
        native = wait_owned(task,max(180,projected-300),expected_frames,False,state,status_path,plan['max_observation_gap_seconds'])
        directory = collect(name,native,private)
        measured = validate_result(directory,native,plan,evaluation,values,chosen)
        if mode == 'smoke':
            smoke = measured
        state[mode] = measured
    state.update(state='COMPLETE_E1_PAIRED_LATENCY',waiting_for=None)
    save(status_path,state)


if __name__ == '__main__':
    main()
