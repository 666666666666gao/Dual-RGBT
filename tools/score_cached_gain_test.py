"""Verify native cached-gain execution and score with the unchanged toolkit."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import groundtruth, load_manifest, sha
from cached_gain_evaluation_contract import scope
from joint_evaluation_contract import check_files, evaluation_plan
from joint_candidate_results import official_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run
    native_path = ROOT / 'runs/pipeline_tasks' / ('eval_' + args.run) / 'status.json'
    native = json.loads(native_path.read_text())
    while args.wait and native['state'] == 'RUNNING':
        time.sleep(180)
        native = json.loads(native_path.read_text())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    config = json.loads((run / 'config.json').read_text())
    complete = json.loads((run / 'COMPLETE.json').read_text())
    assert config['run'] == args.run and native['task'] == config['task'] == complete['task']
    assert native['command'] == config['native_command']
    split = config['split']
    plan_path = ROOT / config['plan']
    plan = json.loads(plan_path.read_text())
    expected_sequences, expected_frames, exploratory_test = scope(
        split, config['smoke_frames'], complete.get('exploratory_test') is True,
        plan['full_evaluation_allowed'])
    assert (complete['sequences'], complete['frames']) == (expected_sequences, expected_frames)
    assert complete['optimizer_updates'] == complete['backward_calls'] == 0
    assert sha(plan_path) == config['plan_sha256'] == complete['plan_sha256']
    parent, spec, _ = evaluation_plan(ROOT, plan['parent_evaluation_plan'])
    assert config['parent_checkpoint_sha256'] == parent['checkpoint_sha256']
    assert config['checkpoint_sha256'] == complete['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert sha(ROOT / plan['checkpoint']) == config['checkpoint_sha256']
    assert config['cost_weight'] == plan['cost_weight'] == 0.15
    assert config['max_candidates'] == parent['max_candidates'] == 8
    assert config['test_eligible'] is complete['test_eligible'] is False
    check_files(ROOT, config['source_sha256'])
    manifest_path = ROOT / spec['datasets']['LasHeR'][split]
    assert sha(manifest_path) == config['manifest_sha256']
    manifest = load_manifest(manifest_path)
    assert config['sequences'] == [s['name'] for s in manifest['sequences']]
    for sequence in manifest['sequences']:
        supplied = np.loadtxt(run / 'groundtruth' / (sequence['name'] + '.txt'), delimiter=',', dtype=np.float32, ndmin=2)
        expected, _ = groundtruth(sequence, 'LasHeR')
        assert np.array_equal(supplied, expected), sequence['name']
    if plan['mode'] == 'parent_control':
        assert complete['parent_prediction_and_route_equal']
    toolkit = ROOT / 'vendor/cadtrack/toolkit'
    toolkit_protocol = ROOT / 'reports/JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json'
    toolkit_hashes = json.loads(toolkit_protocol.read_text())['toolkit_code_sha256']
    check_files(toolkit, toolkit_hashes)
    output = ROOT / args.output
    assert not output.exists() and not output.with_suffix('.npz').exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {p.relative_to(run).as_posix(): sha(p) for p in run.rglob('*') if p.is_file()}
    result = official_metrics(toolkit, 'LasHeR', split, [run], [config], output)
    result.update(status='SCORED_COMPLETE', evaluation_class=plan['mode'], test_eligible=False,
        exploratory_test=exploratory_test,
        total_parameters=config['total_parameters'], parent_checkpoint_sha256=parent['checkpoint_sha256'],
        evaluation_plan_sha256=sha(plan_path), source_sha256=config['source_sha256'],
        run_artifacts_sha256=artifacts, native_status_sha256=sha(native_path),
        toolkit_code_sha256=toolkit_hashes, scoring_script_sha256=sha(Path(__file__)),
        curves_sha256=sha(output.with_suffix('.npz')), fps_caveat=config['fps_caveat'])
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'evaluation_class', 'PR', 'NPR', 'SR', 'fps')}))


if __name__ == '__main__':
    main()
