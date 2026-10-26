"""Verify full dev ownership/GT, then call the existing unchanged LasHeR scorer."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import load_manifest, sha
import score_tracking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--cost-weight', required=True, type=float)
    parser.add_argument('--toolkit', required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_bytes())
    assert plan['status'] == 'READY'
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    toolkit = Path(args.toolkit)
    assert all(sha(toolkit / name) == value for name, value in plan['toolkit_code_sha256'].items())
    registered = plan['runs'][str(args.cost_weight)]
    run = ROOT / 'runs' / registered['run']
    config = json.loads((run / 'config.json').read_bytes())
    native_path = ROOT / 'runs/pipeline_tasks' / registered['task'] / 'status.json'
    native = json.loads(native_path.read_bytes())
    complete = json.loads((run / 'COMPLETE.json').read_bytes())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert native['command'] == config['native_command']
    assert native['task'] == complete['task'] == config['task'] == registered['task']
    assert native['physical_gpu'] == config['physical_gpu'] and native['physical_gpu'] in ('0', '1')
    assert config['plan_sha256'] == complete['plan_sha256'] == sha(ROOT / args.plan)
    assert config['source_sha256'] == plan['source_sha256']
    assert config['checkpoint_sha256'] == complete['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert config['checkpoint'] == plan['checkpoint'] and config['run'] == registered['run']
    assert config['cost_weight'] == args.cost_weight == registered['cost_weight']
    assert config['split'] == 'dev' and not config['smoke_frames'] and not complete['smoke_only']
    assert complete['backward_calls'] == complete['optimizer_updates'] == 0
    manifest_path = ROOT / plan['manifests']['dev']
    assert sha(manifest_path) == config['manifest_sha256'] == plan['input_sha256'][plan['manifests']['dev']]
    manifest = load_manifest(manifest_path)
    assert config['sequences'] == [row['name'] for row in manifest['sequences']]
    assert complete['sequences'] == 97 and complete['frames'] == 54646
    assert config['split_sequences'] == 97 and config['split_frames'] == 54646
    for sequence in manifest['sequences']:
        actual = np.loadtxt(run / 'groundtruth' / (sequence['name'] + '.txt'), delimiter=',', ndmin=2, dtype=np.float32)
        assert np.array_equal(actual, np.asarray(sequence['boxes'], dtype=np.float32))
    output = ROOT / registered['metrics']
    assert not output.exists() and not output.with_suffix('.npz').exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = ['score_tracking.py', '--toolkit', args.toolkit, '--runs', str(run), '--output', str(output)]
    score_tracking.main()
    report = json.loads(output.read_bytes())
    report.update(training_provenance_warnings=config['training_provenance_warnings'],
                  plan_sha256=sha(ROOT / args.plan), native_task_sha256=sha(native_path),
                  scope='Full LasHeR dev; compare only with dev. Test eligibility is a separate gate.')
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
