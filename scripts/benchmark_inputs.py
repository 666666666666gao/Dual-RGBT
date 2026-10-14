"""Private benchmark manifests; annotations never select an online action."""
import hashlib
import json
from pathlib import Path

import numpy as np

REMOTE_ROOT = '/home/gaob/Dual-RGBT'
REMOTE_PYTHON = REMOTE_ROOT + '/conda-envs/dual_rgbt/bin/python'


def evaluation_command(config):
    command = [REMOTE_PYTHON, REMOTE_ROOT + '/scripts/run_benchmark_tracking.py']
    for flag in ('plan', 'variant', 'benchmark', 'split', 'checkpoint_sha256', 'task', 'run',
                 'cost_weight', 'max_candidates', 'shard', 'shards', 'smoke_frames'):
        command.extend(['--' + flag.replace('_', '-'), str(config[flag])])
    if config['split'] == 'test':
        command.extend(['--selection-sha256', config['selection_sha256']])
    return command


def frozen_selection(root, plan, variant, checkpoint_sha, plan_sha, selection_sha):
    path = root / plan['models'][variant]['frozen_selection']
    assert sha(path) == selection_sha
    selection = json.loads(path.read_bytes())
    return validate_selection(root, plan, variant, checkpoint_sha, plan_sha, selection)


def validate_selection(root, plan, variant, checkpoint_sha, plan_sha, selection):
    size = plan['split_sizes']['LasHeR']['dev']
    assert selection['selection_split'] == 'LasHeR_dev97'
    assert (selection['dev_sequences'], selection['dev_frames']) == (size['sequences'], size['frames'])
    assert selection['variant'] == variant and selection['checkpoint_sha256'] == checkpoint_sha
    assert selection['evaluation_plan_sha256'] == plan_sha
    receipts = selection['dev_receipts']
    assert len(receipts) == len(plan['dev_cost_candidates'])
    reports = []
    manifest = load_manifest(root / plan['datasets']['LasHeR']['dev'])
    by_name = {row['name']: row for row in manifest['sequences']}
    for receipt in receipts:
        report_path = root / receipt['path']
        assert sha(report_path) == receipt['sha256']
        report = json.loads(report_path.read_bytes())
        assert (report['benchmark'], report['split'], report['variant']) == ('LasHeR', 'dev', variant)
        assert (report['sequences'], report['frames']) == (size['sequences'], size['frames'])
        assert report['evaluation_plan_sha256'] == plan_sha and report['checkpoint_sha256'] == checkpoint_sha
        assert report['manifest_sha256'] == plan['input_sha256'][plan['datasets']['LasHeR']['dev']]
        assert report['max_candidates'] == (0 if variant == 'e0' else 8)
        artifacts, covered, configs = {}, [], []
        assert len(report['run_names']) == len(set(report['run_names'])) > 0
        for name in report['run_names']:
            run = root / 'runs' / name
            config = json.loads((run / 'config.json').read_bytes())
            complete = json.loads((run / 'COMPLETE.json').read_bytes())
            native = json.loads((run / 'native_task.json').read_bytes())
            assert config['run'] == name and config['variant'] == variant
            assert (config['benchmark'], config['split'], config['smoke_frames']) == ('LasHeR', 'dev', 0)
            assert config['checkpoint'] == plan['models'][variant]['formal_checkpoint']
            assert config['checkpoint_sha256'] == checkpoint_sha and config['plan_sha256'] == plan_sha
            assert config['source_sha256'] == plan['source_sha256']
            assert config['manifest_sha256'] == report['manifest_sha256']
            assert config['cost_weight'] == report['cost_weight'] and config['max_candidates'] == report['max_candidates']
            assert config['total_parameters'] == report['total_parameters']
            assert (config['split_sequences'], config['split_frames']) == (size['sequences'], size['frames'])
            assert config['sequences'] == [row['name'] for row in manifest['sequences'][config['shard']::config['shards']]]
            assert config['selection_sha256'] is config['frozen_selection_sha256'] is None
            assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
            assert native['task'] == complete['task'] == config['task']
            assert native['physical_gpu'] == config['physical_gpu'] and native['physical_gpu'] in ('0', '1')
            assert native['command'] == config['native_command'] == evaluation_command(config)
            assert not complete['smoke_only'] and complete['optimizer_updates'] == complete['backward_calls'] == 0
            assert complete['sequences'] == len(config['sequences'])
            frames = sum(len(by_name[name]['rgb']) for name in config['sequences'])
            assert complete['frames'] == frames
            for folder in ('predictions', 'groundtruth'):
                assert {path.stem for path in (run / folder).glob('*.txt')} == set(config['sequences'])
            timing = np.load(run / 'frame_seconds.npy')
            assert timing.shape == (frames,) and np.isfinite(timing).all() and (timing > 0).all()
            assert np.isclose(timing.sum(), complete['seconds'], rtol=1e-12)
            artifacts.update({str(path.relative_to(root)).replace('\\', '/'): sha(path)
                              for path in run.rglob('*') if path.is_file()})
            covered.extend(config['sequences'])
            configs.append(config)
        assert artifacts == report['run_artifacts_sha256']
        assert len(covered) == len(set(covered)) == size['sequences'] and set(covered) == set(by_name)
        assert sorted(config['shard'] for config in configs) == list(range(len(configs)))
        assert all(config['shards'] == len(configs) for config in configs)
        assert np.isfinite([report[key] for key in ('SR', 'PR', 'NPR', 'mean_ms')]).all()
        reports.append(report)
    assert sorted(report['cost_weight'] for report in reports) == sorted(plan['dev_cost_candidates'])
    assert len({report['total_parameters'] for report in reports}) == 1
    best = max(reports, key=lambda report: (report['SR'], report['PR'], report['NPR'], -report['mean_ms']))
    configuration = dict(cost_weight=best['cost_weight'], max_candidates=best['max_candidates'])
    assert selection['configurations'] == {benchmark: configuration for benchmark in plan['datasets']}
    return selection


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groundtruth(sequence, benchmark):
    if benchmark == 'RGBT234':
        return tuple(np.asarray(sequence[key], dtype=np.float32) for key in ('visible_boxes', 'infrared_boxes'))
    boxes = np.asarray(sequence['boxes'], dtype=np.float32)
    return boxes, boxes


def load_manifest(path):
    manifest = json.loads(Path(path).read_bytes())
    assert manifest['benchmark'] in ('LasHeR', 'RGBT210', 'RGBT234')
    assert manifest['split'] in ('train_smoke', 'dev', 'test')
    rows = manifest['sequences']
    names = [row['name'] for row in rows]
    assert len(names) == len(set(names)) == manifest['sequence_count']
    assert sum(len(row['rgb']) for row in rows) == manifest['frame_count']
    for row in rows:
        visible, infrared = groundtruth(row, manifest['benchmark'])
        assert len(row['rgb']) == len(row['tir']) > 1
        assert visible.shape == infrared.shape == (len(row['rgb']), 4)
        assert np.isfinite(visible).all() and np.isfinite(infrared).all()
        assert np.array_equal(np.asarray(row['initial_box'], dtype=np.float32), visible[0])
        assert (visible[0, 2:] > 0).all()
    return manifest
