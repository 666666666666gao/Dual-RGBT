"""Identity and full-coverage checks for the registered joint-controller models."""
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_inputs import REMOTE_ROOT, REMOTE_PYTHON, groundtruth, load_manifest, sha

PREFIXES = ('executor.preview.', 'executor.quality.', 'query_gain.')
TRAINERS = {
    'joint_preview_relative_quality': 'scripts/train_candidate_joint_quality_trajectory.py',
    'signed_gain_and_same_state_difference_regression': 'scripts/train_candidate_joint_gainpair_trajectory.py',
}


def check_files(root, mapping):
    assert all(sha(root / name) == digest for name, digest in mapping.items())


def protocol(root, path):
    spec = json.loads((root / path).read_bytes())
    assert spec['status'] == 'READY'
    review = json.loads((root / spec['source_review']).read_bytes())
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['protocol_sha256'] == sha(root / path)
    assert review['source_sha256'] == spec['source_sha256']
    check_files(root, spec['source_sha256'])
    return spec


def training_metadata(root, spec, training_path):
    assert sha(root / training_path) == spec['training_plan_sha256'][training_path]
    training = json.loads((root / training_path).read_bytes())
    directory = root / 'runs' / training['runs']['formal']
    native_path = root / 'runs/pipeline_tasks' / training['tasks']['formal'] / 'status.json'
    native = json.loads(native_path.read_bytes())
    # Check terminal ownership before inspecting a potentially active checkpoint.
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    complete = json.loads((directory / 'COMPLETE.json').read_bytes())
    config = json.loads((directory / 'config.json').read_bytes())
    assert complete['status'] == 'COMPLETE' and not complete['sanity'] and not config['sanity']
    assert {key: complete[key] for key in spec['formal_counts']} == spec['formal_counts']
    assert training['expected_formal_counts'] == spec['formal_counts']
    assert training['rounds'] == 2 and training['label_stride'] == 5 and training['seed'] == spec['seed']
    assert len(set(training['sequence_names'])) == len(training['sequence_names']) == 882
    assert sorted(complete['seen_train_sequences']) == sorted(training['sequence_names'])
    assert config['run'] == training['runs']['formal'] and config['stage'] == 'candidate_c'
    assert config['plan'] == training_path and config['plan_sha256'] == sha(root / training_path)
    assert config['source_sha256'] == training['source_sha256']
    assert config['input_sha256'] == training['input_sha256']
    assert config['factory_checkpoint'] == training['factory_checkpoint']
    assert config['max_candidates'] == training['max_candidates'] in (4, 8)
    assert config['quality_rank_supervision'] == 'gap_weighted_margin'
    assert config['adapted_modules'] == list(PREFIXES)
    assert complete['selected_execution_bit_equal'] and complete['teacher_preserves_tracker_state']
    assert complete['frozen_state_tensors'] == 251 and len(complete['changed_state_tensors']) == 14
    assert native['task'] == training['tasks']['formal']
    assert native['physical_gpu'] == str(training['physical_gpu'])
    trainer = TRAINERS[training['repair']['kind']]
    assert native['command'] == [REMOTE_PYTHON, REMOTE_ROOT + '/' + trainer,
        '--plan', training_path, '--run', config['run'],
        '--sanity-complete-sha256', config['sanity_complete_sha256']]
    assert sha(directory / 'latest.pth') == complete['checkpoint_sha256']
    return training, config, complete, native_path


def tensor_witness(root, training, config, complete):
    import torch
    torch.set_num_threads(1)
    checkpoint = root / 'runs' / config['run'] / 'latest.pth'
    state = torch.load(checkpoint, map_location='cpu')
    assert state['stage'] == 'candidate_c' and state['step'] == complete['updates'] == 182182
    assert state['config'] == config
    parent_path = root / training['checkpoint']
    assert sha(parent_path) == training['input_sha256'][training['checkpoint']]
    parent = torch.load(parent_path, map_location='cpu')
    assert parent['stage'] == 'candidate_b' and parent['step'] == 6250 and not parent['config']['sanity']
    net, original = state['net'], parent['net']
    assert tuple(net) == tuple(original) and len(net) == 265
    assert all(net[name].shape == original[name].shape and net[name].dtype == original[name].dtype for name in net)
    assert all(torch.isfinite(value).all() for value in net.values())
    changed = [name for name in net if not torch.equal(net[name], original[name])]
    assert sorted(changed) == sorted(complete['changed_state_tensors'])
    assert sorted(changed) == sorted(name for name in original if name.startswith(PREFIXES))
    assert len(changed) == 14
    return dict(state_tensors=265, changed_state_tensors=changed, frozen_state_tensors=251,
        original_parent_sha256=sha(parent_path), all_state_finite=True,
        scope='CPU tensor and metadata check; no tracking accuracy or FPS measurement.')


def evaluation_plan(root, path):
    plan = json.loads((root / path).read_bytes())
    assert plan['status'] == 'READY'
    spec = protocol(root, plan['protocol'])
    assert plan['protocol_sha256'] == sha(root / plan['protocol'])
    assert plan['source_sha256'] == spec['source_sha256']
    assert all(plan['input_sha256'][k] == v for k, v in spec['input_sha256'].items())
    check_files(root, plan['input_sha256'])
    training, config, complete, _ = training_metadata(root, spec, plan['training_plan'])
    assert plan['candidate_id'] == training['runs']['formal']
    assert plan['checkpoint'] == 'runs/' + plan['candidate_id'] + '/latest.pth'
    assert plan['checkpoint_sha256'] == complete['checkpoint_sha256']
    assert plan['factory_checkpoint'] == training['factory_checkpoint']
    assert plan['max_candidates'] == training['max_candidates']
    assert plan['training_config_sha256'] == sha(root / 'runs' / config['run'] / 'config.json')
    return plan, spec, training


def run_name(plan, benchmark, split, cost, shard, shards):
    tag = str(float(cost)).replace('.', 'p')
    return plan['candidate_id'] + '_' + benchmark.lower() + '_' + split + '_cost' + tag + '_s%dof%d' % (shard, shards)


def evaluation_command(config):
    command = [REMOTE_PYTHON, REMOTE_ROOT + '/scripts/run_joint_candidate_evaluation.py']
    for key in ('plan', 'benchmark', 'split', 'task', 'run', 'cost_weight', 'shard', 'shards', 'smoke_frames'):
        command += ['--' + key.replace('_', '-'), str(config[key])]
    if config['split'] == 'test':
        command += ['--selection-sha256', config['selection_sha256']]
    return command


def smoke_admission(root, plan_path, plan, training):
    name = run_name(plan, 'LasHeR', 'train_smoke', training['cost_weight'], 0, 1)
    directory = root / 'runs' / name
    cfg = json.loads((directory / 'config.json').read_bytes())
    native = json.loads((root / 'runs/pipeline_tasks' / ('eval_' + name) / 'status.json').read_bytes())
    result = json.loads((directory / 'SMOKE.json').read_bytes())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert native['command'] == cfg['native_command'] == evaluation_command(cfg)
    assert native['task'] == cfg['task'] == result['task'] == 'eval_' + name
    assert native['physical_gpu'] == cfg['physical_gpu'] in ('0', '1', '2', '3')
    assert cfg['plan'] == plan_path and cfg['plan_sha256'] == result['plan_sha256'] == sha(root / plan_path)
    assert cfg['candidate_id'] == plan['candidate_id'] and cfg['run'] == name
    assert cfg['source_sha256'] == plan['source_sha256']
    assert cfg['checkpoint_sha256'] == result['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert (cfg['benchmark'], cfg['split'], cfg['smoke_frames'], cfg['shard'], cfg['shards']) == ('LasHeR', 'train_smoke', 11, 0, 1)
    assert cfg['cost_weight'] == training['cost_weight'] and cfg['max_candidates'] == plan['max_candidates']
    assert cfg['selection_sha256'] is None and result['smoke_only']
    assert result['frames'] == 11 and result['sequences'] == 1
    assert result['optimizer_updates'] == result['backward_calls'] == 0
    assert result['reload']['status'] == 'PASS' and result['reload']['strict_reload_box_route_bit_equal']
    assert result['reload']['strict_reload_templates_history_bit_equal']


def verify_runs(root, plan_path, manifest_path, run_names):
    plan, spec, training = evaluation_plan(root, plan_path)
    smoke_admission(root, plan_path, plan, training)
    manifest = load_manifest(root / manifest_path)
    benchmark, split = manifest['benchmark'], manifest['split']
    assert split in ('dev', 'test') and (split == 'test' or benchmark == 'LasHeR')
    assert spec['datasets'][benchmark][split] == manifest_path
    expected = spec['split_sizes'][benchmark][split]
    assert (manifest['sequence_count'], manifest['frame_count']) == (expected['sequences'], expected['frames'])
    configs, seconds, artifacts, covered = [], [], {}, []
    heavy_tokens = track_frames = 0
    assert len(run_names) == len(set(run_names)) > 0
    by_name = {row['name']: row for row in manifest['sequences']}
    for name in run_names:
        directory = root / 'runs' / name
        cfg = json.loads((directory / 'config.json').read_bytes())
        complete = json.loads((directory / 'COMPLETE.json').read_bytes())
        native_path = root / 'runs/pipeline_tasks' / cfg['task'] / 'status.json'
        native = json.loads(native_path.read_bytes())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
        assert native['command'] == cfg['native_command'] == evaluation_command(cfg)
        assert native['physical_gpu'] == cfg['physical_gpu'] in ('0', '1', '2', '3')
        assert native['task'] == complete['task'] == cfg['task'] == 'eval_' + name
        assert cfg['candidate_id'] == plan['candidate_id'] and cfg['plan'] == plan_path
        assert cfg['plan_sha256'] == complete['plan_sha256'] == sha(root / plan_path)
        assert cfg['checkpoint_sha256'] == complete['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert cfg['source_sha256'] == plan['source_sha256']
        assert cfg['benchmark'] == benchmark and cfg['split'] == split and not cfg['smoke_frames']
        assert cfg['manifest_sha256'] == sha(root / manifest_path)
        assert cfg['max_candidates'] == plan['max_candidates']
        assert cfg['total_parameters'] == complete['total_parameters'] == spec['total_parameters']
        assert not complete['smoke_only'] and complete['optimizer_updates'] == complete['backward_calls'] == 0
        assert name == run_name(plan, benchmark, split, cfg['cost_weight'], cfg['shard'], cfg['shards'])
        selected = manifest['sequences'][cfg['shard']::cfg['shards']]
        names = [row['name'] for row in selected]
        assert cfg['sequences'] == names and complete['sequences'] == len(names)
        assert cfg['split_sequences'] == expected['sequences'] and cfg['split_frames'] == expected['frames']
        assert {p.stem for p in (directory / 'predictions').glob('*.txt')} == set(names)
        assert {p.stem for p in (directory / 'traces').glob('*.csv')} == set(names)
        if benchmark == 'LasHeR':
            assert {p.stem for p in (directory / 'groundtruth').glob('*.txt')} == set(names)
        else:
            assert {p.name for p in (directory / 'groundtruth').iterdir()} == set(names)
        timing = np.load(directory / 'frame_seconds.npy')
        frames = sum(len(row['rgb']) for row in selected)
        assert timing.shape == (frames,) and np.isfinite(timing).all() and (timing > 0).all()
        assert complete['frames'] == frames and np.isclose(timing.sum(), complete['seconds'], rtol=1e-12)
        offset = 0
        for sequence in selected:
            n = sequence['name']
            visible, infrared = groundtruth(sequence, benchmark)
            prediction = np.loadtxt(directory / 'predictions' / (n + '.txt'), delimiter=',', dtype=np.float32, ndmin=2)
            assert prediction.shape == visible.shape == (len(sequence['rgb']), 4)
            assert np.isfinite(prediction).all() and (prediction[:, 2:] > 0).all()
            pairs = [('groundtruth/' + n + '.txt', visible)] if benchmark == 'LasHeR' else [
                ('groundtruth/' + n + '/visible.txt', visible), ('groundtruth/' + n + '/infrared.txt', infrared)]
            for path, gt in pairs:
                saved = np.loadtxt(directory / path, delimiter=',', dtype=np.float32, ndmin=2)
                assert np.array_equal(saved, gt)
            with (directory / 'traces' / (n + '.csv')).open(newline='') as handle:
                trace = list(csv.DictReader(handle))
            assert [int(row['frame']) for row in trace] == list(range(1, len(prediction)))
            assert np.array_equal(np.asarray([float(row['seconds_including_decode']) for row in trace]),
                                  timing[offset + 1:offset + len(prediction)])
            for row in trace:
                action, tokens = int(row['action']), int(row['auxiliary_tokens'])
                assert action in range(5) and tokens == (0 if action == 0 else 385 if action == 4 else 193)
                assert int(row['primary_tokens']) == 385 and int(row['modality']) in (0, 1)
                heavy_tokens += 385 + tokens
            track_frames += len(trace)
            offset += len(prediction)
        artifacts[native_path.relative_to(root).as_posix()] = sha(native_path)
        for path in directory.rglob('*'):
            if path.is_file():
                artifacts[path.relative_to(root).as_posix()] = sha(path)
        seconds.extend(timing.tolist())
        covered.extend(names)
        configs.append(cfg)
    first = configs[0]
    invariants = ('checkpoint_sha256', 'cost_weight', 'max_candidates', 'shards', 'selection_sha256', 'total_parameters')
    assert all(all(cfg[k] == first[k] for k in invariants) for cfg in configs)
    assert sorted(cfg['shard'] for cfg in configs) == list(range(first['shards']))
    assert len(covered) == len(set(covered)) == expected['sequences'] and set(covered) == set(by_name)
    assert len(seconds) == expected['frames']
    if split == 'dev':
        assert first['selection_sha256'] is None and first['cost_weight'] in spec['dev_cost_candidates']
    else:
        selection = selected_configuration(root, plan_path, first['selection_sha256'])
        assert selection['configurations'][benchmark] == dict(
            cost_weight=first['cost_weight'], max_candidates=first['max_candidates'])
    return plan, spec, manifest, configs, np.asarray(seconds), artifacts, heavy_tokens / track_frames


def checked_dev_report(root, plan_path, plan, spec, report_path):
    report = json.loads(report_path.read_bytes())
    assert report['status'] == 'SCORED_COMPLETE' and report['candidate_id'] == plan['candidate_id']
    assert report['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert (report['benchmark'], report['split'], report['sequences'], report['frames']) == ('LasHeR', 'dev', 97, 54646)
    assert report['evaluation_plan_sha256'] == sha(root / plan_path)
    assert report['manifest_sha256'] == spec['input_sha256'][spec['datasets']['LasHeR']['dev']]
    assert report['source_sha256'] == spec['source_sha256']
    assert report['max_candidates'] == plan['max_candidates'] and report['frozen_selection_sha256'] is None
    check_files(root, report['run_artifacts_sha256'])
    for name in report['run_names']:
        config_path = root / 'runs' / name / 'config.json'
        assert report['run_artifacts_sha256'][config_path.relative_to(root).as_posix()] == sha(config_path)
        assert json.loads(config_path.read_bytes())['cost_weight'] == report['cost_weight']
    curves_path = report_path.with_suffix('.npz')
    assert sha(curves_path) == report['curves_sha256']
    with np.load(curves_path) as curves:
        for key in ('PR', 'NPR', 'SR'):
            assert curves[key].shape == (97, 21 if key == 'SR' else 51) and np.isfinite(curves[key]).all()
            assert (curves[key] >= 0).all() and (curves[key] <= 1).all()
        assert report['PR'] == 100 * float(curves['PR'][:, 20].mean())
        assert report['NPR'] == 100 * float(curves['NPR'][:, 20].mean())
        assert report['SR'] == 100 * float(curves['SR'].mean())
    return report


def selected_configuration(root, plan_path, selection_sha):
    plan, spec, _ = evaluation_plan(root, plan_path)
    path = root / plan['frozen_selection']
    assert sha(path) == selection_sha
    selection = json.loads(path.read_bytes())
    assert selection['status'] == 'FROZEN' and selection['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert selection['evaluation_plan_sha256'] == sha(root / plan_path)
    assert selection['selection_split'] == 'LasHeR_dev97'
    assert selection['candidate_id'] == plan['candidate_id'] and selection['criterion'] == spec['selection_criterion']
    audit_path = root / selection['result_review']
    assert sha(audit_path) == selection['result_review_sha256']
    audit = json.loads(audit_path.read_bytes())
    assert audit['status'] == 'DEV_RESULT_AUDIT_PASS' and audit['blocking_issues'] == 0
    assert audit['evaluation_plan_sha256'] == sha(root / plan_path)
    assert audit['checkpoint_sha256'] == plan['checkpoint_sha256']
    reports = []
    for receipt in selection['dev_receipts']:
        report_path = root / receipt['path']
        assert sha(report_path) == receipt['sha256']
        report = checked_dev_report(root, plan_path, plan, spec, report_path)
        assert audit['dev_reports_sha256'][receipt['path']] == receipt['sha256']
        assert all(np.isfinite(report[k]) for k in ('PR', 'NPR', 'SR'))
        reports.append(report)
    assert sorted(r['cost_weight'] for r in reports) == spec['dev_cost_candidates']
    assert set(audit['dev_reports_sha256']) == {r['path'] for r in selection['dev_receipts']}
    best = max(reports, key=lambda r: (r['SR'], r['PR'], r['NPR'], -r['mean_heavy_tokens'], r['cost_weight']))
    reference = json.loads((root / spec['reference_decision']).read_bytes())
    assert best['SR'] > reference['reference_dev']['SR']
    config = dict(cost_weight=best['cost_weight'], max_candidates=plan['max_candidates'])
    assert selection['configurations'] == {name: config for name in spec['datasets']}
    return selection
