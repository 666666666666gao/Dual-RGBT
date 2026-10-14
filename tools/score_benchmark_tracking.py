"""Score full registered predictions with the unchanged provided RGBT toolkit."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import evaluation_command, frozen_selection, groundtruth, load_manifest, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--toolkit', required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_bytes())
    manifest = load_manifest(args.manifest)
    benchmark, split = manifest['benchmark'], manifest['split']
    assert split in ('dev', 'test') and (split == 'test' or benchmark == 'LasHeR')
    assert plan['status'] == 'READY'
    registered_manifest = plan['datasets'][benchmark][split]
    assert sha(args.manifest) == plan['input_sha256'][registered_manifest]
    size = plan['split_sizes'][benchmark][split]
    assert (manifest['sequence_count'], manifest['frame_count']) == (size['sequences'], size['frames'])
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    toolkit = Path(args.toolkit)
    assert all(sha(toolkit / name) == value for name, value in plan['toolkit_code_sha256'].items())
    if benchmark != 'LasHeR':
        assert all(sha(toolkit / name) == value for name, value in manifest['annotation_file_sha256'].items())
    runs = [Path(path) for path in args.runs]
    configs = [json.loads((run / 'config.json').read_bytes()) for run in runs]
    first = configs[0]
    assert first['variant'] in plan['models']
    assert first['checkpoint'] == plan['models'][first['variant']]['formal_checkpoint']
    assert first['plan'] == str(Path(args.plan).resolve().relative_to(ROOT)).replace('\\', '/')
    assert first['max_candidates'] == (0 if first['variant'] == 'e0' else 8)
    assert first['cost_weight'] in plan['dev_cost_candidates']
    assert isinstance(first['total_parameters'], int) and first['total_parameters'] > 0
    invariant_fields = ('variant', 'benchmark', 'split', 'checkpoint_sha256', 'cost_weight', 'max_candidates',
                        'shards', 'split_sequences', 'split_frames', 'plan_sha256', 'manifest_sha256',
                        'frozen_selection_sha256', 'total_parameters', 'selection_sha256', 'plan', 'checkpoint')
    assert all(all(config[key] == first[key] for key in invariant_fields) for config in configs)
    assert all(config['source_sha256'] == plan['source_sha256'] for config in configs)
    assert first['plan_sha256'] == sha(args.plan) and first['manifest_sha256'] == sha(args.manifest)
    assert first['benchmark'] == benchmark and first['split'] == split
    assert all(not config['smoke_frames'] for config in configs)
    assert sorted(config['shard'] for config in configs) == list(range(first['shards']))
    names = [name for config in configs for name in config['sequences']]
    by_name = {row['name']: row for row in manifest['sequences']}
    assert len(names) == len(set(names)) == first['split_sequences'] == manifest['sequence_count']
    assert set(names) == set(by_name) and first['split_frames'] == manifest['frame_count']
    assert all(config['sequences'] == [row['name'] for row in manifest['sequences'][config['shard']::config['shards']]] for config in configs)
    if split == 'test':
        assert first['frozen_selection_sha256'] == first['selection_sha256']
        selection = frozen_selection(ROOT, plan, first['variant'], first['checkpoint_sha256'], sha(args.plan), first['selection_sha256'])
        assert selection['configurations'][benchmark] == dict(cost_weight=first['cost_weight'], max_candidates=first['max_candidates'])
    else:
        assert first['frozen_selection_sha256'] is first['selection_sha256'] is None
    output = Path(args.output)
    assert not output.exists()
    sys.path.insert(0, str(toolkit))
    from rgbt import LasHeR, RGBT210, RGBT234
    from rgbt.dataset.basedataset import BaseRGBTDataet, TrackerResult
    from rgbt.metrics.metrics import MPR, MSR, PR, SR
    dataset = (LasHeR() if benchmark == 'LasHeR' and split == 'test' else
               RGBT210() if benchmark == 'RGBT210' else RGBT234() if benchmark == 'RGBT234' else None)
    if dataset is not None:
        assert set(dataset.seqs_name) == set(names)
    seconds, rows, artifacts = [], [], {}
    curves = {name: [] for name in (('PR', 'SR') if benchmark == 'RGBT210' else ('MPR', 'MSR'))}
    for run, config in zip(runs, configs):
        complete = json.loads((run / 'COMPLETE.json').read_bytes())
        native = json.loads((run / 'native_task.json').read_bytes())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0 and native['task'] == config['task']
        assert native['physical_gpu'] == config['physical_gpu']
        assert native['physical_gpu'] in ('0', '1')
        assert native['command'] == config['native_command'] == evaluation_command(config)
        for path in run.rglob('*'):
            if path.is_file():
                artifacts['runs/' + config['run'] + '/' + str(path.relative_to(run)).replace('\\', '/')] = sha(path)
        assert complete['task'] == config['task'] and not complete['smoke_only']
        assert complete['optimizer_updates'] == complete['backward_calls'] == 0
        assert complete['sequences'] == len(config['sequences'])
        assert {path.stem for path in (run / 'predictions').glob('*.txt')} == set(config['sequences'])
        frames = 0
        for name in config['sequences']:
            sequence = by_name[name]
            visible, infrared = groundtruth(sequence, benchmark)
            prediction = np.loadtxt(run / 'predictions' / (name + '.txt'), delimiter=',', dtype=np.float32, ndmin=2)
            assert prediction.shape == visible.shape == infrared.shape == (len(sequence['rgb']), 4)
            assert np.isfinite(prediction).all() and (prediction[:, 2:] > 0).all()
            if benchmark == 'LasHeR':
                saved = np.loadtxt(run / 'groundtruth' / (name + '.txt'), delimiter=',', dtype=np.float32, ndmin=2)
                assert np.array_equal(saved, visible)
                if dataset is not None:
                    assert np.array_equal(visible.round(0), np.asarray(dataset[name]))
            else:
                for modality, expected in (('visible', visible), ('infrared', infrared)):
                    saved = np.loadtxt(run / 'groundtruth' / name / (modality + '.txt'), delimiter=',', dtype=np.float32, ndmin=2)
                    assert np.array_equal(saved, expected)
                    assert np.array_equal(expected, np.asarray(dataset[name][modality], dtype=np.float32))
            frames += len(prediction)
        timing = np.load(run / 'frame_seconds.npy')
        assert timing.shape == (frames,) and np.isfinite(timing).all() and (timing > 0).all()
        assert complete['frames'] == frames and np.isclose(float(timing.sum()), complete['seconds'], rtol=1e-12)
        seconds.extend(timing.tolist())
        if benchmark != 'LasHeR':
            result = TrackerResult('Dual-RGBT-' + config['variant'], str(run / 'predictions'), config['sequences'], '', 'ltwh')
            metrics = [('PR', PR()), ('SR', SR())] if benchmark == 'RGBT210' else [('MPR', MPR()), ('MSR', MSR())]
            chunk = {}
            for key, metric in metrics:
                _, array = metric(dataset, result, config['sequences'])
                curves[key].extend(array.tolist())
                chunk[key] = array
            for index, name in enumerate(config['sequences']):
                item = dict(sequence=name)
                item.update({key:100 * float(array[index, 20] if key in ('PR', 'MPR') else array[index].mean()) for key, array in chunk.items()})
                rows.append(item)
    assert len(seconds) == manifest['frame_count']
    if benchmark == 'LasHeR':
        import score_tracking
        sys.argv = ['score_tracking.py', '--toolkit', args.toolkit, '--runs'] + args.runs + ['--output', args.output]
        score_tracking.main()
        report = json.loads(output.read_bytes())
    else:
        arrays = {key:np.asarray(values) for key, values in curves.items()}
        report = dict(benchmark=benchmark, split=split, sequences=len(names), frames=len(seconds),
                      per_sequence=rows, checkpoint_sha256=first['checkpoint_sha256'],
                      **{key:100 * float(array[:, 20].mean() if key in ('PR', 'MPR') else array.mean()) for key, array in arrays.items()})
        np.savez_compressed(output.with_suffix('.npz'), **arrays)
    report.update(benchmark=benchmark, variant=first['variant'], cost_weight=first['cost_weight'], max_candidates=first['max_candidates'],
                  evaluation_plan_sha256=sha(args.plan), manifest_sha256=sha(args.manifest),
                  total_parameters=first['total_parameters'], annotation_source=manifest['annotation_source'],
                  mean_ms=1000 * float(np.mean(seconds)), p95_ms=1000 * float(np.quantile(seconds, .95)),
                  fps=len(seconds) / sum(seconds), image_decoding_included=True, initialization_included=True,
                  model_loading_excluded=True, fps_definition='Frames divided by summed service time across all shards.',
                  frozen_selection_sha256=first['frozen_selection_sha256'])
    report['run_artifacts_sha256'] = artifacts
    report['run_names'] = [config['run'] for config in configs]
    output.write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps({key:value for key,value in report.items() if key != 'per_sequence'}, indent=2))


if __name__ == '__main__':
    main()
