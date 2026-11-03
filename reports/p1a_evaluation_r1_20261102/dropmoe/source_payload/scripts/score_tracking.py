"""Score completed predictions using the user's existing RGBT toolkit, unchanged."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--toolkit', required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.toolkit)))
    from rgbt.dataset.basedataset import BaseRGBTDataet, TrackerResult
    from rgbt.metrics.metrics import PR_LasHeR, NPR, SR_LasHeR
    from rgbt import LasHeR
    runs = [Path(path) for path in args.runs]
    configs = [json.loads((run / 'config.json').read_text()) for run in runs]
    assert all(not config['smoke_frames'] for config in configs)
    assert len({c['split'] for c in configs}) == 1
    assert len({c['checkpoint_sha256'] for c in configs}) == 1
    assert len({c['cost_weight'] for c in configs}) == 1
    assert len({c['shards'] for c in configs}) == 1
    assert len({c['split_sequences'] for c in configs}) == 1
    assert len({c['split_frames'] for c in configs}) == 1
    assert sorted(c['shard'] for c in configs) == list(range(configs[0]['shards']))
    names = [name for c in configs for name in c['sequences']]
    assert len(names) == len(set(names))
    assert len(names) == configs[0]['split_sequences']
    split = configs[0]['split']
    assert len(names) == (97 if split == 'dev' else 245)
    dataset = LasHeR() if split == 'test' else None
    if split == 'test':
        assert set(names) == set(dataset.seqs_name) and len(names) == 245
    rows, curves, seconds = [], dict(PR=[], NPR=[], SR=[]), []
    for run, config in zip(runs, configs):
        complete = json.loads((run / 'COMPLETE.json').read_text())
        assert complete['sequences'] == len(config['sequences'])
        assert {p.stem for p in (run / 'predictions').glob('*.txt')} == set(config['sequences'])
        gt = dataset if split == 'test' else BaseRGBTDataet(str(run / 'groundtruth'), config['sequences'], 'ltwh')
        result = TrackerResult('Dual-RGBT', str(run / 'predictions'), config['sequences'], '', 'ltwh')
        shard_frames = 0
        for name in config['sequences']:
            prediction = np.loadtxt(run / 'predictions' / (name + '.txt'), delimiter=',', ndmin=2)
            supplied = np.loadtxt(run / 'groundtruth' / (name + '.txt'), delimiter=',', dtype=np.float32, ndmin=2)
            assert prediction.shape == supplied.shape == (len(gt[name]), 4), name
            assert np.isfinite(prediction).all() and (prediction[:, 2:] > 0).all(), name
            assert np.array_equal(supplied.round(0), np.asarray(gt[name])), name
            shard_frames += len(prediction)
        frame_seconds = np.load(run / 'frame_seconds.npy')
        assert frame_seconds.shape == (shard_frames,)
        assert np.isfinite(frame_seconds).all() and (frame_seconds > 0).all()
        assert complete['frames'] == shard_frames
        chunk = {}
        for name, metric in [('PR', PR_LasHeR()), ('NPR', NPR()), ('SR', SR_LasHeR())]:
            _, curve = metric(gt, result, config['sequences'])
            curves[name].extend(curve.tolist())
            chunk[name] = curve
        for index, name in enumerate(config['sequences']):
            rows.append(dict(sequence=name, PR=100 * chunk['PR'][index, 20],
                             NPR=100 * chunk['NPR'][index, 20], SR=100 * chunk['SR'][index].mean()))
        seconds.extend(frame_seconds.tolist())
    assert len(seconds) == configs[0]['split_frames']
    arrays = {name: np.asarray(curve) for name, curve in curves.items()}
    report = dict(split=split, sequences=len(names), frames=len(seconds),
                  checkpoint_sha256=configs[0]['checkpoint_sha256'], cost_weight=configs[0]['cost_weight'],
                  PR=100 * float(arrays['PR'][:, 20].mean()),
                  NPR=100 * float(arrays['NPR'][:, 20].mean()), SR=100 * float(arrays['SR'].mean()),
                  mean_ms=1000 * float(np.mean(seconds)), p95_ms=1000 * float(np.quantile(seconds, .95)),
                  fps=len(seconds) / sum(seconds), image_decoding_included=True, initialization_included=True,
                  model_loading_excluded=True, fps_definition='Frames divided by summed service time across shards.',
                  toolkit=str(Path(args.toolkit).resolve()), per_sequence=rows)
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2))
    np.savez_compressed(output.with_suffix('.npz'), **arrays)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_sequence'}, indent=2))


if __name__ == '__main__':
    main()
