"""CPU diagnostics of an archived, completed 97-sequence dev run.

Reconstruct search geometry from saved causal predictions. This is not toolkit
scoring, alternative-action gain evaluation, or a neural experiment.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ACTION_NAMES = ('STOP', 'PRIMARY', 'SECONDARY', 'PREVIEW', 'EXPAND')


def summarize(mask, values):
    count = int(mask.sum())
    result = dict(frames=count,
                  action_counts=[int(((values['action'] == a) & mask).sum()) for a in range(5)],
                  modality_counts=[int(((values['modality'] == m) & mask).sum()) for m in range(2)])
    for field in ('raw_iou', 'raw_centre_error_px', 'confidence', 'heavy_tokens'):
        result['mean_' + field] = float(values[field][mask].mean()) if count else None
    return result


def analyse_sequence(run, name):
    prediction = np.loadtxt(run / 'predictions' / (name + '.txt'), delimiter=',',
                            dtype=np.float32, ndmin=2)
    target = np.loadtxt(run / 'groundtruth' / (name + '.txt'), delimiter=',',
                       dtype=np.float32, ndmin=2)
    assert prediction.shape == target.shape and prediction.shape[1] == 4, name
    assert np.isfinite(prediction).all() and np.isfinite(target).all(), name
    assert (prediction[:, 2:] > 0).all() and np.array_equal(prediction[0], target[0]), name
    with (run / 'traces' / (name + '.csv')).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r['frame']) for r in rows] == list(range(1, len(target))), name
    action = np.array([int(r['action']) for r in rows])
    modality = np.array([int(r['modality']) for r in rows])
    confidence = np.array([float(r['confidence']) for r in rows])
    primary = np.array([int(r['primary_tokens']) for r in rows])
    auxiliary = np.array([int(r['auxiliary_tokens']) for r in rows])
    assert np.isin(action, range(5)).all() and np.isin(modality, (0, 1)).all(), name
    assert np.isfinite(confidence).all() and (primary == 385).all(), name
    assert np.array_equal(auxiliary, np.where(action == 0, 0, np.where(action == 4, 385, 193))), name
    seconds = np.array([float(r['seconds_including_decode']) for r in rows])
    assert np.isfinite(seconds).all() and (seconds > 0).all(), name

    # Match crop_pair's float32 previous box, sqrt(float(w*h))*4, and returned
    # float32 geometry. Six-decimal saved predictions make this approximate.
    geometry = []
    for box in prediction[:-1]:
        centre = box[:2] + box[2:] / 2
        side = math.sqrt(float(box[2] * box[3])) * 4.
        left, top = centre - side / 2
        geometry.append((left, top, side))
    geometry = np.asarray(geometry, dtype=np.float32)
    target = target[1:]
    prediction = prediction[1:]
    valid = (target[:, 2:] > 0).all(1)
    centre = target[:, :2] + target[:, 2:] / 2
    relative = (centre - geometry[:, :2]) / geometry[:, 2:]
    inside = ((relative >= 0) & (relative <= 1)).all(1)
    intersection = np.maximum(0, np.minimum(prediction[:, :2] + prediction[:, 2:],
                                           target[:, :2] + target[:, 2:]) -
                              np.maximum(prediction[:, :2], target[:, :2])).prod(1)
    union = prediction[:, 2:].prod(1) + target[:, 2:].prod(1) - intersection
    iou = np.full(len(target), np.nan)
    iou[valid] = intersection[valid] / union[valid]
    centre_error = np.linalg.norm(prediction[:, :2] + prediction[:, 2:] / 2 - centre, axis=1)
    values = dict(valid=valid, inside=inside, action=action, modality=modality,
                  confidence=confidence, heavy_tokens=primary + auxiliary,
                  raw_iou=iou, raw_centre_error_px=centre_error)
    # Count only adjacent valid non-initialization frames in this sequence.
    reentry = np.zeros(len(target), dtype=bool)
    reentry[1:] = valid[:-1] & valid[1:] & ~inside[:-1] & inside[1:]
    values['roi_reentry'] = reentry
    input_hashes = {directory: hashlib.sha256((run / directory / (name + suffix)).read_bytes()).hexdigest()
                    for directory, suffix in (('predictions', '.txt'), ('groundtruth', '.txt'), ('traces', '.csv'))}
    result = dict(sequence=name, frames=len(target) + 1, tracked_frames=len(target),
                  input_sha256=input_hashes,
                  invalid_gt_frames=int((~valid).sum()),
                  all_valid=summarize(valid, values),
                  centre_inside=summarize(valid & inside, values),
                  centre_outside=summarize(valid & ~inside, values),
                  adjacent_roi_reentry=summarize(reentry, values))
    return result, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='Local archived complete dev directory.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    complete = json.loads((run / 'COMPLETE.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    assert config['split'] == 'dev' and config['smoke_frames'] == 0
    assert config['shards'] == 1 and config['shard'] == 0
    names = config['sequences']
    assert len(names) == len(set(names)) == config['split_sequences'] == complete['sequences'] == 97
    assert config['split_frames'] == complete['frames'] == 54646 and not complete['smoke_only']
    for directory, suffix in (('predictions', '*.txt'), ('groundtruth', '*.txt'), ('traces', '*.csv')):
        assert {p.stem for p in (run / directory).glob(suffix)} == set(names), directory
    results, chunks = [], []
    for name in names:
        result, values = analyse_sequence(run, name)
        results.append(result)
        chunks.append(values)
    assert sum(r['frames'] for r in results) == 54646
    combined = {field: np.concatenate([v[field] for v in chunks]) for field in chunks[0]}
    valid, inside = combined['valid'], combined['inside']
    report = dict(scope='Completed full-dev causal trace diagnostics only; not PR/NPR/SR or action gains.',
                  sequences=97, frames=54646, tracked_frames=len(valid),
                  checkpoint_sha256=config['checkpoint_sha256'], cost_weight=config['cost_weight'],
                  input_run=str(run.resolve()),
                  input_config_sha256=hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
                  input_complete_sha256=hashlib.sha256((run / 'COMPLETE.json').read_bytes()).hexdigest(),
                  action_names=ACTION_NAMES, modality_names=['RGB', 'TIR'],
                  all_tracked_routes=dict(
                      action_counts=[int((combined['action'] == a).sum()) for a in range(5)],
                      modality_counts=[int((combined['modality'] == m).sum()) for m in range(2)],
                      mean_heavy_tokens=float(combined['heavy_tokens'].mean())),
                  invalid_gt_frames=int((~valid).sum()),
                  all_valid=summarize(valid, combined),
                  centre_inside=summarize(valid & inside, combined),
                  centre_outside=summarize(valid & ~inside, combined),
                  adjacent_roi_reentry=summarize(combined['roi_reentry'], combined),
                  per_sequence=results,
                  limitations=[
                      'Geometry reconstructed from six-decimal saved predictions is approximate.',
                      'Valid GT means positive width/height; differs from original toolkit validity.',
                      'Raw continuous IoU/errors are frame-weighted, without toolkit rounding or replacement.',
                      'Frame zero is excluded; empty diagnostic strata have null means.',
                      'ROI centre reentry does not mean accurate tracking recovery.',
                      'Only selected final boxes are saved; no counterfactual auxiliary gains can be inferred.',
                      'EXPAND still encodes the same source-image search region; correlations are not causal.'
                  ])
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'per_sequence'}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
