"""Geometry-only diagnosis of a completed dev trajectory; no neural execution.

Count target distortion from the current training loss's xyxy clamp and template
update conditions from saved traces. No counterfactual proposals are available.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analyze_full_dev_coverage import analyse_sequence, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    output = Path(args.output)
    assert not output.exists()
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    complete = json.loads((run / 'COMPLETE.json').read_text(encoding='utf-8'))
    names = config['sequences']
    assert config['split'] == 'dev' and config['smoke_frames'] == 0
    assert config['shards'] == 1 and config['shard'] == 0
    assert len(names) == len(set(names)) == config['split_sequences'] == complete['sequences'] == 97
    assert config['split_frames'] == complete['frames'] == 54646 and not complete['smoke_only']
    for directory, suffix in (('predictions', '*.txt'), ('groundtruth', '*.txt'), ('traces', '*.csv')):
        assert {path.stem for path in (run / directory).glob(suffix)} == set(names)
    per_sequence, chunks = [], []
    for name in names:
        base, values = analyse_sequence(run, name)
        prediction = np.loadtxt(run / 'predictions' / (name + '.txt'), delimiter=',', dtype=np.float32, ndmin=2)
        target = np.loadtxt(run / 'groundtruth' / (name + '.txt'), delimiter=',', dtype=np.float32, ndmin=2)[1:]
        geometry = []
        for previous in prediction[:-1]:
            centre = previous[:2] + previous[2:] / 2
            side = math.sqrt(float(previous[2] * previous[3])) * 4.
            left, top = centre - side / 2
            geometry.append((left, top, side))
        geometry = np.asarray(geometry, dtype=np.float32)
        # The current loss converts normalized xywh to xyxy, then clamps corners.
        relative_lt = (target[:, :2] - geometry[:, :2]) / geometry[:, 2:]
        relative_wh = target[:, 2:] / geometry[:, 2:]
        corners = np.concatenate([relative_lt, relative_lt + relative_wh], axis=1)
        clipped = corners.clip(0, 1)
        original_area = (corners[:, 2:] - corners[:, :2]).prod(1)
        clipped_area = (clipped[:, 2:] - clipped[:, :2]).prod(1)
        unchanged = (clipped == corners).all(1)
        collapsed = clipped_area == 0
        partial = ~unchanged & ~collapsed
        valid = values['valid']
        assert np.array_equal(unchanged | partial | collapsed, np.ones(len(target), dtype=bool))
        assert not (valid & unchanged & collapsed).any()
        assert (original_area[valid] > 0).all()
        coverage = np.zeros(len(target), dtype=np.float32)
        coverage[valid] = clipped_area[valid] / original_area[valid]
        frames = np.arange(1, len(target) + 1)
        update = (frames % 5 == 0) & (values['confidence'] >= .65)
        values.update(unchanged=unchanged, partial=partial, collapsed=collapsed,
                      target_area_retained=coverage, template_update=update,
                      update_low_iou=update & (values['raw_iou'] < .1))
        result = dict(sequence=name, frames=base['frames'], input_sha256=base['input_sha256'])
        for label, mask in [('unchanged', valid & unchanged), ('partial', valid & partial),
                            ('collapsed', valid & collapsed), ('template_update', valid & update),
                            ('update_low_iou', valid & values['update_low_iou'])]:
            result[label] = summarize(mask, values)
            result[label]['mean_target_area_retained'] = float(coverage[mask].mean()) if mask.any() else None
        per_sequence.append(result)
        chunks.append(values)
    assert sum(result['frames'] for result in per_sequence) == 54646
    combined = {field: np.concatenate([chunk[field] for chunk in chunks]) for field in chunks[0]}
    valid = combined['valid']
    groups = {}
    for label in ('unchanged', 'partial', 'collapsed', 'template_update', 'update_low_iou'):
        mask = valid & combined[label]
        groups[label] = summarize(mask, combined)
        groups[label]['mean_target_area_retained'] = float(combined['target_area_retained'][mask].mean()) if mask.any() else None
        groups[label]['centre_outside_frames'] = int((mask & ~combined['inside']).sum())
    root = Path(__file__).resolve().parents[1]
    sources = ['scripts/analyze_clipped_targets.py', 'scripts/analyze_full_dev_coverage.py',
               'dual_rgbt/data.py', 'dual_rgbt/losses.py', 'dual_rgbt/tracker.py']
    report = dict(scope='Completed dev geometry and template-update diagnostics only; not actual train labels or gains.',
                  sequences=97, frames=54646, tracked_frames=len(valid), valid_frames=int(valid.sum()),
                  invalid_frames=int((~valid).sum()), checkpoint_sha256=config['checkpoint_sha256'],
                  cost_weight=config['cost_weight'], input_run=str(run.resolve()),
                  input_config_sha256=hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
                  input_complete_sha256=hashlib.sha256((run / 'COMPLETE.json').read_bytes()).hexdigest(),
                  source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
                  groups=groups, per_sequence=per_sequence, neural_execution=False,
                  limitations=[
                      'Search geometry is approximate, reconstructed from six-decimal saved predictions.',
                      'Positive width/height validity differs from the original toolkit.',
                      'Clipped boxes are hypothetical targets at dev states; these states did not train the model.',
                      'No saved raw network boxes or counterfactual action outputs; no signed label flip can be inferred.',
                      'Selected final IoU is raw frame-weighted geometry, not official PR/NPR/SR.',
                      'Update events infer the unchanged tracker condition frame%5==0 and confidence>=0.65.',
                      'Low-IoU updates can indicate drift but do not prove the updates caused subsequent failure.'])
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'per_sequence'}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
