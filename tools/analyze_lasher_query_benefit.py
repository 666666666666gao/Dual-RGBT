"""Analyze executed queries on a fully scored LasHeR dev run; never selects a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-5  # Predictions are saved to six decimal places.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def localization(boxes, gt):
    """Continuous xywh geometry, intentionally separate from official toolkit metrics."""
    low = np.maximum(boxes[:, :2], gt[:, :2])
    high = np.minimum(boxes[:, :2] + boxes[:, 2:], gt[:, :2] + gt[:, 2:])
    intersection = np.maximum(high - low, 0).prod(axis=1)
    union = boxes[:, 2:].prod(axis=1) + gt[:, 2:].prod(axis=1) - intersection
    iou = intersection / union
    delta = boxes[:, :2] + boxes[:, 2:] / 2 - gt[:, :2] - gt[:, 2:] / 2
    return iou, np.linalg.norm(delta, axis=1), np.linalg.norm(delta / gt[:, 2:], axis=1)


def summarize(rows):
    valid = [row for row in rows if row['valid_gt']]
    queries = [row for row in valid if row['action'] != 0]
    out = dict(frames=len(rows), valid_gt_frames=len(valid), query_frames=len(queries),
               stop_frames=sum(row['action'] == 0 for row in valid),
               invalid_gt_frames=len(rows) - len(valid))
    if queries:
        gain = np.asarray([row['iou_gain'] for row in queries])
        residual = np.asarray([row['selected_gain'] - row['iou_gain'] for row in queries])
        out.update(helpful_queries=int((gain > TOLERANCE).sum()),
                   harmful_queries=int((gain < -TOLERANCE).sum()),
                   neutral_queries=int((np.abs(gain) <= TOLERANCE).sum()),
                   mean_iou_gain=float(gain.mean()),
                   mean_center_error_reduction=float(np.mean([r['center_error_reduction'] for r in queries])),
                   mean_normalized_center_error_reduction=float(np.mean([r['normalized_center_error_reduction'] for r in queries])),
                   gain_prediction_bias=float(residual.mean()), gain_prediction_mae=float(np.abs(residual).mean()),
                   negative_realized_utility_queries=sum(r['realized_utility'] < -TOLERANCE for r in queries))
    return out


def analyze(root, report_path):
    report = json.loads(report_path.read_bytes())
    assert report['status'] == 'SCORED_COMPLETE'
    assert (report['benchmark'], report['split'], report['sequences']) == ('LasHeR', 'dev', 97)
    artifacts = report['run_artifacts_sha256']
    used, rows, names = {}, [], []

    def checked(path):
        name = path.relative_to(root).as_posix()
        digest = sha(path)
        assert artifacts[name] == digest, name
        used[name] = digest
        return path

    for run in report['run_names']:
        directory = root / 'runs' / run
        cfg = json.loads(checked(directory / 'config.json').read_bytes())
        assert cfg['evidence_schema'] == 'primary_executed_box_and_selected_gain_v1'
        assert cfg['checkpoint_sha256'] == report['checkpoint_sha256']
        assert cfg['cost_weight'] == report['cost_weight']
        for name in cfg['sequences']:
            names.append(name)
            final = np.loadtxt(checked(directory / 'predictions' / (name + '.txt')), delimiter=',', ndmin=2)
            gt = np.loadtxt(checked(directory / 'groundtruth' / (name + '.txt')), delimiter=',', ndmin=2)
            with checked(directory / 'evidence' / (name + '.csv')).open(newline='') as handle:
                evidence = list(csv.DictReader(handle))
            with checked(directory / 'traces' / (name + '.csv')).open(newline='') as handle:
                trace = list(csv.DictReader(handle))
            assert final.shape == gt.shape and final.shape[1] == 4
            assert len(evidence) == len(trace) == len(final) - 1
            assert [int(row['frame']) for row in evidence] == list(range(1, len(final)))
            assert [int(row['frame']) for row in trace] == list(range(1, len(final)))
            primary = np.asarray([[float(row['primary_' + k]) for k in ('x', 'y', 'w', 'h')] for row in evidence])
            assert np.isfinite(primary).all() and (primary[:, 2:] > 0).all()
            assert np.isfinite(final).all() and (final[:, 2:] > 0).all()
            valid = np.isfinite(gt[1:]).all(axis=1) & (gt[1:, 2:] > 0).all(axis=1)
            main_metrics = localization(primary[valid], gt[1:][valid])
            final_metrics = localization(final[1:][valid], gt[1:][valid])
            metric_index = 0
            for index, (record, route) in enumerate(zip(evidence, trace)):
                action = int(route['action'])
                assert action in range(5)
                if action == 0:
                    assert np.allclose(primary[index], final[index + 1], rtol=0, atol=1e-6)
                row = dict(sequence=name, frame=index + 1, action=action,
                           modality=int(route['modality']), valid_gt=bool(valid[index]),
                           selected_gain=float(record['selected_gain']),
                           selected_token_cost=float(record['selected_token_cost']),
                           selected_utility=float(record['selected_utility']))
                assert np.isfinite([row[k] for k in ('selected_gain', 'selected_token_cost', 'selected_utility')]).all()
                assert abs(row['selected_utility'] - (row['selected_gain'] - cfg['cost_weight'] * row['selected_token_cost'])) < TOLERANCE
                if valid[index]:
                    main_iou, main_ce, main_nce = [value[metric_index] for value in main_metrics]
                    final_iou, final_ce, final_nce = [value[metric_index] for value in final_metrics]
                    row.update(primary_iou=float(main_iou), final_iou=float(final_iou),
                               iou_gain=float(final_iou - main_iou),
                               center_error_reduction=float(main_ce - final_ce),
                               normalized_center_error_reduction=float(main_nce - final_nce),
                               realized_utility=float(final_iou - main_iou - cfg['cost_weight'] * row['selected_token_cost']))
                    metric_index += 1
                rows.append(row)
    assert len(names) == len(set(names)) == 97
    assert len(rows) + len(names) == report['frames']
    result = dict(status='FULL_DEV_EXECUTED_QUERY_DIAGNOSTIC',
                  scored_report_sha256=sha(report_path), checkpoint_sha256=report['checkpoint_sha256'],
                  cost_weight=report['cost_weight'], used_artifacts_sha256=used,
                  tolerance=TOLERANCE, all_frames=summarize(rows),
                  by_action={str(a): summarize([r for r in rows if r['action'] == a]) for a in range(5)},
                  by_modality={str(m): summarize([r for r in rows if r['modality'] == m]) for m in range(2)},
                  scope='Immediate executed-query benefit on visited states. Not official PR/NPR/SR, not a STOP-only trajectory, not missed-query or future-drift evidence. Invalid GT excluded; initial frames excluded. No model selection or threshold changes.')
    return result, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists() and not output.with_suffix('.csv').exists()
    result, rows = analyze(ROOT, ROOT / args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    result['frame_csv_sha256'] = sha(output.with_suffix('.csv'))
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['all_frames']))


if __name__ == '__main__':
    main()
