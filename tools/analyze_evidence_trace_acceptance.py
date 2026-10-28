"""Analyze immediate query acceptance from a trace with saved box evidence.

This is a development diagnostic only. It compares the already executed
primary and query boxes on the same frame; it does not rerun the tracker,
change templates, select a test checkpoint, or claim a closed-loop metric.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def iou(box, target):
    left = np.maximum(box[:, :2], target[:, :2])
    right = np.minimum(box[:, :2] + box[:, 2:], target[:, :2] + target[:, 2:])
    intersection = np.maximum(right - left, 0).prod(1)
    union = box[:, 2:].prod(1) + target[:, 2:].prod(1) - intersection
    return intersection / np.maximum(union, 1e-12)


def box_from(row, prefix):
    return [float(row[prefix + suffix]) for suffix in ('_x', '_y', '_w', '_h')]


def analyze(run):
    run = Path(run)
    rows = []
    for trace_path in sorted((run / 'traces').glob('*.csv')):
        name = trace_path.stem
        prediction = np.loadtxt(run / 'predictions' / (name + '.txt'), delimiter=',', ndmin=2)
        target = np.loadtxt(run / 'groundtruth' / (name + '.txt'), delimiter=',', ndmin=2)
        with trace_path.open(newline='') as handle:
            trace = list(csv.DictReader(handle))
        assert len(trace) == len(prediction) - 1 == len(target) - 1
        valid = np.isfinite(target[1:]).all(1) & (target[1:, 2:] > 0).all(1)
        for index, record in enumerate(trace):
            action = int(record['action'])
            if not valid[index]:
                continue
            primary = np.asarray(box_from(record, 'primary'), dtype=np.float32)[None]
            final = prediction[index + 1:index + 2].astype(np.float32)
            target_box = target[index + 1:index + 2].astype(np.float32)
            primary_iou = float(iou(primary, target_box)[0])
            final_iou = float(iou(final, target_box)[0])
            rows.append(dict(sequence=name, frame=index + 1, modality=int(record['modality']),
                             action=action, query_selected=int(record['query_selected']),
                             primary_confidence=float(record['primary_confidence']),
                             query_confidence=float(record['query_confidence']),
                             primary_iou=primary_iou, final_iou=final_iou,
                             immediate_gain=final_iou - primary_iou))
    assert rows
    query_rows = [row for row in rows if row['query_selected']]
    baseline = float(np.mean([row['primary_iou'] for row in rows]))
    executed = float(np.mean([row['final_iou'] for row in rows]))
    oracle = float(np.mean([max(row['primary_iou'], row['final_iou']) for row in rows]))
    route_reports = {}
    for route in sorted({(row['modality'], row['action']) for row in query_rows}):
        route_rows = [row for row in query_rows if (row['modality'], row['action']) == route]
        candidates = sorted({0.0, 1.0} | {row['query_confidence'] for row in route_rows})
        best = None
        for threshold in candidates:
            values = [row['final_iou'] if row['query_confidence'] >= threshold else row['primary_iou']
                      for row in route_rows]
            score = (float(np.mean(values)), -threshold)
            if best is None or score > best[0]:
                best = (score, threshold, float(np.mean(values)),
                        int(sum(row['query_confidence'] >= threshold for row in route_rows)))
        route_reports[f'm{route[0]}_a{route[1]}'] = dict(frames=len(route_rows),
            threshold=best[1], immediate_mean_iou=best[2], accepted_query_frames=best[3],
            no_gate_mean_iou=float(np.mean([row['final_iou'] for row in route_rows])),
            mean_gain=float(np.mean([row['immediate_gain'] for row in route_rows])))
    return dict(status='IMMEDIATE_TRACE_ANALYSIS_ONLY', run=str(run), frames=len(rows),
                query_frames=len(query_rows), baseline_primary_mean_iou=baseline,
                executed_mean_iou=executed, oracle_accept_or_keep_mean_iou=oracle,
                immediate_query_gain=executed - baseline, oracle_improvement=oracle - baseline,
                routes=route_reports,
                scope='Same-frame diagnostic only; no closed-loop state/template update, no test selection, no official PR/NPR/SR.'), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    result, rows = analyze(args.run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with output.with_suffix('.csv').open('w', newline='') as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
