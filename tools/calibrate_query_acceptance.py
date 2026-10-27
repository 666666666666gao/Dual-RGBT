"""Fit route-specific query thresholds from a scored LasHeR dev report.

This is a calibration report only. It does not alter predictions or claim a
closed-loop gain; the frozen thresholds must be rerun through the tracker on
the same development split before any test selection.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from dual_rgbt_global.query_acceptance import ROUTES, route_name

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(boxes, gt):
    low = np.maximum(boxes[:, :2], gt[:, :2])
    high = np.minimum(boxes[:, :2] + boxes[:, 2:], gt[:, :2] + gt[:, 2:])
    intersection = np.maximum(high - low, 0).prod(1)
    union = boxes[:, 2:].prod(1) + gt[:, 2:].prod(1) - intersection
    return intersection / union


def rows_from_report(report_path):
    report = json.loads(report_path.read_bytes())
    assert report['status'] == 'SCORED_COMPLETE'
    assert (report['benchmark'], report['split'], report['sequences']) == ('LasHeR', 'dev', 97)
    rows, covered, used = [], [], {}

    def checked(path):
        name = path.relative_to(ROOT).as_posix()
        assert report['run_artifacts_sha256'][name] == sha(path), name
        used[name] = report['run_artifacts_sha256'][name]
        return path

    for run_name in report['run_names']:
        directory = ROOT / 'runs' / run_name
        config = json.loads(checked(directory / 'config.json').read_bytes())
        assert config['evidence_schema'] == 'primary_executed_box_and_selected_gain_v1'
        for name in config['sequences']:
            prediction = np.loadtxt(checked(directory / 'predictions' / (name + '.txt')), delimiter=',', ndmin=2)
            gt = np.loadtxt(checked(directory / 'groundtruth' / (name + '.txt')), delimiter=',', ndmin=2)
            with checked(directory / 'traces' / (name + '.csv')).open(newline='') as handle:
                trace = list(csv.DictReader(handle))
            with checked(directory / 'evidence' / (name + '.csv')).open(newline='') as handle:
                evidence = list(csv.DictReader(handle))
            assert len(trace) == len(evidence) == len(prediction) - 1
            valid = np.isfinite(gt[1:]).all(1) & (gt[1:, 2:] > 0).all(1)
            for index, (route, record) in enumerate(zip(trace, evidence)):
                if not valid[index]:
                    continue
                action = int(route['action'])
                rows.append(dict(route=route_name(int(route['modality']), action), action=action,
                                 confidence=float(route['confidence']),
                                 primary_confidence=float(record['primary_confidence']),
                                 primary_iou=float(iou(prediction[index:index + 1], gt[index + 1:index + 2])[0])
                                 if action == 0 else float('nan'),
                                 query_iou=float(iou(prediction[index + 1:index + 2], gt[index + 1:index + 2])[0])
                                 if action else float('nan'),
                                 sequence=name, frame=index + 1))
            covered.extend(config['sequences'])
    assert len(set(covered)) == len(covered) == 97
    return report, rows, used


def fit(rows):
    thresholds = {route: 1.0 for route in ROUTES}
    route_reports = {}
    for route in ROUTES:
        candidates = sorted({0.0, 1.0} | {row['confidence'] for row in rows if row['route'] == route and row['action']})
        route_rows = [row for row in rows if row['route'] == route]
        query_rows = [row for row in route_rows if row['action']]
        if not query_rows:
            route_reports[route] = dict(status='NO_QUERY_ROWS', frames=len(route_rows), query_frames=0)
            continue
        best = None
        for threshold in candidates:
            values = []
            for row in query_rows:
                values.append(row['query_iou'] if row['confidence'] >= threshold else row['primary_iou'])
            mean = float(np.mean(values))
            accepted = int(sum(row['confidence'] >= threshold for row in query_rows))
            candidate = (mean, accepted, -threshold, threshold)
            if best is None or candidate > best[0]:
                best = (candidate, mean, accepted)
        thresholds[route] = float(best[0][3])
        route_reports[route] = dict(status='CALIBRATED_IMMEDIATE_DEV', frames=len(route_rows),
                                    query_frames=len(query_rows), threshold=thresholds[route],
                                    accepted_query_frames=best[2], accepted_rate=best[2] / len(query_rows),
                                    immediate_mean_iou=float(best[1]),
                                    no_gate_mean_iou=float(np.mean([r['query_iou'] for r in query_rows])))
    return thresholds, route_reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists()
    report, rows, used = rows_from_report(ROOT / args.report)
    thresholds, route_reports = fit(rows)
    result = dict(status='CALIBRATION_ONLY_NOT_CLOSED_LOOP', source_report=args.report,
                  source_report_sha256=sha(ROOT / args.report), checkpoint_sha256=report['checkpoint_sha256'],
                  route_thresholds=thresholds, routes=route_reports, used_artifacts_sha256=used,
                  scope='Immediate same-frame dev calibration from saved predictions; must rerun tracker to measure closed-loop dev. No test selection, no weight change, no official metric claim.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], route_thresholds=thresholds)))


if __name__ == '__main__':
    main()
