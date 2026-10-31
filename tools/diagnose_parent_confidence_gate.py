"""Measure a fit-only confidence gate on existing verified TRAIN observations."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha


def fit_threshold(rows, retained_good_fraction=.95):
    good = sorted(row['confidence'] for row in rows if row['final_iou'] >= .5)
    assert good and 0 < retained_good_fraction < 1
    permitted = math.floor((1 - retained_good_fraction) * len(good))
    return good[permitted]


def operating_point(rows, threshold):
    assert rows
    lost = [row for row in rows if row['final_iou'] < .1]
    good = [row for row in rows if row['final_iou'] >= .5]
    assert lost and good
    detected = sum(row['confidence'] < threshold for row in lost)
    interrupted = sum(row['confidence'] < threshold for row in good)
    return dict(threshold=threshold, gate_rule='confidence < threshold',
        states=len(rows), lost_states=len(lost), good_overlap_states=len(good),
        triggered_states=sum(row['confidence'] < threshold for row in rows),
        lost_detected=detected, lost_missed=len(lost) - detected,
        lost_recall=detected / len(lost),
        good_interrupted=interrupted, good_retained=len(good) - interrupted,
        good_retained_fraction=1 - interrupted / len(good))


def lost_vs_good_auc(rows):
    selected = [row for row in rows if row['final_iou'] < .1 or row['final_iou'] >= .5]
    scores = np.asarray([1 - row['confidence'] for row in selected], dtype=np.float64)
    positive = np.asarray([row['final_iou'] < .1 for row in selected], dtype=bool)
    positives, negatives = int(positive.sum()), int((~positive).sum())
    assert positives and negatives and np.isfinite(scores).all()
    order = np.argsort(scores, kind='mergesort')
    ordered = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
    ends = np.r_[starts[1:], len(ordered)]
    # Average the ranks of exact ties before the Mann-Whitney calculation.
    ranks = np.repeat((starts + 1 + ends) / 2, ends - starts)
    numerator = ranks[positive[order]].sum() - positives * (positives + 1) / 2
    return dict(auc=float(numerator / (positives * negatives)),
        positive_lost_states=positives, negative_good_states=negatives,
        excluded_uncertain_states=len(rows) - len(selected),
        score='1 - final_response_confidence', ties='average_rank')


def read_verified_rows(root, report):
    expected_parent = '21e07cd1f645c6f7237e063c81cf5d60f6a14bca710e0180e8a3ca29dc827073'
    assert report['status'] == 'VERIFIED_SAMPLED_TRAIN_RELIABILITY_DIAGNOSTIC'
    assert report['split'] == 'train' and report['test_eligible'] is False
    assert report['checkpoint_sha256'] == expected_parent and report['sequences'] == 882
    assert sha(root / 'tools/analyze_parent_train_reliability_cache.py') == report['analysis_source_sha256']
    for name, digest in report['used_inputs_sha256'].items():
        assert sha(root / name) == digest, name
    rows = []
    for name in sorted(report['used_inputs_sha256']):
        if not name.endswith('_labels.csv'):
            continue
        with (root / name).open(newline='') as handle:
            for frame, records in itertools.groupby(csv.DictReader(handle), key=lambda row: int(row['frame'])):
                records = list(records)
                first = records[0]
                sequence, partition = first['sequence'], first['partition']
                assert frame > 0 and frame % 5 == 0 and partition in ('fit', 'holdout')
                assert all(row['sequence'] == sequence and row['partition'] == partition for row in records)
                assert len({row['primary_iou'] for row in records}) == 1
                assert len({row['deployed_action'] for row in records}) == 1
                selected = [row for row in records if int(row['selected'])]
                action = int(first['deployed_action'])
                assert action in range(5) and len(selected) == int(action != 0)
                confidence = float(selected[0]['query_confidence'] if action else first['primary_confidence'])
                iou = float(selected[0]['query_iou'] if action else first['primary_iou'])
                assert 0 <= confidence <= 1 and 0 <= iou <= 1
                rows.append(dict(sequence=sequence, frame=frame, partition=partition,
                    confidence=confidence, final_iou=iou))
    assert len({(row['sequence'], row['frame']) for row in rows}) == len(rows)
    assert len({row['sequence'] for row in rows}) == 882
    assert len(rows) == report['summary']['states']
    assert sum(row['final_iou'] < .1 for row in rows) == report['summary']['lost_states']
    assert sum(row['final_iou'] < .1 and row['confidence'] >= .65 for row in rows) == (
        report['summary']['lost_high_confidence_states'])
    return rows


def diagnose(rows):
    fit = [row for row in rows if row['partition'] == 'fit']
    held = [row for row in rows if row['partition'] == 'holdout']
    assert {row['sequence'] for row in fit}.isdisjoint(row['sequence'] for row in held)
    threshold = fit_threshold(fit)
    assert operating_point(fit, threshold)['good_retained_fraction'] >= .95
    return dict(threshold_selection=dict(partition='fit',
        rule='Retain at least 95 percent of fit states with final IoU >= .5; strict low-confidence trigger.',
        threshold=threshold, holdout_used=False, test_used=False),
        by_partition={name: dict(auc=lost_vs_good_auc(part),
            fit95_good_retention_gate=operating_point(part, threshold),
            existing_confidence_065_reference=operating_point(part, .65))
            for name, part in (('fit', fit), ('holdout', held))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--report-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report_path = ROOT / args.report
    assert sha(report_path) == args.report_sha256
    report = json.loads(report_path.read_text())
    rows = read_verified_rows(ROOT, report)
    for partition, count in (('fit', 706), ('holdout', 176)):
        part = [row for row in rows if row['partition'] == partition]
        assert len({row['sequence'] for row in part}) == count
        assert len(part) == report['by_partition'][partition]['states']
    result = dict(status='READ_ONLY_TRAIN_CONFIDENCE_GATE_DIAGNOSTIC',
        checkpoint_sha256=report['checkpoint_sha256'], split='train',
        sequences=882, sampled_states=len(rows), test_eligible=False,
        scientific_result_audit_accepted=False, optimizer_updates=0,
        source_report=args.report, source_report_sha256=args.report_sha256,
        input_hashes_reverified=len(report['used_inputs_sha256']),
        analysis_source_sha256=sha(Path(__file__)), **diagnose(rows),
        scope='Existing parent21e07 sampled TRAIN observations, not protected E0 or all TRAIN frames. '
              'Low-confidence gate is diagnostic only and is not deployed. IoU below .1 defines '
              'lost; IoU at least .5 defines good localization, not verified target identity. '
              'AUROC excludes intermediate overlap states. Fit-only threshold; upstream model '
              'has seen the component holdout. No causal recovery benefit, benchmark metric, '
              'independent generalization or scientific audit claim.')
    output = ROOT / args.output
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
