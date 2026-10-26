"""Condition saved train-query gains on primary IoU; no neural execution."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / 'runs/global_candidate_rankstop_trajectory_formal_20261018'
    complete = json.loads((run / 'COMPLETE.json').read_bytes())
    labels = run / 'query_labels.csv'
    label_sha = hashlib.sha256(labels.read_bytes()).hexdigest()
    assert label_sha == complete['private_labels_sha256']
    counts = defaultdict(lambda: dict(labels=0, positive=0, negative=0, near_zero=0,
                                     query_iou_at_least_0p5=0, sum_gain=0.0))
    with labels.open(newline='') as handle:
        for row in csv.DictReader(handle):
            main_iou, query_iou, gain = [float(row[key]) for key in ('main_iou', 'query_iou', 'delta_from_stop')]
            assert all(math.isfinite(value) for value in (main_iou, query_iou, gain))
            assert 0 <= main_iou <= 1 and 0 <= query_iou <= 1
            assert abs(query_iou - main_iou - gain) < 1e-6
            regime = 'primary_lt_0p1' if main_iou < .1 else 'primary_0p1_to_lt_0p5' if main_iou < .5 else 'primary_ge_0p5'
            key = (row['purpose'], regime, int(row['kind']))
            item = counts[key]
            item['labels'] += 1
            item['positive'] += gain > .01
            item['negative'] += gain < -.01
            item['near_zero'] += abs(gain) <= .01
            item['query_iou_at_least_0p5'] += query_iou >= .5
            item['sum_gain'] += gain
    groups = []
    for (purpose, regime, kind), item in sorted(counts.items()):
        item['mean_gain'] = item.pop('sum_gain') / item['labels']
        assert item['positive'] + item['negative'] + item['near_zero'] == item['labels']
        groups.append(dict(purpose=purpose, primary_iou_regime=regime, kind=kind, **item))
    assert sum(item['labels'] for item in groups) == complete['labelled_queries'] == 399401
    assert sum(item['labels'] for item in groups if item['purpose'] == 'sampled') == complete['sampled_query_labels']
    assert sum(item['labels'] for item in groups if item['purpose'] == 'selected_extra') == complete['extra_selected_query_labels']
    report = dict(status='CPU_CONDITIONAL_LABEL_CHECK_PASS', labelled_queries=399401,
                  label_sha256=label_sha, checkpoint_sha256=complete['checkpoint_sha256'],
                  groups=groups, thresholds=dict(primary_iou=[.1, .5], gain_neutral_band=.01),
                  scope='Saved TRAIN counterfactual queries on both primary roles, plus selected-extra queries. '
                        'Bins are descriptive diagnostics only. They are not observable runtime GT gates, '
                        'independent frames, deployed harmful-query rates, full dev/test scores or recovery trajectories. '
                        'Do not infer that a query family should be disabled from its unconditional average.')
    output = ROOT / 'reports/RANKSTOP_CONDITIONAL_QUERY_DIAGNOSTIC_20261026.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
