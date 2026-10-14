"""Magnitude-aware summaries of already CPU-verified fixed-state query utilities."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    gains = np.asarray([float(row['delta_from_stop']) for row in rows])
    ious = np.asarray([float(row['iou']) for row in rows])
    return dict(states=len(rows), mean_query_iou=float(ious.mean()),
        mean_signed_gain=float(gains.mean()), gt_stop_oracle_gain=float(np.maximum(gains, 0).mean()),
        gain_quantiles=dict(zip(('min', 'p10', 'median', 'p90', 'max'), np.quantile(gains, [0, .1, .5, .9, 1]).tolist())),
        positive=int((gains > 0).sum()), negative=int((gains < 0).sum()), zero=int((gains == 0).sum()),
        substantial_gain_counts={str(t): int((gains >= t).sum()) for t in (.01, .05, .1)},
        substantial_harm_counts={str(t): int((gains <= -t).sum()) for t in (.01, .05, .1)},
        query_iou_at_least={str(t): int((ious >= t).sum()) for t in (.1, .3, .5)})


def main():
    parser = argparse.ArgumentParser()
    for name in ('analysis', 'csv', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    analysis = json.loads(Path(args.analysis).read_bytes())
    assert analysis['status'] == 'CPU_ANALYSIS_PASS' and analysis['states'] == 1752
    assert sha(args.csv) == analysis['input_sha256']['csv']
    with Path(args.csv).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    states = {}
    for row in rows:
        states.setdefault(int(row['state']), []).append(row)
    assert len(states) == 1752 and len(rows) == analysis['auxiliary_queries'] == 31384
    groups = {}
    for scope in ('all', 'main_centre_outside'):
        selected = [group for group in states.values() if scope == 'all' or group[0]['main_centre_inside'] == 'False']
        assert len(selected) == (1752 if scope == 'all' else 239)
        fixed = {}
        for region in ('LOCAL_MAIN', 'LOCAL_ALTERNATE', 'GLOBAL_LOCAL', 'GLOBAL_WIDE'):
            rank = -1 if region.startswith('LOCAL_') else 0
            chosen = [next(row for row in group if row['region'] == region and int(row['candidate_rank']) == rank) for group in selected]
            fixed[region + ('_TOP1' if rank == 0 else '')] = summarize(chosen)
        oracle = {}
        for k in (1, 2, 4, 8):
            oracle[str(k)] = {}
            for region in ('GLOBAL_LOCAL', 'GLOBAL_WIDE'):
                chosen = [max((row for row in group if row['region'] == region and int(row['candidate_rank']) < k),
                              key=lambda row: float(row['iou'])) for group in selected]
                oracle[str(k)][region] = summarize(chosen)
        groups[scope] = dict(states=len(selected), fixed_queries=fixed, query_oracles=oracle)
    result = dict(status='CPU_CHOICE_SUMMARY_COMPLETE', nn_calls=0, per_scope=groups,
        input_sha256=dict(analysis=sha(args.analysis), csv=sha(args.csv)), analyzer_sha256=sha(__file__),
        scope='Only fixed train20 reference states and already executed single-query counterfactuals. '+
              'Top1 uses the cheap cosine proposal order; best-prefix and GT-dependent STOP are oracle diagnoses. '+
              'No new NN, deployable selector, closed-loop metric or latency measurement. '+
              'Comparisons within GLOBAL_LOCAL or GLOBAL_WIDE hold one-query encoder size/tokens fixed, '+
              'without claiming matched actual system latency.')
    output = Path(args.output)
    assert not output.exists()
    output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(dict(status=result['status'], outside_top1=groups['main_centre_outside']['fixed_queries'],
                         outside_k8=groups['main_centre_outside']['query_oracles']['8']), indent=2))


if __name__ == '__main__':
    main()
