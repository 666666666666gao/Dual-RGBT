"""Audit confidence, queries and template writes on verified dev trajectories."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_lasher_query_benefit import analyze, sha

ROOT = Path(__file__).resolve().parents[1]
ACTION_NAMES = ['STOP', 'LOCAL_MAIN', 'LOCAL_ALTERNATE', 'GLOBAL_LOCAL', 'GLOBAL_WIDE']


def failure_episodes(rows):
    episodes, active = [], []

    def finish():
        if active:
            episodes.append(dict(sequence=active[0]['sequence'], start=active[0]['frame'],
                end=active[-1]['frame'], length=len(active),
                query_frames=sum(r['action'] != 0 for r in active),
                template_writes=sum(r['template_updated'] for r in active),
                high_confidence_frames=sum(r['confidence'] >= .65 for r in active)))
            active.clear()

    for row in rows:
        if active and (row['sequence'] != active[-1]['sequence'] or row['frame'] != active[-1]['frame'] + 1):
            finish()
        if row['valid_gt'] and row['final_iou'] < .1:
            active.append(row)
        else:
            finish()
    finish()
    return episodes


def summarize(rows):
    valid = [r for r in rows if r['valid_gt']]
    lost = [r for r in valid if r['final_iou'] < .1]
    primary_lost = [r for r in valid if r['primary_iou'] < .1]
    writes = [r for r in valid if r['template_updated']]
    queries = [r for r in valid if r['action']]
    episodes = failure_episodes(rows)
    bins = []
    for low, high in [(0, .1), (.1, .3), (.3, .5), (.5, .65), (.65, .9), (.9, 1.000001)]:
        group = [r for r in valid if low <= r['confidence'] < high]
        bins.append(dict(low=low, high=high, frames=len(group),
            lost_frames=sum(r['final_iou'] < .1 for r in group),
            good_overlap_frames=sum(r['final_iou'] >= .5 for r in group),
            template_writes=sum(r['template_updated'] for r in group)))
    return dict(valid_frames=len(valid), invalid_gt_frames=len(rows)-len(valid),
        lost_frames=len(lost), lost_high_confidence_frames=sum(r['confidence'] >= .65 for r in lost),
        primary_lost_frames=len(primary_lost),
        primary_lost_stop_frames=sum(r['action'] == 0 for r in primary_lost),
        primary_lost_queried_frames=sum(r['action'] != 0 for r in primary_lost),
        immediate_query_recoveries=sum(r['primary_iou'] < .1 and r['final_iou'] >= .5 for r in queries),
        immediate_query_failures=sum(r['primary_iou'] >= .5 and r['final_iou'] < .1 for r in queries),
        query_frames=len(queries), template_writes=len(writes),
        lost_template_writes=sum(r['final_iou'] < .1 for r in writes),
        low_overlap_template_writes=sum(r['final_iou'] < .5 for r in writes),
        bad_template_writes_after_stop=sum(r['final_iou'] < .1 and r['action'] == 0 for r in writes),
        bad_template_writes_after_query=sum(r['final_iou'] < .1 and r['action'] != 0 for r in writes),
        confidence_bins=bins, lost_episodes=len(episodes),
        lost_episodes_at_least10=sum(e['length'] >= 10 for e in episodes),
        longest_lost_episodes=sorted(episodes, key=lambda e: e['length'], reverse=True)[:20],
        by_action=[dict(action=name, frames=sum(r['action'] == a for r in valid),
            lost_frames=sum(r['action'] == a and r['final_iou'] < .1 for r in valid),
            template_writes=sum(r['action'] == a and r['template_updated'] for r in valid))
            for a, name in enumerate(ACTION_NAMES)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report_path = ROOT / args.report
    report = json.loads(report_path.read_text())
    tracker_path = ROOT / 'dual_rgbt_global/candidate_tracker.py'
    tracker_sha = sha(tracker_path)
    assert tracker_sha == report['source_sha256']['dual_rgbt_global/candidate_tracker.py']
    evidence, rows = analyze(ROOT, report_path)
    lookup = {(r['sequence'], r['frame']): r for r in rows}
    assert len(lookup) == len(rows)
    for run in report['run_names']:
        directory = ROOT / 'runs' / run
        config = json.loads((directory / 'config.json').read_text())
        for name in config['sequences']:
            path = directory / 'traces' / (name + '.csv')
            assert sha(path) == report['run_artifacts_sha256'][path.relative_to(ROOT).as_posix()]
            with path.open(newline='') as handle:
                for route in csv.DictReader(handle):
                    row = lookup[(name, int(route['frame']))]
                    row.update(confidence=float(route['confidence']),
                        template_updated=int(int(route['frame']) % 5 == 0 and float(route['confidence']) >= .65))
                    assert np.isfinite(row['confidence']) and 0 <= row['confidence'] <= 1
    result = dict(status='VERIFIED_DEV_TRAJECTORY_DIAGNOSTIC', source_report=args.report,
        source_report_sha256=sha(report_path), checkpoint_sha256=report['checkpoint_sha256'],
        cost_weight=report['cost_weight'], sequences=97, frames=report['frames'],
        template_write_reconstruction=dict(tracker_sha256=tracker_sha,
            rule='frame%5==0 and confidence>=0.65; reconstructed from the hash-verified tracker source because legacy traces omit the write flag'),
        validated_artifact_count=len(evidence['used_artifacts_sha256']),
        summary=summarize(rows), by_modality={str(m): summarize([r for r in rows if r['modality'] == m]) for m in (0, 1)},
        source_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('analyze_lasher_query_benefit.py')]},
        scope='Frame-weighted continuous-IoU diagnostics on already visited dev states, excluding initialization and invalid GT. Not official metrics. A STOP on a failed state does not prove an unexecuted query would help. Template-write correlation is not causal evidence. Modality-filtered episodes split at gaps.')
    output = ROOT / args.output
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result['summary'].items() if k not in ('confidence_bins', 'longest_lost_episodes', 'by_action')}, indent=2))


if __name__ == '__main__':
    main()
