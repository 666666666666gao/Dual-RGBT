"""Summarize verified fixed-state dev artifacts without further model execution."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(frames, queries):
    valid = [f for f in frames if f['primary']['valid_gt']]
    sampled = [f for f in valid if f['sampled']]
    groups = defaultdict(list)
    for query in queries:
        if query['metrics']['valid_gt']:
            groups[query['frame']].append(query)
    assert {f['frame'] for f in sampled} == set(groups)
    recoverable, missed_stop, selected_recovery, harms = 0, 0, 0, 0
    by_action = defaultdict(list)
    for frame in sampled:
        pool = groups[frame['frame']]
        can_recover = any(q['recovery'] for q in pool)
        recoverable += can_recover
        missed_stop += can_recover and frame['action'] == 0
        selected_recovery += any(q['selected'] and q['recovery'] for q in pool)
        harms += any(q['selected'] and q['harm'] for q in pool)
        for query in pool:
            by_action[query['action']].append(query)
    writes = [f for f in valid if f['template_write_modalities']]
    selected = [q for q in queries if q['selected'] and q['metrics']['valid_gt']]
    lost_intervals, start, end = [], None, None
    for frame in frames:
        lost = frame['final']['valid_gt'] and frame['final']['iou'] < .1
        if lost:
            if start is None:
                start = frame['frame']
            end = frame['frame']
        elif start is not None:
            lost_intervals.append(dict(start=start, end=end, frames=end - start + 1))
            start = None
    if start is not None:
        lost_intervals.append(dict(start=start, end=end, frames=end - start + 1))
    size_bad = [f for f in valid if f['final']['center_error'] <= 20 and f['final']['iou'] < .5]
    return dict(tracked_frames=len(frames), valid_gt_frames=len(valid),
        stop_frames=sum(f['action'] == 0 for f in frames), sampled_valid_states=len(sampled),
        candidate_executions=len(queries), sampled_recoverable_states=recoverable,
        sampled_recoverable_stop_states=missed_stop,
        sampled_selected_recoveries=selected_recovery, sampled_selected_harms=harms,
        sampled_selected_queries=len(selected),
        sampled_selected_negative_gain=sum(q['gain'] < -1e-6 for q in selected),
        sampled_selected_mean_gain=sum(q['gain'] for q in selected) / len(selected) if selected else None,
        alternate_primary_better_by_iou=sum(f['other_primary']['iou'] > f['primary']['iou'] + .01
                                           for f in sampled),
        alternate_primary_recovers=sum(f['primary']['iou'] < .1 and f['other_primary']['iou'] >= .5
                                       for f in sampled),
        both_primaries_lost=sum(f['primary']['iou'] < .1 and f['other_primary']['iou'] < .1
                               for f in sampled),
        center_within20_iou_below_half=len(size_bad),
        same_center_area_outside_band=sum(f['final']['area_ratio'] < 2/3 or
                                         f['final']['area_ratio'] > 1.5 for f in size_bad),
        observed_template_write_frames=len(writes),
        observed_template_writes_when_lost=sum(f['final']['iou'] < .1 for f in writes),
        longest_lost_intervals=sorted(lost_intervals, key=lambda x: -x['frames'])[:5],
        actions={str(action): dict(executions=len(pool),
                    recovery_queries=sum(q['recovery'] for q in pool),
                    harmful_queries=sum(q['harm'] for q in pool),
                    mean_gain=sum(q['gain'] for q in pool) / len(pool),
                    predicted_positive_utility=sum(q['utility_prediction'] > 0 for q in pool),
                    recoveries_predicted_positive=sum(q['recovery'] and q['utility_prediction'] > 0
                                                     for q in pool))
                 for action, pool in sorted(by_action.items())})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    results = []
    for name in args.run:
        run = ROOT / 'runs' / name
        complete_path = run / 'COMPLETE.json'
        complete = json.loads(complete_path.read_text())
        assert complete['status'] == 'DIAGNOSTIC_COMPLETE'
        assert complete['saved_predictions_and_routes_equal'] and complete['diagnostic_preserves_live_state']
        assert complete['frozen_state_tensors'] == 265
        assert complete['optimizer_updates'] == complete['backward_calls'] == 0
        assert complete['test_eligible'] is complete['training_allowed'] is False
        for file, digest in complete['artifacts_sha256'].items():
            assert sha(run / file) == digest
        config = json.loads((run / 'config.json').read_text())
        native_path = ROOT / 'runs/pipeline_tasks' / ('diagnose_' + name) / 'status.json'
        native = json.loads(native_path.read_text())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
        assert native['command'] == config['native_command']
        plan = config['plan']
        reference = ROOT / 'runs' / plan['reference_run']
        sequence = config['sequence']
        assert sha(reference / 'config.json') == config['reference_config_sha256']
        assert sha(reference / 'predictions' / (sequence + '.txt')) == config['reference_predictions_sha256']
        assert sha(reference / 'traces' / (sequence + '.csv')) == config['reference_trace_sha256']
        frames = [json.loads(line) for line in (run / 'frames.jsonl').read_text().splitlines()]
        queries = [json.loads(line) for line in (run / 'queries.jsonl').read_text().splitlines()]
        assert len(frames) == complete['frames'] - 1
        assert [f['frame'] for f in frames] == list(range(1, complete['frames']))
        assert len(queries) == complete['query_executions']
        results.append(dict(run=name, reference_run=plan['reference_run'], sequence=sequence,
                            checkpoint_sha256=complete['checkpoint_sha256'],
                            complete_sha256=sha(complete_path), native_sha256=sha(native_path),
                            **summary(frames, queries)))
    output = ROOT / args.output
    assert not output.exists()
    output.write_text(json.dumps(dict(status='VERIFIED_DIAGNOSTIC', runs=results,
        analysis_script_sha256=sha(Path(__file__)),
        scope='Dev-only, four purposively chosen sequences per policy. Each policy uses its own '
              'visited states; histories differ between policies. All candidates at every fifth frame '
              'are measured, not all frames. No dev labels enter training. Template writes are '
              'observed, not proven causes. Immediate oracle gains are not closed-loop accuracy, '
              'official SR or a future-trajectory upper bound.'), indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
