"""Read verified exploratory full-test artifacts without changing model selection."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from analyze_lasher_query_benefit import localization, sha, summarize as query_summary
from diagnose_tracking_failures import summarize as failure_summary

ROOT = Path(__file__).resolve().parents[1]
TRACKER_SHA = '89e18853f60eac309e47206005450164931b68b3934ae36a40034cd3b3ce03ed'


def geometry_summary(iou, center_error, area_ratio):
    low_overlap = iou < .5
    close_center = center_error <= 20
    area_mismatch = (area_ratio < .5) | (area_ratio > 2)
    return dict(
        frames=len(iou), mean_continuous_iou=float(iou.mean()),
        low_overlap_frames=int(low_overlap.sum()),
        low_overlap_center_within20_frames=int((low_overlap & close_center).sum()),
        low_overlap_center_beyond20_frames=int((low_overlap & ~close_center).sum()),
        low_overlap_center_within20_area_mismatch_frames=int(
            (low_overlap & close_center & area_mismatch).sum()),
        area_ratio_below_half_frames=int((area_ratio < .5).sum()),
        area_ratio_above_double_frames=int((area_ratio > 2).sum()))


def read_checked(root, directory, artifacts, relative, used):
    path = directory / relative
    digest = sha(path)
    assert artifacts[relative] == digest, relative
    used[path.relative_to(root).as_posix()] = digest
    return path


def analyze(root, report_path, run_name):
    report = json.loads(report_path.read_bytes())
    assert report['status'] == 'SCORED_COMPLETE'
    assert (report['split'], report['sequences'], report['frames']) == ('test', 245, 220703)
    assert report['exploratory_test'] is True and report['test_eligible'] is False
    assert sha(report_path.with_suffix('.npz')) == report['curves_sha256']
    directory = root / 'runs' / run_name
    artifacts, used = report['run_artifacts_sha256'], {}

    def checked(relative):
        return read_checked(root, directory, artifacts, relative, used)

    config = json.loads(checked('config.json').read_bytes())
    complete = json.loads(checked('COMPLETE.json').read_bytes())
    native_path = root / 'runs/pipeline_tasks' / config['task'] / 'status.json'
    assert sha(native_path) == report['native_status_sha256']
    native = json.loads(native_path.read_bytes())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert native['command'] == config['native_command']
    assert native['task'] == config['task'] == complete['task']
    assert config['run'] == run_name
    assert config['benchmark'] == 'LasHeR' and config['split'] == 'test'
    assert config['smoke_frames'] == 0
    assert config['exploratory_test'] is complete['exploratory_test'] is True
    assert config['test_eligible'] is complete['test_eligible'] is False
    assert complete['optimizer_updates'] == complete['backward_calls'] == 0
    assert (complete['sequences'], complete['frames']) == (245, 220703)
    assert complete['config_sha256'] == sha(directory / 'config.json')
    assert config['checkpoint_sha256'] == complete['checkpoint_sha256'] == report['checkpoint_sha256']
    assert config['parent_checkpoint_sha256'] == report['parent_checkpoint_sha256']
    assert config['cost_weight'] == report['cost_weight']
    assert config['plan_sha256'] == complete['plan_sha256'] == report['evaluation_plan_sha256']
    assert sha(root / config['plan']) == config['plan_sha256']
    plan = json.loads((root / config['plan']).read_bytes())
    assert plan['checkpoint_sha256'] == config['checkpoint_sha256']
    assert sha(root / config['checkpoint']) == config['checkpoint_sha256']
    tracker_path = root / 'dual_rgbt_global/candidate_tracker.py'
    assert sha(tracker_path) == config['source_sha256']['dual_rgbt_global/candidate_tracker.py']
    assert sha(tracker_path) == report['source_sha256']['dual_rgbt_global/candidate_tracker.py'] == TRACKER_SHA
    names = config['sequences']
    assert len(names) == len(set(names)) == 245
    assert set(names) == {row['sequence'] for row in report['per_sequence']}
    rows, sequences = [], []
    all_iou, all_ce, all_ratio = [], [], []
    for name in names:
        final = np.loadtxt(checked('predictions/' + name + '.txt'), delimiter=',', ndmin=2)
        gt = np.loadtxt(checked('groundtruth/' + name + '.txt'), delimiter=',', ndmin=2)
        with checked('evidence/' + name + '.csv').open(newline='') as handle:
            evidence = list(csv.DictReader(handle))
        with checked('traces/' + name + '.csv').open(newline='') as handle:
            trace = list(csv.DictReader(handle))
        assert final.shape == gt.shape and final.shape[1] == 4
        assert len(evidence) == len(trace) == len(final) - 1
        assert np.isfinite(final).all() and (final[:, 2:] > 0).all()
        assert np.allclose(final[0], gt[0], rtol=0, atol=1e-5)
        assert [int(r['frame']) for r in trace] == list(range(1, len(final)))
        assert [int(r['frame']) for r in evidence] == list(range(1, len(final)))
        primary = np.asarray([[float(r['primary_' + k]) for k in ('x', 'y', 'w', 'h')]
                              for r in evidence])
        assert np.isfinite(primary).all() and (primary[:, 2:] > 0).all()
        valid = np.isfinite(gt[1:]).all(axis=1) & (gt[1:, 2:] > 0).all(axis=1)
        primary_metrics = localization(primary[valid], gt[1:][valid])
        final_metrics = localization(final[1:][valid], gt[1:][valid])
        ratio = final[1:][valid, 2:].prod(axis=1) / gt[1:][valid, 2:].prod(axis=1)
        all_iou.append(final_metrics[0])
        all_ce.append(final_metrics[1])
        all_ratio.append(ratio)
        seq_rows, metric_index = [], 0
        for index, (record, route) in enumerate(zip(evidence, trace)):
            action, modality = int(route['action']), int(route['modality'])
            assert action in range(5) and modality in (0, 1)
            # This exact sealed tracker omits the flag but has a verified fixed rule.
            assert route['template_updated'] == ''
            confidence = float(route['confidence'])
            assert np.isfinite(confidence) and 0 <= confidence <= 1
            if action == 0:
                assert np.allclose(primary[index], final[index + 1], rtol=0, atol=1e-6)
            row = dict(sequence=name, frame=index + 1, action=action, modality=modality,
                valid_gt=bool(valid[index]), confidence=confidence,
                template_updated=int((index + 1) % 5 == 0 and confidence >= .65),
                **{k: float(record[k]) for k in
                   ('selected_gain', 'selected_token_cost', 'selected_utility')})
            assert np.isfinite([row[k] for k in
                ('selected_gain', 'selected_token_cost', 'selected_utility')]).all()
            assert abs(row['selected_utility'] - row['selected_gain']
                       + config['cost_weight'] * row['selected_token_cost']) < 1e-5
            if valid[index]:
                piou, pce, pnce = (value[metric_index] for value in primary_metrics)
                fiou, fce, fnce = (value[metric_index] for value in final_metrics)
                row.update(primary_iou=float(piou), final_iou=float(fiou),
                    iou_gain=float(fiou - piou), center_error_reduction=float(pce - fce),
                    normalized_center_error_reduction=float(pnce - fnce),
                    realized_utility=float(fiou - piou - config['cost_weight'] * row['selected_token_cost']))
                metric_index += 1
            seq_rows.append(row)
        sequences.append(dict(sequence=name, frames=len(final), query=query_summary(seq_rows),
            geometry=geometry_summary(final_metrics[0], final_metrics[1], ratio),
            failure=failure_summary(seq_rows)))
        rows.extend(seq_rows)
    assert len(rows) + len(names) == report['frames']
    return dict(status='VERIFIED_EXPLORATORY_FULL_TEST_TRAJECTORY_DIAGNOSTIC',
        benchmark='LasHeR', split='test', run=run_name, sequences=245, frames=220703,
        exploratory_test=True, test_eligible=False,
        source_report=report_path.relative_to(root).as_posix(), source_report_sha256=sha(report_path),
        checkpoint_sha256=report['checkpoint_sha256'],
        parent_checkpoint_sha256=report['parent_checkpoint_sha256'],
        official_metrics={k: report[k] for k in ('PR', 'NPR', 'SR', 'fps', 'total_parameters')},
        cost_weight=report['cost_weight'], used_artifacts_sha256=used,
        native_status_sha256=sha(native_path),
        template_write_reconstruction=dict(tracker_sha256=TRACKER_SHA,
            rule='frame%5==0 and confidence>=0.65; legacy trace flags are empty'),
        query=query_summary(rows), failure=failure_summary(rows),
        geometry=geometry_summary(np.concatenate(all_iou), np.concatenate(all_ce), np.concatenate(all_ratio)),
        per_sequence=sequences,
        analysis_source_sha256={p.name: sha(p) for p in (
            Path(__file__), Path(__file__).with_name('analyze_lasher_query_benefit.py'),
            Path(__file__).with_name('diagnose_tracking_failures.py'))},
        scope='Read-only, frame-weighted continuous geometry on visited full-test states; initialization and invalid GT excluded. '
              'Official PR/NPR/SR remain unchanged and sequence-averaged. Center/area categories describe errors, not causes. '
              'Executed-query gains do not measure unexecuted opportunities. Template-write correlation is not causal evidence. '
              'No training labels, model selection, threshold changes, or independent final-test claim.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists()
    result = analyze(ROOT, ROOT / args.report, args.run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'query', 'geometry')}, indent=2))


if __name__ == '__main__':
    main()
