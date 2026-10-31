"""Summarize sampled TRAIN reliability from verified existing parent caches."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha


def summarize(rows):
    lost = [r for r in rows if r['final_iou'] < .1]
    high = [r for r in rows if r['confidence'] >= .65]
    return dict(states=len(rows), sequences=len({r['sequence'] for r in rows}),
        lost_states=len(lost), lost_high_confidence_states=sum(r['confidence'] >= .65 for r in lost),
        high_confidence_states=len(high), high_confidence_good_overlap_states=sum(r['final_iou'] >= .5 for r in high),
        high_confidence_lost_states=sum(r['final_iou'] < .1 for r in high),
        primary_lost_states=sum(r['primary_iou'] < .1 for r in rows),
        query_states=sum(r['action'] != 0 for r in rows),
        by_confidence=[dict(low=low, high=high, states=len(group),
            lost_states=sum(r['final_iou'] < .1 for r in group),
            good_overlap_states=sum(r['final_iou'] >= .5 for r in group))
            for low, high in [(0, .1), (.1, .3), (.3, .5), (.5, .65), (.65, .9), (.9, 1.000001)]
            for group in [[r for r in rows if low <= r['confidence'] < high]]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    expected_parent = '21e07cd1f645c6f7237e063c81cf5d60f6a14bca710e0180e8a3ca29dc827073'
    assert plan['parent_checkpoint_sha256'] == expected_parent
    rows, seen, used = [], set(), {args.plan: sha(plan_path)}
    for entry in plan['caches']:
        audit_path = ROOT / entry['audit']
        assert sha(audit_path) == entry['audit_sha256']
        audit = json.loads(audit_path.read_text())
        assert audit['status'] == 'VERIFIED_CACHE_DIAGNOSTIC' and not audit['smoke']
        directory = ROOT / 'runs' / entry['run']
        receipt_path = directory / 'COMPLETE.json'
        native_path = ROOT / 'runs/pipeline_tasks' / entry['task'] / 'status.json'
        assert sha(receipt_path) == audit['receipt_sha256'] and sha(native_path) == audit['native_sha256']
        receipt, native = json.loads(receipt_path.read_text()), json.loads(native_path.read_text())
        config_path = directory / 'config.json'
        assert sha(config_path) == receipt['artifacts_sha256']['config.json']
        config = json.loads(config_path.read_text())
        assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
        assert native['command'] == config['native_command']
        assert config['split'] == 'train' and config['checkpoint_sha256'] == expected_parent
        assert receipt['status'] == 'CACHE_COMPLETE' and receipt['teacher_preserves_tracker_state']
        assert receipt['optimizer_updates'] == receipt['backward_calls'] == 0
        for path in (audit_path, receipt_path, native_path, config_path):
            used[path.relative_to(ROOT).as_posix()] = sha(path)
        cache_plan = config['plan']
        for name in config['selected_sequences']:
            assert name not in seen
            seen.add(name)
            partition = 'fit' if name in cache_plan['fit_sequences'] else 'holdout'
            assert name in cache_plan[partition + '_sequences']
            labels_path = directory / (name + '_labels.csv')
            assert sha(labels_path) == receipt['artifacts_sha256'][labels_path.name]
            used[labels_path.relative_to(ROOT).as_posix()] = sha(labels_path)
            with labels_path.open(newline='') as handle:
                for frame, group in itertools.groupby(csv.DictReader(handle), key=lambda row: int(row['frame'])):
                    group = list(group)
                    first = group[0]
                    assert frame % 5 == 0
                    assert all(r['sequence'] == name and r['partition'] == partition for r in group)
                    assert len({r['primary_iou'] for r in group}) == len({r['deployed_action'] for r in group}) == 1
                    selected = [r for r in group if int(r['selected'])]
                    action = int(first['deployed_action'])
                    assert action in range(5) and len(selected) == int(action != 0)
                    confidence = float(selected[0]['query_confidence'] if action else first['primary_confidence'])
                    final_iou = float(selected[0]['query_iou'] if action else first['primary_iou'])
                    assert 0 <= confidence <= 1 and 0 <= final_iou <= 1
                    rows.append(dict(sequence=name, partition=partition, frame=frame,
                        modality=int(first['modality']), action=action,
                        primary_iou=float(first['primary_iou']), confidence=confidence, final_iou=final_iou))
    assert len(seen) == plan['expected_sequences'] == 882
    assert len({(r['sequence'], r['frame']) for r in rows}) == len(rows)
    output = ROOT / args.output
    assert not output.exists()
    result = dict(status='VERIFIED_SAMPLED_TRAIN_RELIABILITY_DIAGNOSTIC',
        checkpoint_sha256=expected_parent, split='train', test_eligible=False, sequences=len(seen),
        summary=summarize(rows),
        by_partition={p: summarize([r for r in rows if r['partition'] == p]) for p in ('fit', 'holdout')},
        by_modality={str(m): summarize([r for r in rows if r['modality'] == m]) for m in (0, 1)},
        used_inputs_sha256=used, analysis_source_sha256=sha(Path(__file__)),
        scope='Read-only sampled fifth-frame states from existing frozen-parent TRAIN caches. '
              'Final confidence/IoU reconstruct the actually executed STOP or selected query. '
              'No unselected candidate is treated as an executed observation. '
              'Not all TRAIN frames, no benchmark metrics, no new collection or parameter update, '
              'no comparison of unobserved primary modalities, no causal memory-effect claim.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'summary', 'by_partition')}, indent=2))


if __name__ == '__main__':
    main()
