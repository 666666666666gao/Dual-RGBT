"""Describe candidate availability on completed TRAIN sequences, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--collection-plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = root / args.output
    assert not output.exists()
    path = root / args.collection_plan
    plan = json.loads(path.read_bytes())
    assert plan['phase'] == 'full' and plan['split'] == 'train'
    manifest_path = root / plan['manifest']
    assert sha(manifest_path) == plan['input_sha256'][plan['manifest']]
    manifest = json.loads(manifest_path.read_bytes())
    by_name = {sequence['name']: sequence for sequence in manifest['train']}
    groups = {part: {state: dict(sets=0, candidate_positive=0, no_match=0, ambiguous=0)
                    for state in ('reliable', 'uncertain', 'lost')}
              for part in ('fit', 'holdout')}
    snapshots, used, seen = [], {}, set()
    fit = set(plan['fit_sequences'])
    for job in plan['jobs']:
        directory = root / 'runs' / job['run']
        progress_path = directory / 'progress.json'
        progress_bytes = progress_path.read_bytes()
        rows = json.loads(progress_bytes)
        native = json.loads((root / 'runs/pipeline_tasks' / job['task'] / 'status.json').read_bytes())
        snapshots.append(dict(task=job['task'], native_state=native['state'],
            completed_sequences=len(rows), progress_sha256=hashlib.sha256(progress_bytes).hexdigest()))
        for row in rows:
            name = row['sequence']
            assert name in job['sequence_names'] and name not in seen
            seen.add(name)
            sequence = by_name[name]
            part = 'fit' if name in fit else 'holdout'
            assert row['partition'] == part
            array_path = directory / (name + '.npz')
            used[array_path.relative_to(root).as_posix()] = sha(array_path)
            with np.load(array_path, allow_pickle=False) as arrays:
                gt = np.asarray(sequence['boxes'], np.float64)[arrays['frame']]
                prediction = arrays['predictions'][arrays['frame']]
                left = np.maximum(prediction[:, :2], gt[:, :2])
                right = np.minimum(prediction[:, :2] + prediction[:, 2:], gt[:, :2] + gt[:, 2:])
                intersection = np.maximum(0, right - left).prod(1)
                iou = intersection / (prediction[:, 2:].prod(1) + gt[:, 2:].prod(1) - intersection)
                assert np.isfinite(iou).all() and ((iou >= 0) & (iou <= 1)).all()
                for index, value in enumerate(iou):
                    state = 'reliable' if value >= .5 else ('lost' if value < .1 else 'uncertain')
                    group = groups[part][state]
                    group['sets'] += 1
                    group['candidate_positive'] += int((arrays['labels'][index, 1:] == 1).any())
                    group['no_match'] += int(arrays['labels'][index, 0] == 1)
                    group['ambiguous'] += int(arrays['labels'][index, 0] == -1)
    report = dict(status='DESCRIPTIVE_TRAIN_CANDIDATE_SNAPSHOT',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        collection_plan=args.collection_plan, collection_plan_sha256=sha(path),
        manifest_sha256=sha(manifest_path), completed_sequences=len(seen),
        groups=groups, shard_snapshots=snapshots, inputs_sha256=used,
        full_collection_accepted=False, test_eligible=False,
        scientific_result_audit_accepted=False,
        scope='Descriptive snapshot of completed TRAIN sequence files only. '
              'Reads stored labels, does not rerun collection acceptance. '
              'E0 overlap strata use >=0.5/reliable, <0.1/lost, otherwise uncertain. '
              'Candidate availability is not learned accuracy or recovery success. '
              'Partial ordering and upstream component exposure preclude generalization claims.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('inputs_sha256', 'shard_snapshots')}), flush=True)


if __name__ == '__main__':
    main()
