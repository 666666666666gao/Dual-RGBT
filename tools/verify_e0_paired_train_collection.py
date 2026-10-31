"""Verify complete TRAIN trajectories before finalizing the first M1 plan."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_e0_reliability import load_collection, localization_targets, sha

VERIFIER = 'tools/verify_e0_paired_train_collection.py'
TESTS = 'tests/test_verify_e0_paired_train_collection.py'


def verify_predictions(directory, sequence, partition, artifacts):
    """Reproduce the collector's mixed float32/float64 overlap arithmetic."""
    name = sequence['name']
    relative = 'predictions/' + name + '.npy'
    assert relative in artifacts
    assert sha(directory / relative) == artifacts[relative]
    predictions = np.load(directory / relative, allow_pickle=False)
    assert predictions.dtype == np.float32
    assert predictions.shape == (len(sequence['rgb']), 4)
    assert np.isfinite(predictions).all() and (predictions[:, 2:] > 0).all()
    assert np.array_equal(predictions[0], np.asarray(sequence['boxes'][0], np.float32))
    label_path = directory / (name + '_labels.csv')
    assert sha(label_path) == artifacts[label_path.name]
    with label_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = [frame for frame in sequence['valid_frames'] if frame > 0 and frame % 5 == 0]
    assert [int(row['frame']) for row in rows] == expected and rows
    for row in rows:
        assert row['sequence'] == name and row['partition'] == partition
        box = predictions[int(row['frame'])]
        gt = np.asarray(sequence['boxes'][int(row['frame'])], dtype=np.float64)
        assert np.isfinite(gt).all() and (gt[2:] > 0).all()
        left = np.maximum(box[:2], gt[:2])
        right = np.minimum(box[:2] + box[2:], gt[:2] + gt[2:])
        intersection = np.maximum(0, right - left).prod()
        iou = float(intersection / (box[2:].prod() + gt[2:].prod() - intersection))
        assert np.isfinite(iou) and 0 <= iou <= 1
        assert float(row['final_iou']) == iou, (name, row['frame'], row['final_iou'], iou)
    return dict(sequence=name, frames=len(predictions), samples=len(rows),
                partition=partition, prediction_sha256=artifacts[relative],
                labels_sha256=artifacts[label_path.name], all_sample_ious_exact=True)


def checked_json(root, relative, digest):
    path = root / relative
    assert sha(path) == digest
    return json.loads(path.read_bytes())


def finalize(root, draft_relative, review_relative, report_relative, output_relative):
    """No output plan is written until all eight real caches pass acceptance."""
    report_path, output_path = root / report_relative, root / output_relative
    assert not report_path.exists() and not output_path.exists()
    draft_path = root / draft_relative
    draft = json.loads(draft_path.read_bytes())
    assert draft['mode'] == 'protected_e0_offline_m1_training'
    assert draft['launch_allowed'] is False and draft['split'] == 'train'
    assert draft['test_eligible'] is False and len(draft['caches']) == 8
    assert all(entry['receipt_sha256'] is entry['native_sha256'] is None
               for entry in draft['caches'])
    review_path = root / review_relative
    review = json.loads(review_path.read_bytes())
    assert review['stage'] == 'M1_full_collection_verifier'
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert review['reviewed_training_draft'] == draft_relative
    assert review['reviewed_training_draft_sha256'] == sha(draft_path)
    sources = dict(draft['source_sha256'])
    sources.update({VERIFIER: sha(root / VERIFIER), TESTS: sha(root / TESTS)})
    assert review['source_sha256'] == sources
    for relative, digest in sources.items():
        assert sha(root / relative) == digest
    training_review = checked_json(root, draft['source_review'], draft['source_review_sha256'])
    assert training_review['status'] == 'SOURCE_PASS' and training_review['blocking_issues'] == 0
    assert training_review['source_sha256'] == draft['source_sha256']
    collection = checked_json(root, draft['collection_plan'], draft['collection_plan_sha256'])
    collector_review = checked_json(root, collection['source_review'], collection['source_review_sha256'])
    parent = checked_json(root, collector_review['parent_review'], collector_review['parent_review_sha256'])
    collector_stage = parent['stage_conclusions']['collector']
    assert collector_stage['status'] == 'SOURCE_PASS' and collector_stage['blocking_issues'] == 0
    original_draft = checked_json(root, collector_review['reviewed_draft'],
                                  collector_review['reviewed_draft_sha256'])
    assert collection['source_sha256'] == original_draft['source_sha256']
    assert collection['input_sha256'] == original_draft['input_sha256']
    binding = collector_stage['binding']
    assert binding['draft'] == collector_review['reviewed_draft']
    assert binding['draft_sha256'] == collector_review['reviewed_draft_sha256']
    for key in ('source_sha256', 'input_sha256'):
        fingerprint = hashlib.sha256(json.dumps(
            collection[key], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert fingerprint == binding[key.replace('_sha256', '_map_canonical_json_sha256')]
    for relative, digest in collection['input_sha256'].items():
        assert sha(root / relative) == digest
    plan = copy.deepcopy(draft)
    for entry in plan['caches']:
        entry['receipt_sha256'] = sha(root / 'runs' / entry['run'] / 'COMPLETE.json')
        entry['native_sha256'] = sha(root / 'runs/pipeline_tasks' / entry['task'] / 'status.json')
    data, used = load_collection(root, plan)
    manifest = json.loads((root / collection['manifest']).read_bytes())
    sequences = {row['name']: row for row in manifest['train']}
    fit, results = set(plan['fit_sequences']), []
    for entry in plan['caches']:
        directory = root / 'runs' / entry['run']
        receipt = json.loads((directory / 'COMPLETE.json').read_bytes())
        config = json.loads((directory / 'config.json').read_bytes())
        assert {p.stem for p in (directory / 'predictions').glob('*.npy')} == set(config['selected_sequences'])
        for name in config['selected_sequences']:
            results.append(verify_predictions(directory, sequences[name],
                'fit' if name in fit else 'holdout', receipt['artifacts_sha256']))
    assert len(results) == collection['expected_sequences'] == 882
    assert sum(row['frames'] for row in results) == collection['expected_frames'] == 459435
    assert sum(row['samples'] for row in results) == 91091
    counts = {part: torch.bincount(localization_targets(values['iou']), minlength=3).tolist()
              for part, values in data.items()}
    assert len(data['fit']['iou']) == 71843 and len(data['holdout']['iou']) == 19248
    report = dict(status='FULL_PAIRED_TRAIN_COLLECTION_ACCEPTED', split='train',
        test_eligible=False, scientific_result_audit_accepted=False,
        training_draft=draft_relative, training_draft_sha256=sha(draft_path),
        verifier_review=review_relative, verifier_review_sha256=sha(review_path),
        source_sha256=sources, collection_plan=plan['collection_plan'],
        collection_plan_sha256=plan['collection_plan_sha256'], caches=plan['caches'],
        sequences=882, frames=459435, sampled_states=91091,
        classes_reliable_uncertain_lost=counts, trajectories=results, inputs_sha256=used,
        full_image_pixel_hashes_verified=False,
        scope='Full returned trajectories and every sampled TRAIN IoU checked exactly. '
              'Paired online parity remains collector execution evidence. '
              'Component holdout was seen by upstream E0; no independent generalization '
              'or identity/recovery/benchmark improvement is established.')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    plan.update(launch_allowed=True, collection_acceptance=report_relative,
                collection_acceptance_sha256=sha(report_path),
                collection_verifier_review=review_relative,
                collection_verifier_review_sha256=sha(review_path))
    output_path.write_text(json.dumps(plan, indent=2) + '\n')
    return dict(report=report_relative, report_sha256=sha(report_path),
                plan=output_relative, plan_sha256=sha(output_path), class_counts=counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output-plan', required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(ROOT, args.draft, args.review, args.report, args.output_plan)), flush=True)


if __name__ == '__main__':
    main()
