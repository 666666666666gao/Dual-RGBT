"""Focused CPU checks for full-trajectory and label acceptance."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from verify_e0_paired_train_collection import finalize, sha, verify_predictions


class PredictionTest(unittest.TestCase):
    def test_exact_collector_arithmetic_and_rehashed_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'predictions').mkdir()
            sequence = dict(name='synthetic', rgb=['image'] * 11,
                boxes=[[2., 3., 7., 9.]] * 11, valid_frames=list(range(11)))
            predictions = np.tile(np.array([2., 3., 7., 9.], np.float32), (11, 1))
            predictions[5] = [3.1234567, 2.567891, 6.456789, 8.345678]
            path = directory / 'predictions/synthetic.npy'
            labels = directory / 'synthetic_labels.csv'
            rows = []
            for frame in (5, 10):
                box, gt = predictions[frame], np.asarray(sequence['boxes'][frame], np.float64)
                intersection = np.maximum(0, np.minimum(box[:2] + box[2:], gt[:2] + gt[2:]) -
                                           np.maximum(box[:2], gt[:2])).prod()
                iou = float(intersection / (box[2:].prod() + gt[2:].prod() - intersection))
                rows.append(dict(sequence='synthetic', partition='fit', frame=frame, final_iou=iou))
            np.save(path, predictions)
            with labels.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            artifacts = {'predictions/synthetic.npy': sha(path), labels.name: sha(labels)}
            result = verify_predictions(directory, sequence, 'fit', artifacts)
            self.assertEqual((result['frames'], result['samples']), (11, 2))
            original_labels = labels.read_bytes()
            for case in ('undeclared', 'truncated', 'wrong_dtype', 'wrong_initial',
                         'nonfinite', 'negative_size', 'wrong_iou', 'wrong_frames', 'wrong_partition'):
                with self.subTest(case=case):
                    changed = predictions.copy()
                    modified = dict(artifacts)
                    if case == 'undeclared':
                        del modified['predictions/synthetic.npy']
                    elif case == 'truncated':
                        changed = changed[:-1]
                    elif case == 'wrong_dtype':
                        changed = changed.astype(np.float64)
                    elif case == 'wrong_initial':
                        changed[0, 0] += 1
                    elif case == 'nonfinite':
                        changed[3, 0] = np.nan
                    elif case == 'negative_size':
                        changed[3, 2] = -1
                    else:
                        bad_rows = [dict(row) for row in rows]
                        key, value = {'wrong_iou': ('final_iou', .123),
                                      'wrong_frames': ('frame', 4),
                                      'wrong_partition': ('partition', 'holdout')}[case]
                        bad_rows[0][key] = value
                        with labels.open('w', newline='') as handle:
                            writer = csv.DictWriter(handle, list(rows[0]))
                            writer.writeheader()
                            writer.writerows(bad_rows)
                        modified[labels.name] = sha(labels)
                    np.save(path, changed)
                    if case != 'undeclared':
                        modified['predictions/synthetic.npy'] = sha(path)
                    with self.assertRaises(AssertionError):
                        verify_predictions(directory, sequence, 'fit', modified)
                    labels.write_bytes(original_labels)
            np.save(path, predictions)
            verify_predictions(directory, sequence, 'fit', artifacts)

    def test_unreviewed_finalization_writes_no_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft = dict(mode='protected_e0_offline_m1_training', launch_allowed=False,
                split='train', test_eligible=False,
                caches=[dict(receipt_sha256=None, native_sha256=None) for _ in range(8)])
            (root / 'draft.json').write_text(json.dumps(draft))
            (root / 'review.json').write_text(json.dumps(dict(
                stage='M1_full_collection_verifier', status='SOURCE_BLOCKED', blocking_issues=1)))
            with self.assertRaises(AssertionError):
                finalize(root, 'draft.json', 'review.json', 'acceptance.json', 'final.json')
            self.assertFalse((root / 'acceptance.json').exists())
            self.assertFalse((root / 'final.json').exists())


if __name__ == '__main__':
    unittest.main()
