"""Multi-positive supervision, cache corruption and standalone-head checks."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]
from collect_e0_identity_train import global_windows, normalized_cxcywh, overlap_labels
from train_e0_identity import E0CandidateIdentity, metrics, multi_positive_loss, train_head
from verify_e0_identity_collection import crop_geometry, load_collection, verify_arrays


class IdentityTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        torch.set_num_threads(1)

    def test_multiple_positives_are_jointly_rewarded_and_unknowns_have_no_gradient(self):
        logits = torch.tensor([[0., 1., 2., 100.]], requires_grad=True)
        labels = torch.tensor([[0, 1, 1, -1]], dtype=torch.int8)
        loss = multi_positive_loss(logits, labels)
        expected = torch.logsumexp(logits[0, :3], 0) - torch.logsumexp(logits[0, 1:3], 0)
        self.assertTrue(torch.equal(loss, expected))
        loss.backward()
        self.assertEqual(logits.grad[0, 3].item(), 0.)
        self.assertLess(logits.grad[0, 1].item(), 0.)
        self.assertLess(logits.grad[0, 2].item(), 0.)
        self.assertGreater(logits.grad[0, 0].item(), 0.)
        order = torch.tensor([0, 2, 1, 3])
        self.assertTrue(torch.allclose(loss.detach(),
                                       multi_positive_loss(logits.detach()[:, order], labels[:, order])))

    def test_no_match_supervision_and_all_ignored_rows_are_not_negatives(self):
        logits = torch.tensor([[3., 0., float('-inf')]])
        labels = torch.tensor([[1, 0, -1]], dtype=torch.int8)
        self.assertLess(multi_positive_loss(logits, labels).item(), .05)
        with self.assertRaises(AssertionError):
            multi_positive_loss(logits, torch.full_like(labels, -1))
        with self.assertRaises(AssertionError):
            multi_positive_loss(logits, torch.tensor([[0, 0, 1]], dtype=torch.int8))

    def test_inference_metrics_keep_ignored_competitors(self):
        class Fixed(torch.nn.Module):
            def forward(self, anchor, features, coverage):
                return torch.tensor([[0., 1., 9.], [9., 0., 1.], [0., 9., 1.]])[:len(anchor)]
        data = dict(anchor=torch.zeros(3, 2, 768), features=torch.zeros(3, 2, 2, 768),
                    coverage=torch.ones(3, 2, 2),
                    labels=torch.tensor([[0, 1, -1], [1, 0, 0], [-1, 0, -1]], dtype=torch.int8))
        result = metrics(Fixed(), data, 3)
        self.assertEqual(result['supervised_sets'], 2)
        self.assertEqual(result['ambiguous_sets'], 1)
        self.assertEqual(result['supervised_top1_positive_fraction'], .5)
        self.assertEqual(result['supervised_ignored_picks'], 1)
        self.assertEqual(result['correct_no_match_picks'], 1)
        self.assertEqual(result['correct_candidate_picks'], 0)

    @staticmethod
    def sequence_arrays():
        sequence = dict(name='synthetic', boxes=[[0., 0., 10., 10.]] * 3,
                        rgb=['synthetic'] * 3, valid_frames=[0, 1, 2])
        boxes = np.tile(np.asarray([100., 100., 10., 10.], np.float32), (1, 24, 1))
        boxes[0, 0] = [0, 0, 10, 10]
        coverage = np.ones((1, 24, 2), np.float32)
        iou, labels = overlap_labels(boxes[0], coverage[0], sequence['boxes'][1])
        previous = np.asarray(sequence['boxes'], np.float32)
        geometry = crop_geometry(previous[0])
        crops = np.asarray([previous[0]] + global_windows(200, 200) +
                           [np.asarray([100., 100., 10., 10.], np.float32)] * 2)
        arrays = dict(box=boxes, features=np.zeros((1, 24, 2, 768), np.float32),
            coverage=coverage, score=np.zeros((1, 24)), query=np.repeat(np.arange(12), 2)[None],
            modality=np.tile([0, 1], (1, 12)),
            geometry=np.stack([crop_geometry(box) for box in crops])[None],
            crop_boxes=crops[None],
            refinement_parents=np.asarray([[2, 3]], np.int64),
            frame=np.asarray([1], np.int64), iou=iou[None], labels=labels[None],
            anchor=np.zeros((2, 768), np.float32), anchor_geometry=geometry,
            anchor_box=normalized_cxcywh(previous[:1], geometry)[0], predictions=previous)
        return sequence, arrays

    def test_reconstructed_labels_and_candidate_queries_reject_corruption(self):
        sequence, arrays = self.sequence_arrays()
        self.assertEqual(verify_arrays(arrays, sequence, 3, [(200, 200)])['positive_sets'], 1)
        for key, index, bad in [('labels', (0, 1), 0), ('iou', (0, 0), .3),
                                 ('refinement_parents', (0, 0), 19),
                                 ('crop_boxes', (0, 0, 0), 5.),
                                 ('crop_boxes', (0, 1, 0), 5.),
                                 ('crop_boxes', (0, 10, 0), 5.),
                                 ('geometry', (0, 0, 0), 5.)]:
            changed = {name: value.copy() for name, value in arrays.items()}
            changed[key][index] = bad
            with self.assertRaises(AssertionError, msg=key):
                verify_arrays(changed, sequence, 3, [(200, 200)])
        changed = {name: value.copy() for name, value in arrays.items()}
        changed['anchor_geometry'][0] += 5.
        changed['anchor_box'] = normalized_cxcywh(
            changed['predictions'][:1], changed['anchor_geometry'])[0]
        with self.assertRaises(AssertionError):
            verify_arrays(changed, sequence, 3, [(200, 200)])

    def test_smoke_cannot_enter_full_collection_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'smoke.json'
            path.write_text(json.dumps(dict(phase='smoke', shards=1, jobs=[])))
            from benchmark_inputs import sha
            with self.assertRaises(AssertionError):
                load_collection(root, path.name, sha(path))

    def test_standalone_training_updates_and_preserves_final_five(self):
        anchor = torch.randn(8, 2, 768)
        features = torch.randn(8, 3, 2, 768)
        features[:, 0] = anchor
        data = {part: dict(anchor=anchor.clone(), features=features.clone(),
                          coverage=torch.ones(8, 3, 2),
                          labels=torch.tensor([[0, 1, -1, 0]] * 8, dtype=torch.int8))
                for part in ('fit', 'holdout')}
        original = {key: value.clone() for key, value in data['fit'].items()}
        plan = dict(seed=2027, device='cpu', epochs=6, batch_sets=4, lr=.003,
                    weight_decay=0., checkpoint_sha256='synthetic-no-e0')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'synthetic'
            with contextlib.redirect_stdout(io.StringIO()):
                train_head(data, plan, output, 'synthetic-plan', 'synthetic-acceptance')
            history = json.loads((output / 'history.json').read_bytes())
            receipt = json.loads((output / 'COMPLETE.json').read_bytes())
            self.assertEqual(receipt['optimizer_updates'], 12)
            self.assertTrue(receipt['strict_reload_bit_equal'])
            self.assertFalse(receipt['protected_e0_loaded'])
            self.assertFalse(receipt['online_policy_changed'])
            self.assertEqual([p.name for p in sorted(output.glob('epoch_*.pth'))],
                             ['epoch_%03d.pth' % i for i in range(2, 7)])
            self.assertLess(history[-1]['fit']['multi_positive_loss'],
                            history[0]['fit']['multi_positive_loss'])
            for key, value in original.items():
                self.assertTrue(torch.equal(value, data['fit'][key]))


if __name__ == '__main__':
    unittest.main()
