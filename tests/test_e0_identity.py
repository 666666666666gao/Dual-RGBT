"""Synthetic checks for identity inputs; not tracking or identity efficacy."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_rgbt.e0_identity import (
    E0CandidateIdentity, identity_set_loss, initial_identity_anchor, primary_candidate_evidence)


class EncoderOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def encode(self, templates, search):
        self.calls.append((templates.clone(), search.clone()))
        return search.mean().expand(1, 256, 768), 385


class IdentityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        torch.set_num_threads(1)

    def test_locality_and_unobserved_region(self):
        grid = torch.arange(16).float()[None].expand(16, -1)
        features = grid.reshape(1, 256, 1).expand(-1, -1, 768)
        before = features.clone()
        descriptors, coverage = primary_candidate_evidence(features, torch.tensor([
            [.25, .5, .1, .1], [.75, .5, .1, .1], [2., 2., .1, .1]]))
        self.assertEqual(descriptors.shape, (3, 768))
        self.assertTrue((descriptors[1] > descriptors[0]).all())
        self.assertEqual(coverage.tolist(), [1., 1., 0.])
        self.assertEqual(descriptors[2].abs().sum().item(), 0.)
        self.assertTrue(torch.equal(features, before))

    def test_initial_anchor_uses_two_deep_encoder_calls_and_preserves_inputs(self):
        model = EncoderOnly().eval()
        templates = torch.randn(1, 2, 6, 128, 128)
        search = torch.cat([torch.ones(1, 3, 256, 256), torch.full((1, 3, 256, 256), 2.)], 1)
        before = templates.clone()
        anchor = initial_identity_anchor(model, templates, search, torch.tensor([[.5, .5, .2, .2]]))
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(anchor.shape, (1, 2, 768))
        self.assertTrue(torch.allclose(anchor[:, 0], torch.ones(1, 768)))
        self.assertTrue(torch.allclose(anchor[:, 1], torch.full((1, 768), 2.)))
        self.assertTrue(torch.equal(templates, before))
        self.assertFalse(anchor.requires_grad)

    def test_unobserved_candidates_cannot_win_and_no_match_is_available(self):
        head = E0CandidateIdentity().eval()
        anchor, candidates = torch.randn(1, 2, 768), torch.randn(1, 3, 2, 768)
        coverage = torch.zeros(1, 3, 2)
        logits = head(anchor, candidates, coverage)
        self.assertEqual(logits.argmax(1).item(), 0)
        self.assertTrue(torch.isneginf(logits[0, 1:]).all())
        self.assertEqual(identity_set_loss(logits, torch.tensor([0])).item(), 0.)
        with self.assertRaises(AssertionError):
            identity_set_loss(logits, torch.tensor([1]))

    def test_candidate_permutation_and_covered_modality_only(self):
        head = E0CandidateIdentity().eval()
        anchor, candidates = torch.randn(1, 2, 768), torch.randn(1, 3, 2, 768)
        coverage = torch.tensor([[[1., 0.], [1., 1.], [.5, .5]]])
        before = candidates.clone()
        logits = head(anchor, candidates, coverage)
        order = torch.tensor([1, 0, 2])
        permuted = head(anchor, candidates[:, order], coverage[:, order])
        self.assertTrue(torch.equal(permuted[:, :1], logits[:, :1]))
        self.assertTrue(torch.equal(permuted[:, 1:], logits[:, 1:][:, order]))
        candidates[:, 0, 1] += 100
        changed = head(anchor, candidates, coverage)
        self.assertTrue(torch.equal(logits, changed))
        self.assertTrue(torch.equal(candidates[:, :, 0], before[:, :, 0]))

    def test_set_loss_updates_head_and_strict_reload(self):
        head = E0CandidateIdentity()
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
        anchor, candidates = torch.randn(3, 2, 768), torch.randn(3, 4, 2, 768)
        coverage = torch.ones(3, 4, 2)
        target = torch.tensor([0, 2, 4])
        initial = identity_set_loss(head(anchor, candidates, coverage), target).item()
        for _ in range(8):
            optimizer.zero_grad()
            loss = identity_set_loss(head(anchor, candidates, coverage), target)
            loss.backward()
            optimizer.step()
        head.eval()
        self.assertLess(identity_set_loss(head(anchor, candidates, coverage), target).item(), initial)
        restored = E0CandidateIdentity().eval()
        restored.load_state_dict(head.state_dict(), strict=True)
        self.assertTrue(torch.equal(head(anchor, candidates, coverage),
                                    restored(anchor, candidates, coverage)))


if __name__ == '__main__':
    unittest.main()
