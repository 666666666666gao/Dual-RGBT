"""Check explicit KEEP semantics, masks and relative targets before training."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_rgbt.e0_identity import E0CandidateIdentity
from dual_rgbt.e0_relative_identity import E0RelativeIdentity, relative_action_targets


class RelativeIdentityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        torch.set_num_threads(1)
        self.head = E0RelativeIdentity().eval()
        self.anchor = torch.randn(2, 2, 768)
        self.current = torch.randn(2, 2, 768)
        self.current_coverage = torch.ones(2, 2)
        self.candidates = torch.randn(2, 3, 2, 768)
        self.coverage = torch.ones(2, 3, 2)

    def scores(self):
        return self.head(self.anchor, self.current, self.current_coverage,
                         self.candidates, self.coverage)

    def test_original_identity_weights_load_exactly_and_no_match_is_unused(self):
        original = E0CandidateIdentity().eval()
        self.head.load_state_dict(original.state_dict(), strict=True)
        expected = original(
            self.anchor, torch.cat([self.current[:, None], self.candidates], 1),
            torch.cat([self.current_coverage[:, None], self.coverage], 1))[:, 1:]
        before = self.scores()
        self.assertTrue(torch.equal(before, expected))
        self.assertFalse(self.head.no_match.requires_grad)
        self.head.no_match.data.fill_(10000)
        self.assertTrue(torch.equal(before, self.scores()))

    def test_unobserved_current_is_keep_only_and_candidates_remain_masked(self):
        self.current_coverage[0] = .5
        self.coverage[1, 1] = 0
        scores = self.scores()
        self.assertEqual(scores[0, 0].item(), 0)
        self.assertTrue(torch.isneginf(scores[0, 1:]).all())
        self.assertTrue(torch.isneginf(scores[1, 2]))
        self.assertEqual(scores.argmax(1)[0].item(), 0)

    def test_exact_inference_tie_prefers_current_e0(self):
        self.candidates[:, 0] = self.current
        self.coverage[:, 1:] = 0
        scores = self.scores()
        self.assertTrue(torch.equal(scores[:, 0], scores[:, 1]))
        self.assertEqual(scores.argmax(1).tolist(), [0, 0])

    def test_targets_preserve_equal_e0_and_choose_best_improvement(self):
        current = torch.tensor([.8, .6, .05, .01, .3, .2], dtype=torch.float64)
        candidates = torch.tensor([
            [.8, .7, .6], [.7, .9, .8], [.5, .5, .1],
            [.49, .1, .05], [.8, .9, .7], [.9, .8, .7]], dtype=torch.float64)
        cov = torch.ones(6, 3, 2)
        current_cov = torch.ones(6, 2)
        cov[4, 1] = 0
        current_cov[5] = 0
        targets, trainable = relative_action_targets(current, candidates, current_cov, cov)
        self.assertEqual(targets.tolist(), [0, 2, 1, 0, 1, 0])
        self.assertEqual(trainable.tolist(), [True, True, True, True, True, False])
        self.assertTrue((candidates[trainable].gather(
            1, (targets[trainable] - 1).clamp_min(0)[:, None]).squeeze(1)[
                targets[trainable] > 0] >= .5).all())

    def test_masked_cross_entropy_updates_only_matcher_and_strict_reloads(self):
        self.coverage[0, 2] = 0
        optimizer = torch.optim.AdamW(
            [p for p in self.head.parameters() if p.requires_grad], lr=.01)
        target = torch.tensor([0, 2])
        optimizer.zero_grad()
        before = {k: v.clone() for k, v in self.head.state_dict().items()}
        loss = torch.nn.functional.cross_entropy(self.scores(), target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(self.head.no_match.grad)
        self.assertTrue(all(torch.isfinite(p.grad).all()
                            for p in self.head.parameters() if p.grad is not None))
        optimizer.step()
        self.assertTrue(any(not torch.equal(v, before[k])
                            for k, v in self.head.state_dict().items()))
        restored = E0RelativeIdentity().eval()
        restored.load_state_dict(self.head.state_dict(), strict=True)
        self.assertTrue(torch.equal(self.scores(), restored(
            self.anchor, self.current, self.current_coverage, self.candidates, self.coverage)))


if __name__ == "__main__":
    unittest.main()
