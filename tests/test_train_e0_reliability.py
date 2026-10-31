"""Synthetic CPU checks, not real-data training or source-review admission."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_e0_reliability import (
    E0Reliability, fit_normalization, load_collection, metrics, sha, train_head)


class TrainingTest(unittest.TestCase):
    def test_fit_normalization_constant_channel(self):
        fit = torch.tensor([[1., 2.], [3., 2.]])
        mean, scale = fit_normalization(fit)
        self.assertEqual(mean.tolist(), [2., 2.])
        self.assertEqual(scale.tolist(), [1., 1.])
        self.assertTrue(torch.equal((fit - mean) / scale,
                                    torch.tensor([[-1., 0.], [1., 0.]])))
        held = torch.tensor([[101., 2.]])
        self.assertEqual(((held - mean) / scale).tolist(), [[99., 0.]])

    def test_metrics_reports_failure_and_false_interruption(self):
        logits = torch.tensor([[8., 0., 0.], [0., 8., 0.], [8., 0., 0.], [0., 0., 8.]])
        with self.assertRaises(AssertionError):
            metrics(torch.nn.Identity(), logits, torch.tensor([.9, .7, .01, .02]), 2)

    def test_complete_three_class_report(self):
        logits = torch.tensor([[8., 0., 0.], [0., 8., 0.], [8., 0., 0.],
                               [0., 0., 8.], [0., 8., 0.]])
        result = metrics(torch.nn.Identity(), logits, torch.tensor([.9, .7, .01, .02, .3]), 2)
        self.assertEqual(result['confusion_true_rows_predicted_columns'], [[1, 1, 0], [0, 1, 0], [1, 0, 1]])
        self.assertEqual(result['lost_recall'], .5)
        self.assertEqual(result['reliable_interrupted_fraction'], .5)

    def test_prefix_collection_rejected(self):
        with self.assertRaises(AssertionError):
            load_collection(Path('.'), dict(caches=[dict(run='prefix')]))

    def test_training_save_reload_and_latest_five(self):
        fit = torch.zeros(48, 914)
        held = torch.zeros(18, 914)
        for features in (fit, held):
            labels = torch.arange(len(features)) % 3
            features[torch.arange(len(features)), labels] = 3
        # Holdout shift must not enter the saved preprocessing statistics.
        held[:, 3] = 100
        data = dict(fit=dict(features=fit.clone(), iou=torch.tensor([.9, .3, .01] * 16)),
                    holdout=dict(features=held.clone(), iou=torch.tensor([.9, .3, .01] * 6)))
        plan = dict(seed=2027, epochs=6, batch_states=12, lr=.01, weight_decay=0.,
                    device='cpu', checkpoint_sha256='synthetic-no-e0-loaded')
        torch.manual_seed(plan['seed'])
        initial = E0Reliability().state_dict()
        mean, scale = fit_normalization(fit)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'synthetic_training_only'
            with contextlib.redirect_stdout(io.StringIO()):
                train_head(data, plan, output, 'synthetic-not-reviewed', {})
            history = json.loads((output / 'history.json').read_bytes())
            receipt = json.loads((output / 'COMPLETE.json').read_bytes())
            latest = torch.load(output / 'latest.pth', map_location='cpu')
            best = torch.load(output / 'best.pth', map_location='cpu')
            numbered = sorted(path.name for path in output.glob('epoch_*.pth'))
            self.assertEqual(numbered, ['epoch_%03d.pth' % epoch for epoch in range(2, 7)])
            self.assertEqual(receipt['optimizer_updates'], 24)
            self.assertTrue(receipt['strict_reload_bit_equal'])
            self.assertFalse(receipt['protected_e0_loaded'])
            self.assertFalse(receipt['online_policy_changed'])
            self.assertTrue(torch.equal(latest['feature_mean'], mean))
            self.assertTrue(torch.equal(latest['feature_scale'], scale))
            self.assertEqual(latest['feature_mean'][3].item(), 0)
            self.assertEqual(latest['epoch'], 6)
            self.assertLess(history[-1]['fit']['cross_entropy'], history[0]['fit']['cross_entropy'])
            self.assertTrue(any(not torch.equal(value, latest['net'][key])
                                for key, value in initial.items()))
            head = E0Reliability().eval()
            head.load_state_dict(latest['net'], strict=True)
            reloaded = metrics(head, (held - mean) / scale, data['holdout']['iou'], 12)
            self.assertEqual(reloaded, history[-1]['holdout'])
            recalls = [entry['holdout']['macro_recall'] for entry in history]
            self.assertEqual(best['epoch'], recalls.index(max(recalls)) + 1)
            self.assertEqual(sha(output / 'best.pth'), receipt['artifacts_sha256']['best.pth'])
            for relative, digest in receipt['artifacts_sha256'].items():
                self.assertEqual(sha(output / relative), digest)

    def test_complete_shard_loading_and_artifact_tamper_rejection(self):
        names = ['synthetic_%03d' % index for index in range(882)]
        plan = dict(caches=[], fit_sequences=names[:706], holdout_sequences=names[706:],
                    checkpoint_sha256='synthetic-no-tracker-run')
        fit = set(plan['fit_sequences'])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(dict(train=[
                dict(name=name, rgb=['synthetic'] * 6, valid_frames=list(range(6)))
                for name in names], dev=[], test_sequences=[])))
            for shard in range(8):
                run, task = 'synthetic_shard%d' % shard, 'synthetic_task%d' % shard
                directory = root / 'runs' / run
                directory.mkdir(parents=True)
                native_path = root / 'runs/pipeline_tasks' / task / 'status.json'
                native_path.parent.mkdir(parents=True)
                native_path.write_text(json.dumps(dict(
                    state='COMPLETE', exit_code=0, command=['synthetic-only'])))
                selected = names[shard::8]
                collection_plan = dict(checkpoint_sha256=plan['checkpoint_sha256'],
                    fit_sequences=plan['fit_sequences'], holdout_sequences=plan['holdout_sequences'],
                    full_collection_allowed=True, cost_weight=0., matmul_tf32=False, cudnn_tf32=False,
                    manifest='manifest.json', input_sha256={'manifest.json': sha(manifest_path)})
                config = dict(task=task, run=run, split='train', shard=shard,
                    selected_sequences=selected, native_command=['synthetic-only'], plan=collection_plan)
                (directory / 'config.json').write_text(json.dumps(config))
                (directory / 'progress.json').write_text(json.dumps([
                    dict(sequence=name, frames=6, matched_tracked_frames=5,
                         all_online_states_bit_equal=True, feature_states=1) for name in selected]))
                for name in selected:
                    index = names.index(name)
                    np.save(directory / (name + '_features.npy'),
                            np.full((1, 914), index, dtype=np.float32))
                    with (directory / (name + '_labels.csv')).open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, ['sequence', 'partition', 'frame', 'final_iou'])
                        writer.writeheader()
                        writer.writerow(dict(sequence=name, partition='fit' if name in fit else 'holdout',
                                             frame=5, final_iou=(.9, .3, .01)[index % 3]))
                receipt = dict(task=task, split='train', status='PAIRED_TRAIN_CACHE_COMPLETE',
                    checkpoint_sha256=plan['checkpoint_sha256'], optimizer_updates=0, backward_calls=0,
                    frozen_state_tensors=265, all_online_states_bit_equal=True, feature_width=914,
                    feature_states=len(selected), sequences=len(selected),
                    frames=6 * len(selected), tracked_frames=5 * len(selected),
                    artifacts_sha256={p.name: sha(p) for p in directory.iterdir() if p.is_file()})
                receipt_path = directory / 'COMPLETE.json'
                receipt_path.write_text(json.dumps(receipt))
                plan['caches'].append(dict(run=run, task=task, receipt_sha256=sha(receipt_path),
                                           native_sha256=sha(native_path)))
            data, used = load_collection(root, plan)
            self.assertEqual(data['fit']['features'].shape, (706, 914))
            self.assertEqual(data['holdout']['features'].shape, (176, 914))
            self.assertEqual(len(used), 1797)
            progress_path = root / 'runs/synthetic_shard0/progress.json'
            original_progress = progress_path.read_bytes()
            progress_path.write_text('[]')
            receipt_path = root / 'runs/synthetic_shard0/COMPLETE.json'
            original_receipt = receipt_path.read_bytes()
            receipt = json.loads(original_receipt)
            receipt['artifacts_sha256']['progress.json'] = sha(progress_path)
            receipt_path.write_text(json.dumps(receipt))
            original_hash = plan['caches'][0]['receipt_sha256']
            plan['caches'][0]['receipt_sha256'] = sha(receipt_path)
            with self.assertRaises(AssertionError):
                load_collection(root, plan)
            progress_path.write_bytes(original_progress)
            receipt_path.write_bytes(original_receipt)
            plan['caches'][0]['receipt_sha256'] = original_hash
            label_path = root / 'runs/synthetic_shard0' / (names[0] + '_labels.csv')
            label_path.write_text('tampered')
            with self.assertRaises(AssertionError):
                load_collection(root, plan)


if __name__ == '__main__':
    unittest.main()
