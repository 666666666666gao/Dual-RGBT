"""Synthetic retention workflows; no GPU work or native locking is exercised."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'tools/run_size_tail_epochs_keep_five.py'


class KeepFiveTests(unittest.TestCase):
    def simulate(self, scores, fail_epoch=None, regression=False):
        tree = ast.parse(SOURCE.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        namespace = dict(
            argparse=argparse, json=json, os=os, Path=Path, subprocess=subprocess,
            time=SimpleNamespace(time=lambda: 0),
            sha=sha, check_files=lambda root, mapping: None,
            fcntl=SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None))
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'),
             namespace)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace['ROOT'] = root
            (root / 'runs/pipeline_tasks').mkdir(parents=True)
            (root / 'parent.pth').write_bytes(b'PROTECTED ORIGINAL PARENT')
            gate = dict(status='SCORED_COMPLETE', sequences=97, frames=54646)
            (root / 'gate.json').write_text(json.dumps(gate))
            (root / 'template.json').write_text('{}')
            plan = dict(
                formal_launch_allowed=True, epochs=10, source_sha256={},
                parent_control_score='gate.json', parent_control_score_sha256=sha(root / 'gate.json'),
                diagnostic_score='gate.json', diagnostic_score_sha256=sha(root / 'gate.json'),
                evaluation_template='template.json', max_existing_gpu_mib=3000,
                parent_metrics=dict(SR=56), protected_e1_metrics={}, max_sr_drop_pp=5)
            (root / 'plan.json').write_text(json.dumps(plan))
            commands = []

            class Process:
                def __init__(self, command, **kwargs):
                    commands.append(command)
                    self.pid = 123
                    child = command[command.index('--') + 1:]
                    task = command[command.index('--task') + 1]
                    run = child[child.index('--run') + 1]
                    epoch = int(run[-3:])
                    training = '--epoch' in child
                    self.returncode = int(training and epoch == fail_epoch)
                    native = root / 'runs/pipeline_tasks' / task
                    native.mkdir()
                    (native / 'status.json').write_text(json.dumps(dict(
                        command=child, state='FAILED' if self.returncode else 'COMPLETE',
                        exit_code=self.returncode)))
                    if not training:
                        assert child[child.index('--smoke-frames') + 1] == '0'
                        return
                    if self.returncode:
                        return
                    directory = root / 'runs' / run
                    directory.mkdir()
                    checkpoint = directory / ('epoch_%03d.pth' % epoch)
                    checkpoint.write_bytes(('SYNTHETIC EPOCH %d' % epoch).encode())
                    if epoch > 1:
                        resume = root / child[child.index('--resume') + 1]
                        assert sha(resume) == child[child.index('--resume-sha256') + 1]
                    (directory / 'COMPLETE.json').write_text(json.dumps(dict(
                        strict_reload_bit_equal=True, checkpoint_sha256=sha(checkpoint))))

                def poll(self):
                    return self.returncode

            def score(command, **kwargs):
                run = command[command.index('--run') + 1]
                epoch = int(run[-3:])
                output = root / command[command.index('--output') + 1]
                output.write_text(json.dumps(dict(
                    status='SCORED_COMPLETE', sequences=97, frames=54646,
                    PR=70, NPR=60 + epoch / 100, SR=scores[epoch - 1], fps=10)))

            with patch.object(sys, 'argv', [
                    'controller', '--plan', 'plan.json', '--name', 'pilot', '--gpu', '0']), \
                    patch.object(subprocess, 'Popen', Process), \
                    patch.object(subprocess, 'run', score), \
                    patch.object(subprocess, 'check_output', return_value='0\n0\n0\n0\n'), \
                    patch('builtins.print'):
                if fail_epoch:
                    with self.assertRaises(AssertionError):
                        namespace['main']()
                else:
                    namespace['main']()
            self.assertEqual((root / 'parent.pth').read_bytes(), b'PROTECTED ORIGINAL PARENT')
            directory = root / 'runs/pilot'
            if fail_epoch or regression:
                self.assertFalse((directory / 'SELECTION.json').exists())
                self.assertTrue((root / 'runs/pilot_train_e001/epoch_001.pth').exists())
                if regression:
                    state = json.loads((directory / 'status.json').read_text())
                    self.assertEqual(state['status'], 'REJECTED_REGRESSION')
                return
            state = json.loads((directory / 'status.json').read_text())
            selection = json.loads((directory / 'SELECTION.json').read_text())
            retained = list((root / 'runs').glob('pilot_train_*/epoch_*.pth'))
            self.assertEqual(len(retained), 5)
            self.assertIn(root / selection['selected']['checkpoint'], retained)
            self.assertEqual({int(path.stem[-3:]) for path in retained},
                             set(selection['selection_window']))
            self.assertEqual(state['status'], 'COMPLETE')
            for row in state['rows']:
                self.assertEqual(row['checkpoint_retained'],
                                 (root / row['checkpoint']).exists())
                self.assertTrue((root / row['score_path']).exists())
            return selection

    def test_plateau_keeps_five_and_selected_checkpoint(self):
        selected = self.simulate([56, 56.02, 56.01, 56.02, 56])
        self.assertEqual(selected['selection_window'], [1, 2, 3, 4, 5])
        self.assertEqual(selected['selected']['epoch'], 4)

    def test_later_stop_keeps_only_latest_five(self):
        selected = self.simulate([56, 56.2, 56.4, 56.6, 56.8, 57, 57.01, 57.02, 57.01])
        self.assertEqual(selected['selection_window'], [5, 6, 7, 8, 9])
        self.assertEqual(selected['selected']['epoch'], 8)

    def test_failed_training_preserves_existing_checkpoint_without_selection(self):
        self.simulate([56], fail_epoch=2)

    def test_dev_regression_preserves_checkpoint_without_selection(self):
        self.simulate([41], regression=True)


if __name__ == '__main__':
    unittest.main()
