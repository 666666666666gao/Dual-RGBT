"""Synthetic workflow checks; these do not measure model accuracy."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'tools/run_size_tail_epochs.py'
tree = ast.parse(SOURCE.read_text())
functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('stopping_epoch', 'best_recent')], type_ignores=[])
namespace = {}
exec(compile(functions, str(SOURCE), 'exec'), namespace)


class Decisions(unittest.TestCase):
    def test_no_stop_before_five_and_progress_resets_patience(self):
        stop = namespace['stopping_epoch']
        self.assertIsNone(stop([50, 49, 48, 47]))
        self.assertEqual(stop([50, 49, 48, 47, 46]), 5)
        self.assertEqual(stop([50, 50.2, 50.4, 50.3, 50.2, 50.3]), 6)

    def test_selection_uses_last_five_and_metric_order(self):
        rows = [dict(epoch=i, metrics=dict(SR=55, PR=70, NPR=i)) for i in range(1, 7)]
        rows[0]['metrics']['SR'] = 99
        rows[2]['metrics']['PR'] = 71
        self.assertEqual(namespace['best_recent'](rows)['epoch'], 3)


@unittest.skipUnless(sys.platform.startswith('linux'), 'Native controller uses Linux flock')
class Workflow(unittest.TestCase):
    def simulate(self, scores, fail_epoch=None, regression=False):
        spec = importlib.util.spec_from_file_location('size_tail_controller_test', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module.ROOT = root
            (root / 'runs/pipeline_tasks').mkdir(parents=True)
            (root / 'parent.pth').write_bytes(b'PROTECTED ORIGINAL PARENT')
            gate = dict(status='SCORED_COMPLETE', sequences=97, frames=54646)
            (root / 'gate.json').write_text(json.dumps(gate))
            (root / 'template.json').write_text('{}')
            plan = dict(formal_launch_allowed=True, epochs=10, source_sha256={},
                parent_control_score='gate.json', parent_control_score_sha256=module.sha(root / 'gate.json'),
                diagnostic_score='gate.json', diagnostic_score_sha256=module.sha(root / 'gate.json'),
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
                    self.returncode = 1 if training and epoch == fail_epoch else 0
                    native = root / 'runs/pipeline_tasks' / task
                    native.mkdir()
                    (native / 'status.json').write_text(json.dumps(dict(command=child,
                        state='COMPLETE' if self.returncode == 0 else 'FAILED', exit_code=self.returncode)))
                    if not training or self.returncode:
                        return
                    folder = root / 'runs' / run
                    folder.mkdir()
                    checkpoint = folder / ('epoch_%03d.pth' % epoch)
                    checkpoint.write_bytes(('SYNTHETIC EPOCH %d' % epoch).encode())
                    if epoch > 1:
                        resume = root / child[child.index('--resume') + 1]
                        assert module.sha(resume) == child[child.index('--resume-sha256') + 1]
                    (folder / 'COMPLETE.json').write_text(json.dumps(dict(strict_reload_bit_equal=True,
                        checkpoint_sha256=module.sha(checkpoint))))

                def poll(self):
                    return self.returncode

            def score(command, **kwargs):
                run = command[command.index('--run') + 1]
                epoch = int(run[-3:])
                output = root / command[command.index('--output') + 1]
                output.write_text(json.dumps(dict(status='SCORED_COMPLETE', sequences=97, frames=54646,
                    PR=70, NPR=60 + epoch/100, SR=scores[epoch-1], fps=10)))

            with patch.object(sys, 'argv', ['controller', '--plan', 'plan.json', '--name', 'pilot', '--gpu', '0']), \
                    patch.object(module.subprocess, 'Popen', Process), \
                    patch.object(module.subprocess, 'run', score), \
                    patch.object(module.subprocess, 'check_output', return_value='2200\n2200\n2200\n2200\n'), \
                    patch('builtins.print'):
                if fail_epoch:
                    with self.assertRaises(AssertionError):
                        module.main()
                else:
                    module.main()
            self.assertEqual((root / 'parent.pth').read_bytes(), b'PROTECTED ORIGINAL PARENT')
            directory = root / 'runs/pilot'
            if fail_epoch:
                self.assertFalse((directory / 'SELECTION.json').exists())
                self.assertEqual(len(commands), 3)  # train1, dev1, failed train2
                self.assertTrue((root / 'runs/pilot_train_e001/epoch_001.pth').exists())
                return
            state = json.loads((directory / 'status.json').read_text())
            if regression:
                self.assertEqual(state['status'], 'REJECTED_REGRESSION')
                self.assertFalse((directory / 'SELECTION.json').exists())
                self.assertTrue((directory / 'REJECTION.json').exists())
                self.assertEqual(len(commands), 2)
                self.assertTrue((root / 'runs/pilot_train_e001/epoch_001.pth').exists())
                return
            selection = json.loads((directory / 'SELECTION.json').read_text())
            retained = list((root / 'runs').glob('pilot_train_*/epoch_*.pth'))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0], root / selection['selected']['checkpoint'])
            self.assertEqual(state['status'], 'COMPLETE')
            return selection

    def test_plateau_stops_at_five_and_keeps_only_selected(self):
        selection = self.simulate([56, 56.02, 56.01, 56.02, 56])
        self.assertEqual(selection['evaluated_epochs'], [1, 2, 3, 4, 5])
        self.assertEqual(selection['selected']['epoch'], 4)

    def test_resume_chain_rolling_retention_and_later_stop(self):
        selection = self.simulate([56, 56.2, 56.4, 56.6, 56.8, 57, 57.01, 57.02, 57.01])
        self.assertEqual(selection['selection_window'], [5, 6, 7, 8, 9])
        self.assertEqual(selection['selected']['epoch'], 8)

    def test_failed_training_cannot_be_selected_or_cleaned(self):
        self.simulate([56], fail_epoch=2)

    def test_severe_dev_regression_stops_before_next_epoch(self):
        self.simulate([41], regression=True)


if __name__ == '__main__':
    unittest.main()
