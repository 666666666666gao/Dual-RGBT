"""Exercise process ownership, negative admission and all-five dev scheduling."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/run_refreshed_gain_all882.py'
TREE = ast.parse(SOURCE.read_text())


def load_function(name, namespace):
    node = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace[name]


class QueueTests(unittest.TestCase):
    def test_terminal_does_not_read_native_status_before_wrapper_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = dict(failed=[])
            updates = []
            terminal = load_function('terminal', dict(
                ROOT=root, STATE=state, json=json, publish=lambda *a, **kw: updates.append(kw)))
            process = SimpleNamespace(poll=lambda: None)
            self.assertFalse(terminal(process, dict(task='missing_status')))
            path = root / 'runs/pipeline_tasks/task/status.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(state='FAILED', exit_code=1)))
            with self.assertRaises(RuntimeError):
                terminal(SimpleNamespace(poll=lambda: 1), dict(task='task'))
            self.assertEqual(len(state['failed']), 1)
            self.assertEqual(updates[-1]['status'], 'FAILED')

    def test_all_five_retained_epochs_receive_smoke_full_dev_and_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state, calls, scored, snapshots = dict(completed=[]), [], [], []
            result = dict(status='TRAIN_COMPLETE_PENDING_DEV',
                          retained=[dict(epoch=epoch) for epoch in range(4, 9)])

            def launch(run, plan, script, gpu, required, frames):
                calls.append((run, gpu, frames))
                return object(), dict(run=run, gpu=gpu)

            def score(command, **kwargs):
                self.assertNotIn('--wait', command)
                output = command[command.index('--output') + 1]
                path = root / output
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(status='SCORED_COMPLETE', PR=76, NPR=71, SR=56)))
                scored.append(command[command.index('--run') + 1])

            evaluate = load_function('evaluate', dict(
                ROOT=root, CPU='cpu', STATE=state, json=json, sha=lambda path: 'hash',
                terminal=lambda *args: True, free_memory=lambda gpu: 10000,
                launch=launch,
                evaluation_plan=lambda checkpoint, training, stage: (
                    'e%d_%s' % (checkpoint['epoch'], stage), 'bound_plan'),
                subprocess=SimpleNamespace(run=score),
                publish=lambda phase, **kw: snapshots.append(kw),
                time=SimpleNamespace(sleep=lambda seconds: self.assertEqual(seconds, 180))))
            evaluate(result, dict(task='train_formal'))
            self.assertEqual(len(calls), 10)
            self.assertEqual({run for run, _, frames in calls if frames == 11},
                             {'e%d_smoke' % epoch for epoch in range(4, 9)})
            self.assertEqual({run for run, _, frames in calls if frames == 0}, set(scored))
            self.assertEqual(len(scored), 5)
            self.assertTrue(all(gpu in ('2', '3') for _, gpu, _ in calls))
            self.assertLessEqual(max(len(row['active']) for row in snapshots), 3)
            self.assertEqual(snapshots[-1]['status'], 'COMPLETE')

    def test_low_recovery_smoke_records_rejection_without_formal_or_dev(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'reports').mkdir()
            (root / 'review.json').write_text(json.dumps(dict(
                status='SOURCE_PASS', blocking_issues=0, source_sha256={}, protocol_sha256='hash')))
            (root / 'reports/POLICY_REFRESH_E6_QUEUE_20261030.json').write_text(json.dumps(
                dict(failed=[], status='COMPLETE')))
            (root / 'reports/PREFIX_SMOKE_PLAN_20261030.json').write_text(json.dumps(dict(
                minimum_recovery_states=dict(fit=100, holdout=15),
                minimum_recovery_sequences=dict(fit=10, holdout=3))))
            calls, states = [], []

            def train(mode, budget):
                calls.append(mode)
                return dict(baseline=dict(
                    fit=dict(sampled_recoverable_states=100, recovery_sequences=10),
                    holdout=dict(sampled_recoverable_states=14, recovery_sequences=3))), {}

            main = load_function('main', dict(
                ROOT=root, STATUS=root / 'new_status.json', REVIEW='review.json',
                PROTOCOL='protocol.md', PREFIX='PREFIX', json=json,
                sha=lambda path: 'hash', check_files=lambda *args: None, train=train,
                publish=lambda phase, **kw: states.append((phase, kw))))
            main()
            self.assertEqual(calls, ['smoke'])
            self.assertEqual(states[-1][0], 'ADMISSION_REJECTED')
            self.assertEqual(states[-1][1]['reasons'], [dict(
                partition='holdout', metric='sampled_recoverable_states', observed=14, minimum=15)])


if __name__ == '__main__':
    unittest.main()
