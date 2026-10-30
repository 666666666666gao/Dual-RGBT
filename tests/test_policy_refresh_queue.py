"""Audit rejection must not abandon other refresh workers."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'tools/queue_policy_refresh.py'
TREE = ast.parse(SOURCE.read_text())
MAIN = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'main')


class QueueTests(unittest.TestCase):
    def test_failed_audit_keeps_other_workers_and_reuses_finished_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'reports').mkdir()
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            jobs = []
            for name in ('a', 'b', 'c'):
                path = root / 'reports' / (name + '.json')
                path.write_text('{}')
                jobs.append(dict(name=name, plan=path.relative_to(root).as_posix(),
                                 plan_sha256=sha(path)))
            (root / 'reports/POLICY_REFRESH_E6_ALL882_PLAN_20261030.json').write_text(
                json.dumps(dict(jobs=jobs)))
            commands, audits, snapshots = [], [], []

            class Process:
                def __init__(self, command, **kwargs):
                    commands.append(command)
                    self.pid = 100 + len(commands)
                    self.run = command[command.index('--run') + 1]
                    self.polls = 0
                    task = command[command.index('--task') + 1]
                    native = root / 'runs/pipeline_tasks' / task
                    native.mkdir(parents=True)
                    (native / 'status.json').write_text(json.dumps(
                        dict(state='COMPLETE', exit_code=0)))
                    directory = root / 'runs' / self.run
                    directory.mkdir()
                    if '--smoke' in command:
                        (directory / 'SMOKE.json').write_text(json.dumps(dict(
                            observer_policy_trajectory_equal=True, frozen_state_tensors=265)))

                def poll(self):
                    self.polls += 1
                    if self.run == 'b_smoke' and self.polls == 1:
                        return None
                    return 0

            def audit(command, **kwargs):
                self.assertEqual(Path(command[1]).name, 'audit_policy_refresh.py')
                run = command[command.index('--run') + 1]
                audits.append(run)
                if run == 'a_smoke':
                    return SimpleNamespace(returncode=1)
                output = root / command[command.index('--output') + 1]
                output.write_text('{"status": "VERIFIED_CACHE_DIAGNOSTIC"}')
                return SimpleNamespace(returncode=0)

            status_path = root / 'reports/POLICY_REFRESH_E6_QUEUE_20261030.json'

            def wait(seconds):
                self.assertEqual(seconds, 180)
                snapshots.append(json.loads(status_path.read_text()))

            namespace = dict(
                ROOT=root, json=json, sha=sha,
                os=SimpleNamespace(environ={}, getpid=lambda: 42),
                subprocess=SimpleNamespace(
                    Popen=Process, run=audit, STDOUT=-2,
                    check_output=lambda *args, **kwargs: '0\n0\n'),
                time=SimpleNamespace(time=lambda: 0, sleep=wait))
            exec(compile(ast.Module(body=[MAIN], type_ignores=[]), str(SOURCE), 'exec'),
                 namespace)
            namespace['main']()
            result = json.loads(status_path.read_text())
            self.assertEqual(result['status'], 'COMPLETE_WITH_FAILURES')
            self.assertEqual(len(result['failed']), 1)
            failure = result['failed'][0]
            self.assertEqual(failure['phase'], 'AUDIT')
            self.assertEqual(failure['audit_exit_code'], 1)
            self.assertEqual(failure['entry']['run'], 'a_smoke')
            self.assertFalse(result['active'])
            self.assertFalse(result['queue'])
            self.assertNotIn('a_full', audits)
            self.assertEqual(set(audits),
                             {'a_smoke', 'b_smoke', 'b_full', 'c_smoke', 'c_full'})
            self.assertTrue(any(
                row['failed'] and row['active']['1']['run'] == 'b_smoke'
                and row['active']['0']['run'] == 'c_smoke'
                for row in snapshots if '0' in row['active'] and '1' in row['active']))
            self.assertEqual(len(result['completed']), 4)


if __name__ == '__main__':
    unittest.main()
