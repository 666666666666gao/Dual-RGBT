"""Source and synthetic handoff checks; no SSH, GPU, or real signals."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / 'tools/run_size_tail_shared_eval.py'
LAUNCHER = ROOT / 'tools/rehome_size_tail_eval.py'
IMMUTABLE = {
    'tools/run_size_tail_epochs.py':
        '2a12ad01ab8b5d3c572c4bf85dbf718bd97b3396778ae99f912f8712c539e6a6',
    'tools/run_size_tail_epochs_keep_five.py':
        'ec003791bdeb57aedbaff115a01427272fc71eb02bf19af85f7d9b348e95e4e3',
    'scripts/train_size_tail_epoch.py':
        '1e5ecd5d9c69358e3ad163ea54bff42b9f494291408e797e84d0cf29bfcde597',
    'scripts/eval_size_tail.py':
        'ccf05b16cd77a816454cd0a65c17c63ee68a9f46984b5a018ab0702b70e32613',
    'tools/score_size_tail.py':
        'dccd1f862308fb3d8dbcb0109554380e9b86824a49dab6a114a2b80853744c6e',
    'reports/SIZE_TAIL_EPOCH_FORMAL_PLAN_20261030.json':
        '7724a6706715b1762e2a5c2f3c8828043b8e72467e0d39e4e5b03abc75f99aff',
    'reports/SIZE_TAIL_B640_FORMAL_PLAN_20261030.json':
        '6e330bdc44571a24c3afe68282f4d37a9e02deb01b0603703df8d8368129371b',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_functions(source, **bindings):
    """Exclude imports/main guards so Linux-only code is testable on Windows."""
    tree = ast.parse(source.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(
        argparse=argparse, json=json, os=os, Path=Path, signal=signal,
        subprocess=subprocess, sys=sys, time=SimpleNamespace(time=lambda: 0),
        sha=sha, check_files=lambda root, mapping: None,
        fcntl=SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=4, flock=lambda *args: None))
    namespace.update(bindings)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'),
         namespace)
    return namespace


class ImmutableSourceTests(unittest.TestCase):
    def test_live_plans_and_bound_sources_are_byte_unchanged(self):
        for path, digest in IMMUTABLE.items():
            with self.subTest(path=path):
                self.assertEqual(sha(ROOT / path), digest)

    def test_original_plans_bind_original_controller_and_training_sources(self):
        for plan_name, controller in (
                ('SIZE_TAIL_EPOCH_FORMAL_PLAN_20261030.json',
                 'tools/run_size_tail_epochs.py'),
                ('SIZE_TAIL_B640_FORMAL_PLAN_20261030.json',
                 'tools/run_size_tail_epochs_keep_five.py')):
            with self.subTest(plan=plan_name):
                plan = json.loads((ROOT / 'reports' / plan_name).read_text())
                self.assertEqual(plan['source_sha256'][controller], IMMUTABLE[controller])
                for path in ('scripts/train_size_tail_epoch.py',
                             'scripts/eval_size_tail.py', 'tools/score_size_tail.py'):
                    self.assertEqual(plan['source_sha256'][path], IMMUTABLE[path])


class SharedEvaluationTests(unittest.TestCase):
    def simulate(self, phase='DEV', gpu='1', scores=None, epoch=2,
                 fail_task=None, bad_score=None, bad_native=None,
                 delayed_native=False, delayed_adopted=False):
        scores = scores or [56, 56.02, 56.01, 56.02, 56]
        decisions = load_functions(ROOT / 'tools/run_size_tail_epochs_keep_five.py')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'runs/pilot'
            directory.mkdir(parents=True)
            (root / 'runs/pipeline_tasks').mkdir()
            proc = root / 'proc'
            (proc / '111').mkdir(parents=True)
            (proc / '222').mkdir()
            (proc / 'meminfo').write_text('MemAvailable: 33554432 kB\n')
            (root / 'parent.pth').write_bytes(b'PROTECTED PARENT')
            gate = dict(status='SCORED_COMPLETE', sequences=97, frames=54646)
            (root / 'gate.json').write_text(json.dumps(gate))
            (root / 'template.json').write_text(json.dumps(
                dict(source_sha256={}, original_template_field='preserved')))
            plan = dict(
                formal_launch_allowed=True, epochs=10, source_sha256={},
                parent_control_score='gate.json',
                parent_control_score_sha256=sha(root / 'gate.json'),
                diagnostic_score='gate.json', diagnostic_score_sha256=sha(root / 'gate.json'),
                evaluation_template='template.json', max_existing_gpu_mib=500,
                parent_metrics=dict(SR=56), protected_e1_metrics={},
                max_sr_drop_pp=5, batch_size=576 if gpu == '1' else 640)
            plan_path = root / 'original_plan.json'
            plan_path.write_text(json.dumps(plan))
            original_bytes = plan_path.read_bytes()
            plan_digest = sha(plan_path)
            launches, scored, sleeps, signals = [], [], [], []
            checked_sources = []
            pending_native = []

            def publish_native():
                for path, receipt in pending_native:
                    path.write_text(json.dumps(receipt))
                pending_native.clear()

            def training_receipt(current):
                run = 'pilot_train_e%03d' % current
                folder = root / 'runs' / run
                folder.mkdir(exist_ok=True)
                checkpoint = folder / ('epoch_%03d.pth' % current)
                checkpoint.write_bytes(('TRAIN EPOCH %d' % current).encode())
                receipt = dict(
                    epoch=current, strict_reload_bit_equal=True,
                    checkpoint_sha256=sha(checkpoint), updates=current * 16,
                    original_plan_sha256=plan_digest)
                (folder / 'COMPLETE.json').write_text(json.dumps(receipt))
                return checkpoint, receipt

            rows = []
            for current in range(1, epoch):
                checkpoint, training = training_receipt(current)
                score_path = directory / ('score_epoch_%03d.json' % current)
                score_path.write_text(json.dumps(dict(
                    status='SCORED_COMPLETE', sequences=97, frames=54646,
                    PR=70, NPR=60 + current / 100, SR=scores[current - 1], fps=10)))
                rows.append(dict(
                    epoch=current, checkpoint=checkpoint.relative_to(root).as_posix(),
                    checkpoint_sha256=sha(checkpoint), checkpoint_retained=True,
                    score_path=score_path.relative_to(root).as_posix(),
                    score_sha256=sha(score_path),
                    metrics=dict(PR=70, NPR=60 + current / 100,
                                 SR=scores[current - 1], fps=10), training=training))
            original_rows = copy.deepcopy(rows)
            current_task = ('train_pilot_train_e%03d' if phase == 'TRAIN'
                            else 'eval_pilot_dev_e%03d') % epoch
            state = dict(
                status='RUNNING', controller_pid=111, plan_sha256=plan_digest,
                physical_gpu=gpu, epoch=epoch, phase=phase, active_task=current_task,
                active_pid=222, rows=rows)
            (directory / 'status.json').write_text(json.dumps(state))
            namespace = load_functions(
                CONTROLLER, ROOT=root,
                Path=lambda value: proc / str(value)[6:] if str(value).startswith('/proc/')
                    else proc if value == '/proc' else Path(value),
                best_recent=decisions['best_recent'],
                stopping_epoch=decisions['stopping_epoch'],
                check_files=lambda base, mapping: checked_sources.append((base, mapping)),
                os=SimpleNamespace(environ={}, getpid=lambda: 333,
                                   kill=lambda *args: signals.append(args)))
            (root / 'review.json').write_text(json.dumps(dict(
                status='SOURCE_PASS', blocking_issues=0, source_sha256={'new.py': 'reviewed'})))
            handoff = dict(
                chain='pilot', plan='original_plan.json', eval_gpu='3',
                identity=namespace['identity'](state),
                source_review='review.json', source_review_sha256=sha(root / 'review.json'))
            (root / 'handoff.json').write_text(json.dumps(handoff))
            adopted = root / 'runs/pipeline_tasks' / current_task
            adopted.mkdir()
            native = dict(
                pid=222, task=current_task, state='RUNNING', physical_gpu=gpu,
                command=[str(root / 'conda-envs/dual_rgbt/bin/python'),
                         str(root / 'scripts' / ('train_size_tail_epoch.py'
                             if phase == 'TRAIN' else 'eval_size_tail.py'))])
            if bad_native:
                native.update(bad_native)
            (adopted / 'status.json').write_text(json.dumps(native))
            if phase == 'DEV':
                training_receipt(epoch)
                (directory / ('eval_epoch_%03d.json' % epoch)).write_text('{"existing": true}')
            evaluation_bytes = ((directory / ('eval_epoch_%03d.json' % epoch)).read_bytes()
                                if phase == 'DEV' else None)

            def sleep(seconds):
                sleeps.append(seconds)
                if seconds == .2:
                    ready = json.loads((directory / 'shared_eval_ready.json').read_text())
                    self.assertEqual(ready['identity'], handoff['identity'])
                    (proc / '111').rmdir()
                    if delayed_adopted:
                        (adopted / 'status.json').write_text('')
                elif seconds == 180:
                    publish_native()
                    native.update(state='COMPLETE', exit_code=0)
                    (adopted / 'status.json').write_text(json.dumps(native))
                    if (proc / '222').exists():
                        (proc / '222').rmdir()
                    if phase == 'TRAIN':
                        training_receipt(epoch)
                elif seconds == .1:
                    publish_native()
                else:
                    self.fail('Unexpected fixture wait %s' % seconds)

            class Process:
                def __init__(self, command, **kwargs):
                    launches.append(command)
                    self.pid = 400 + len(launches)
                    task = command[command.index('--task') + 1]
                    card = command[command.index('--gpu') + 1]
                    child = command[command.index('--') + 1:]
                    run = child[child.index('--run') + 1]
                    current = int(run[-3:])
                    native_path = root / 'runs/pipeline_tasks' / task
                    native_path.mkdir()
                    self.returncode = int(task == fail_task)
                    receipt = dict(
                        pid=self.pid, command=child, physical_gpu=card,
                        state='FAILED' if self.returncode else 'COMPLETE',
                        exit_code=self.returncode)
                    status_path = native_path / 'status.json'
                    if delayed_native:
                        # write_text opens/truncates before its buffered write is visible.
                        status_path.write_text('')
                        pending_native.append((status_path, receipt))
                    else:
                        status_path.write_text(json.dumps(receipt))
                    if '--epoch' in child and not self.returncode:
                        training_receipt(current)

                def poll(self):
                    publish_native()
                    return self.returncode

            def score(command, **kwargs):
                current = int(command[command.index('--run') + 1][-3:])
                scored.append(current)
                output = root / command[command.index('--output') + 1]
                metrics = dict(status='SCORED_COMPLETE', sequences=97, frames=54646,
                               PR=70, NPR=60 + current / 100, SR=scores[current - 1], fps=10)
                if bad_score:
                    metrics.update(bad_score)
                output.write_text(json.dumps(metrics))

            def memory_query(command, **kwargs):
                if '--query-gpu=uuid,memory.free' in command:
                    return '\n'.join('GPU-%d, 20000' % i for i in range(4))
                if '--query-compute-apps=pid,gpu_uuid' in command:
                    return '501, GPU-3\n'
                self.assertIn('--query-gpu=memory.used', command)
                return '0\n0\n0\n0\n'

            namespace.update(
                time=SimpleNamespace(time=lambda: 0, sleep=sleep),
                subprocess=SimpleNamespace(
                    Popen=Process, run=score, STDOUT=subprocess.STDOUT,
                    check_output=memory_query))
            with patch.object(sys, 'argv', ['controller', '--handoff', 'handoff.json']):
                if fail_task or bad_score or bad_native:
                    with self.assertRaises(AssertionError):
                        namespace['main']()
                else:
                    namespace['main']()
            result = json.loads((directory / 'status.json').read_text())
            self.assertEqual(plan_path.read_bytes(), original_bytes)
            self.assertEqual((root / 'parent.pth').read_bytes(), b'PROTECTED PARENT')
            self.assertEqual(signals, [])
            self.assertEqual((proc / '222').exists(), bool(bad_native))
            if bad_native:
                self.assertFalse((directory / 'shared_eval_ready.json').exists())
                self.assertEqual(result, state)
                self.assertEqual(launches, [])
                return result, launches
            self.assertIn(180, sleeps)
            self.assertEqual(result['physical_gpu'], gpu)
            self.assertEqual(result['evaluation_gpu'], '3')
            self.assertEqual(result['previous_controller_pid'], 111)
            self.assertEqual(result['plan_sha256'], plan_digest)
            self.assertEqual(result['rows'][:len(original_rows)], [
                dict(row, checkpoint_retained=result['rows'][i]['checkpoint_retained'])
                for i, row in enumerate(original_rows)])
            if phase == 'DEV':
                self.assertEqual((directory / ('eval_epoch_%03d.json' % epoch)).read_bytes(),
                                 evaluation_bytes)
            for command in launches:
                child = command[command.index('--') + 1:]
                task = command[command.index('--task') + 1]
                self.assertNotEqual(task, current_task)
                card = command[command.index('--gpu') + 1]
                current = int(child[child.index('--run') + 1][-3:])
                if '--epoch' in child:
                    self.assertGreater(current, epoch)
                    self.assertEqual(card, gpu)
                    self.assertEqual(child[child.index('--plan') + 1], 'original_plan.json')
                    self.assertEqual(child[child.index('--epoch') + 1], str(current))
                    self.assertEqual(child[child.index('--resume') + 1],
                                     'runs/pilot_train_e%03d/epoch_%03d.pth' %
                                     (current - 1, current - 1))
                    expected_sha = hashlib.sha256(
                        ('TRAIN EPOCH %d' % (current - 1)).encode()).hexdigest()
                    self.assertEqual(child[child.index('--resume-sha256') + 1], expected_sha)
                else:
                    self.assertEqual(card, '3')
                    self.assertEqual(child[child.index('--smoke-frames') + 1], '0')
                    evaluation = json.loads((root / child[child.index('--plan') + 1]).read_text())
                    self.assertEqual(evaluation['original_template_field'], 'preserved')
                    self.assertEqual(evaluation['training_task'],
                                     'train_pilot_train_e%03d' % current)
            if fail_task or bad_score:
                self.assertFalse((directory / 'SELECTION.json').exists())
                self.assertTrue((root / 'runs/pilot_train_e001/epoch_001.pth').exists())
                return result, launches
            if result['status'] == 'REJECTED_REGRESSION':
                self.assertTrue((directory / 'REJECTION.json').exists())
                self.assertFalse((directory / 'SELECTION.json').exists())
                return result, launches
            selection = json.loads((directory / 'SELECTION.json').read_text())
            self.assertEqual(selection['plan_sha256'], plan_digest)
            self.assertFalse(selection['test_eligible'])
            self.assertEqual(selection['evaluated_epochs'],
                             list(range(1, len(result['rows']) + 1)))
            retained = list((root / 'runs').glob('pilot_train_*/epoch_*.pth'))
            self.assertEqual({int(path.stem[-3:]) for path in retained},
                             set(selection['selection_window']))
            self.assertEqual(len(retained), 5)
            self.assertIn(root / selection['selected']['checkpoint'], retained)
            for row in result['rows']:
                self.assertEqual(row['checkpoint_retained'], (root / row['checkpoint']).exists())
                self.assertEqual(row['score_sha256'], sha(root / row['score_path']))
                if row['checkpoint_retained']:
                    self.assertEqual(row['checkpoint_sha256'], sha(root / row['checkpoint']))
            self.assertEqual(scored, list(range(epoch, len(result['rows']) + 1)))
            self.assertIn((root, {'new.py': 'reviewed'}), checked_sources)
            return selection, launches

    def test_adopts_dev_without_relaunch_and_stops_on_plateau(self):
        selection, _ = self.simulate()
        self.assertEqual(selection['evaluated_epochs'], [1, 2, 3, 4, 5])
        self.assertEqual(selection['selected']['epoch'], 4)

    def test_adopts_train_and_routes_only_future_dev_to_shared_gpu(self):
        selection, _ = self.simulate(phase='TRAIN', gpu='0')
        self.assertEqual(selection['selection_window'], [1, 2, 3, 4, 5])

    def test_later_stop_rolls_five_without_deleting_nonwinners(self):
        selection, _ = self.simulate(
            epoch=3, scores=[56, 56.2, 56.4, 56.6, 56.8, 57, 57.01, 57.02, 57.01])
        self.assertEqual(selection['selection_window'], [5, 6, 7, 8, 9])
        self.assertEqual(selection['selected']['epoch'], 8)

    def test_regression_stops_before_next_training_epoch(self):
        state, launches = self.simulate(scores=[56, 41])
        self.assertEqual(state['status'], 'REJECTED_REGRESSION')
        self.assertEqual(state['epoch'], 2)
        self.assertEqual(launches, [])

    def test_failed_new_training_does_not_score_or_select(self):
        state, _ = self.simulate(fail_task='train_pilot_train_e003')
        self.assertEqual([row['epoch'] for row in state['rows']], [1, 2])

    def test_partial_dev_score_is_never_selected(self):
        state, _ = self.simulate(bad_score=dict(sequences=96))
        self.assertEqual([row['epoch'] for row in state['rows']], [1])

    def test_rejects_bad_native_identity_before_ready(self):
        for changes in (dict(pid=999), dict(physical_gpu='3'),
                        dict(task='unrelated_task'),
                        dict(state='FAILED', exit_code=1),
                        dict(command=['python', 'unrelated.py'])):
            with self.subTest(changes=changes):
                self.simulate(bad_native=changes)

    def test_native_receipt_created_before_write_is_not_parsed_early(self):
        self.simulate(delayed_native=True)

    def test_adopted_wrapper_terminal_write_finishes_before_read(self):
        for phase in ('TRAIN', 'DEV'):
            with self.subTest(phase=phase):
                self.simulate(phase=phase, delayed_adopted=True)


class NativeCapacityTests(unittest.TestCase):
    def test_capacity_reserves_both_simultaneous_dispatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            meminfo = Path(temporary) / 'meminfo'
            namespace = load_functions(
                CONTROLLER, Path=lambda value: meminfo if value == '/proc/meminfo'
                else Path(value))
            for free, count, host_gib, expected in (
                    (6144, 2, 16, True), (6143, 2, 16, False),
                    (6144, 3, 16, False), (6144, 2, 15, False)):
                with self.subTest(free=free, count=count, host_gib=host_gib):
                    meminfo.write_text('MemAvailable: %d kB\n' % (host_gib * 1024 * 1024))

                    def query(command, **kwargs):
                        if '--query-gpu=uuid,memory.free' in command:
                            return '\n'.join('GPU-%d, %d' % (i, free) for i in range(4))
                        self.assertIn('--query-compute-apps=pid,gpu_uuid', command)
                        return '\n'.join(
                            ['1, GPU-0'] + ['%d, GPU-3' % (i + 20) for i in range(count)])

                    namespace['subprocess'] = SimpleNamespace(check_output=query)
                    self.assertEqual(namespace['evaluation_capacity']('3'), expected)

    def test_native_success_failure_pid_and_liveness(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            (proc / '222').mkdir()
            namespace = load_functions(
                CONTROLLER, Path=lambda value: proc if value == '/proc' else Path(value))
            terminal = namespace['task_terminal']
            self.assertFalse(terminal(dict(pid=222, state='RUNNING'), 222))
            self.assertTrue(terminal(dict(pid=222, state='COMPLETE', exit_code=0), 222))
            for native, pid in ((dict(pid=222, state='FAILED', exit_code=1), 222),
                                (dict(pid=222, state='COMPLETE', exit_code=1), 222),
                                (dict(pid=222, state='RUNNING'), 999),
                                (dict(pid=333, state='RUNNING'), 333)):
                with self.subTest(native=native, pid=pid):
                    with self.assertRaises(AssertionError):
                        terminal(native, pid)


class LauncherHandshakeTests(unittest.TestCase):
    def simulate(self, chain_index=0, invalid=None):
        tree = ast.parse(LAUNCHER.read_text())
        chains = ast.literal_eval(next(node.value for node in tree.body
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'CHAINS'
                for target in node.targets)))
        chain = tuple(chains)[chain_index]
        plan, old_controller = chains[chain]
        old_pid = (1736653, 1935434)[chain_index]
        gpu = ('1', '0')[chain_index]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'runs' / chain
            directory.mkdir(parents=True)
            (root / 'reports').mkdir()
            (root / plan).write_text('{"original": "immutable"}')
            original_plan = (root / plan).read_bytes()
            (root / 'reports/SIZE_TAIL_SHARED_EVAL_SOURCE_REVIEW_20261030.json').write_text(
                json.dumps(dict(status='SOURCE_PASS', blocking_issues=0, source_sha256={})))
            proc = root / 'proc'
            (proc / str(old_pid)).mkdir(parents=True)
            (proc / '222').mkdir()
            command = [str(root / 'conda-envs/dual_rgbt/bin/python'),
                       str(root / old_controller), '--plan', plan, '--name', chain,
                       '--gpu', gpu]
            if invalid == 'old_command':
                command[1] = 'unrelated.py'
            (proc / str(old_pid) / 'cmdline').write_bytes(
                ('\0'.join(command) + '\0').encode())
            state = dict(status='RUNNING', phase='DEV', controller_pid=old_pid,
                         plan_sha256=sha(root / plan), physical_gpu=gpu,
                         epoch=2, active_task='native_existing_dev', active_pid=222, rows=[])
            state_path = directory / 'status.json'
            state_path.write_text(json.dumps(state))
            native_directory = root / 'runs/pipeline_tasks/native_existing_dev'
            native_directory.mkdir(parents=True)
            native = dict(pid=222, task='native_existing_dev', state='RUNNING',
                          command=['original', 'eval'], physical_gpu=gpu)
            if invalid == 'failed_child':
                native.update(state='FAILED', exit_code=1)
            if invalid == 'dead_child':
                (proc / '222').rmdir()
            (native_directory / 'status.json').write_text(json.dumps(native))
            if invalid == 'plan_hash':
                (root / plan).write_text('{"changed": true}')
            namespace = load_functions(
                CONTROLLER, Path=lambda value: proc if value == '/proc' else Path(value))
            task_checks, signals, launches = [], [], []

            def task_terminal(receipt, pid):
                task_checks.append((pid, len(signals)))
                return namespace['task_terminal'](receipt, pid)

            def kill(pid, sig):
                self.assertTrue((directory / 'shared_eval_ready.json').exists())
                self.assertEqual(pid, old_pid)
                self.assertEqual(sig, signal.SIGTERM)
                self.assertTrue((proc / '222').exists())
                signals.append((pid, sig))

            class Process:
                def __init__(self, new_command, **kwargs):
                    launches.append(new_command)
                    self.pid = 333
                    handoff_path = root / new_command[new_command.index('--handoff') + 1]
                    handoff = json.loads(handoff_path.read_text())
                    ready = dict(
                        pid=self.pid, handoff_sha256=sha(handoff_path),
                        identity=handoff['identity'], status='READY_TO_ADOPT')
                    if invalid == 'ready_status':
                        ready['status'] = 'NOT_READY'
                    if invalid == 'ready_pid':
                        ready['pid'] = 999
                    if invalid == 'ready_hash':
                        ready['handoff_sha256'] = 'wrong'
                    if invalid == 'ready_identity':
                        ready['identity'] = dict(ready['identity'], active_pid=999)
                    if invalid == 'changed_state':
                        state_path.write_text(json.dumps(dict(state, phase='SCORE')))
                    (directory / 'shared_eval_ready.json').write_text(json.dumps(ready))

                def poll(self):
                    return None

            launcher = load_functions(
                LAUNCHER, ROOT=root, CHAINS=chains,
                Path=lambda value: proc if value == '/proc' else Path(value),
                identity=namespace['identity'], task_terminal=task_terminal,
                os=SimpleNamespace(environ={}, kill=kill),
                subprocess=SimpleNamespace(Popen=Process, STDOUT=subprocess.STDOUT))
            with patch.object(sys, 'argv', [
                    'rehome', '--chain', chain, '--eval-gpu', '3']), patch('builtins.print'):
                if invalid:
                    with self.assertRaises(AssertionError):
                        launcher['main']()
                    self.assertEqual(signals, [])
                    self.assertFalse((directory / 'shared_eval_takeover.json').exists())
                    if invalid in ('old_command', 'plan_hash', 'dead_child', 'failed_child'):
                        self.assertEqual(launches, [])
                    return
                launcher['main']()
            self.assertEqual(signals, [(old_pid, signal.SIGTERM)])
            self.assertEqual(task_checks, [(222, 0), (222, 1)])
            self.assertEqual((root / plan).read_bytes(), original_plan)
            self.assertEqual(json.loads(state_path.read_text()), state)
            self.assertTrue((proc / '222').exists())
            receipt = json.loads((directory / 'shared_eval_takeover.json').read_text())
            self.assertEqual(receipt['preserved_wrapper_pid'], 222)
            self.assertEqual(receipt['original_plan_sha256'], sha(root / plan))
            self.assertEqual(receipt['train_gpu'], gpu)
            self.assertEqual(receipt['future_eval_gpu'], '3')
            self.assertTrue(receipt['terminated_only_cpu_controller'])
            self.assertEqual(len(launches), 1)

    def test_ready_handshake_signals_only_each_verified_old_controller(self):
        for chain_index in (0, 1):
            with self.subTest(chain_index=chain_index):
                self.simulate(chain_index)

    def test_invalid_ready_or_changed_identity_never_signals(self):
        for invalid in ('ready_status', 'ready_pid', 'ready_hash',
                        'ready_identity', 'changed_state'):
            with self.subTest(invalid=invalid):
                self.simulate(invalid=invalid)

    def test_invalid_old_command_plan_or_child_never_launches_or_signals(self):
        for invalid in ('old_command', 'plan_hash', 'failed_child', 'dead_child'):
            with self.subTest(invalid=invalid):
                self.simulate(invalid=invalid)

    def test_controller_has_no_signal_or_child_termination_api(self):
        forbidden = {'kill', 'killpg', 'terminate', 'send_signal'}
        tree = ast.parse(CONTROLLER.read_text())
        calls = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertFalse(forbidden.intersection(calls))


if __name__ == '__main__':
    unittest.main()
