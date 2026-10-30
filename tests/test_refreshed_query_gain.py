"""Local source/fixture checks only; no torch, GPU, or accuracy attestation."""
import ast
from contextlib import nullcontext
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/train_refreshed_query_gain.py'
TREE = ast.parse(SOURCE.read_text())
PARENT_TREE = ast.parse((ROOT / 'scripts/train_cached_query_gain.py').read_text())


def function(tree, name):
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'),
         namespace)
    return namespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))


class Array(np.ndarray):
    """Only the tensor operations exercised by the source snippets below."""
    def abs(self):
        return np.abs(self)

    def sign(self):
        return np.sign(self)

    def new_zeros(self, shape):
        return np.zeros(shape).view(Array)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class CacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / 'runs/cache'
        self.run.mkdir(parents=True)
        self.native_path = self.root / 'runs/pipeline_tasks/collect/status.json'
        self.native_path.parent.mkdir(parents=True)
        self.audit_path = self.root / 'audit.json'
        self.plan = dict(parent_checkpoint_sha256='parent', cache_checkpoint_sha256='e6',
                         seed=2027, expected_sequences=2,
                         fit_sequences=['fit'], holdout_sequences=['held'],
                         caches=[dict(run='cache', task='collect', audit='audit.json')])
        self.config = dict(
            split='train', smoke=False, parent_checkpoint_sha256='parent',
            checkpoint_sha256='e6', native_command=['collector'],
            selected_sequences=['fit', 'held'],
            plan=dict(mode='frozen_policy_train_refresh', overlay_sha256='e6',
                      checkpoint_sha256='e6', seed=2027, cost_weight=.15,
                      label_stride=10, fit_sequences=['fit'],
                      holdout_sequences=['held'], expected_frames=62))
        self.receipt = dict(
            status='CACHE_COMPLETE', checkpoint_sha256='e6', sequences=2, frames=62,
            frozen_state_tensors=265, optimizer_updates=0, backward_calls=0,
            teacher_preserves_tracker_state=True)
        self.native = dict(state='COMPLETE', exit_code=0, command=['collector'])
        self.audit = dict(status='VERIFIED_CACHE_DIAGNOSTIC', smoke=False,
                          complete_planned_coverage=True, run='cache')
        for name, partition in (('fit', 'fit'), ('held', 'holdout')):
            features = np.arange(5 * 955, dtype=np.float32).reshape(5, 955)
            np.save(self.run / (name + '_features.npy'), features)
            rows = [dict(sequence=name, partition=partition, frame=10, kind=kind,
                         purpose='sampled' if kind < 4 else 'selected_extra',
                         primary_iou=.05, gain=.6 if kind == 0 else -.02,
                         predicted_gain=.25) for kind in range(5)]
            with (self.run / (name + '_labels.csv')).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        self.namespace = execute(
            [function(TREE, name) for name in ('check_cache_policy', 'load_caches')],
            dict(ROOT=self.root, sha=sha, json=json, csv=csv, np=np))
        self.seal()

    def seal(self):
        # Rebind fixtures so rejection tests exercise semantics, not stale hashes.
        write_json(self.run / 'config.json', self.config)
        self.receipt['artifacts_sha256'] = {
            path.name: sha(path) for path in self.run.iterdir()
            if path.name != 'COMPLETE.json'}
        write_json(self.run / 'COMPLETE.json', self.receipt)
        write_json(self.native_path, self.native)
        self.audit.update(receipt_sha256=sha(self.run / 'COMPLETE.json'),
                          native_sha256=sha(self.native_path))
        write_json(self.audit_path, self.audit)
        self.plan['caches'][0]['audit_sha256'] = sha(self.audit_path)

    def load(self):
        return self.namespace['load_caches'](self.plan)

    def test_valid_e6_cache_excludes_selected_extra(self):
        data, hashes = self.load()
        self.assertNotEqual(self.plan['parent_checkpoint_sha256'],
                            self.plan['cache_checkpoint_sha256'])
        self.assertEqual(data['fit']['features'].shape, (1, 4, 955))
        self.assertEqual(data['holdout']['sequences'], ['held'])
        np.testing.assert_allclose(data['fit']['gains'], [[.6, -.02, -.02, -.02]])
        self.assertIn(str(Path('runs/cache/config.json')), hashes)

    def test_rejects_parent_and_e6_hash_mismatches(self):
        cases = [
            (self.config, 'parent_checkpoint_sha256', 'e6'),
            (self.config, 'checkpoint_sha256', 'parent'),
            (self.receipt, 'checkpoint_sha256', 'parent'),
            (self.config['plan'], 'overlay_sha256', 'parent'),
            (self.config['plan'], 'checkpoint_sha256', 'parent'),
        ]
        for mapping, key, wrong in cases:
            with self.subTest(key=key, mapping=list(mapping)):
                before = mapping[key]
                mapping[key] = wrong
                self.seal()
                with self.assertRaises(AssertionError):
                    self.load()
                mapping[key] = before

    def test_rejects_truncated_planned_sequences_and_frames(self):
        cases = [(self.config, 'selected_sequences', ['fit']),
                 (self.receipt, 'sequences', 1), (self.receipt, 'frames', 31)]
        for mapping, key, wrong in cases:
            with self.subTest(key=key):
                before = mapping[key]
                mapping[key] = wrong
                self.seal()
                with self.assertRaises(AssertionError):
                    self.load()
                mapping[key] = before

    def test_rejects_false_audit_or_smoke(self):
        cases = [(self.audit, 'status', 'FAILED'),
                 (self.audit, 'complete_planned_coverage', False),
                 (self.audit, 'smoke', True), (self.config, 'smoke', True)]
        for mapping, key, wrong in cases:
            with self.subTest(key=key):
                before = mapping[key]
                mapping[key] = wrong
                self.seal()
                with self.assertRaises(AssertionError):
                    self.load()
                mapping[key] = before

    def test_rejects_wrong_seed_split_cost_or_stride(self):
        cases = [(self.config['plan'], 'seed', 2028),
                 (self.config, 'split', 'dev'),
                 (self.config['plan'], 'cost_weight', .1),
                 (self.config['plan'], 'label_stride', 5)]
        for mapping, key, wrong in cases:
            with self.subTest(key=key):
                before = mapping[key]
                mapping[key] = wrong
                self.seal()
                with self.assertRaises(AssertionError):
                    self.load()
                mapping[key] = before

    def test_rejects_wrong_fixed_partition(self):
        self.plan.update(fit_sequences=['held'], holdout_sequences=['fit'])
        with self.assertRaises(AssertionError):
            self.load()

    def test_rejects_wrong_label_partition(self):
        path = self.run / 'fit_labels.csv'
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        rows[0]['partition'] = 'holdout'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.seal()
        with self.assertRaises(AssertionError):
            self.load()

    def test_rejects_duplicate_cache_and_artifact_tamper(self):
        self.plan['caches'].append(copy.deepcopy(self.plan['caches'][0]))
        with self.assertRaises(AssertionError):
            self.load()
        self.plan['caches'].pop()
        (self.run / 'fit_features.npy').write_bytes(b'tampered')
        with self.assertRaises(AssertionError):
            self.load()


class SourceContractTests(unittest.TestCase):
    def test_objective_diagnostics_initialization_and_training_loop_unchanged(self):
        for name in ('objective', 'diagnostics'):
            self.assertEqual(ast.dump(function(TREE, name)),
                             ast.dump(function(PARENT_TREE, name)))
        main, parent = function(TREE, 'main'), function(PARENT_TREE, 'main')

        def initialization(node):
            start = next(i for i, part in enumerate(node.body)
                         if ast.dump(part) == ast.dump(
                             ast.parse("torch.manual_seed(plan['seed'])").body[0]))
            end = next(i for i, part in enumerate(node.body)
                       if isinstance(part, ast.FunctionDef) and part.name == 'measure')
            return [ast.dump(part) for part in node.body[start:end + 1]]

        self.assertEqual(initialization(main), initialization(parent))
        loops = [next(part for part in node.body if isinstance(part, ast.For)
                      and isinstance(part.target, ast.Name) and part.target.id == 'epoch')
                 for node in (main, parent)]
        self.assertEqual(ast.dump(loops[0]), ast.dump(loops[1]))

    def test_parent_loss_stays_in_gain_units_and_ignores_cost(self):
        def smooth_l1(prediction, gain):
            error = np.abs(prediction - gain)
            return np.asarray(
                np.where(error < 1, .5 * error ** 2, error - .5).mean()).view(Array)

        namespace = execute([function(TREE, 'objective')],
                            dict(F=SimpleNamespace(smooth_l1_loss=smooth_l1,
                                                   relu=lambda x: np.maximum(0, x))))
        loss = namespace['objective']
        gain = np.array([[.6, -.8, .2, 0.]]).view(Array)
        self.assertEqual(loss(gain, gain, np.ones(4), 'parent_loss'), 0)
        prediction = np.array([[0.]]).view(Array)
        gain = np.array([[.02]]).view(Array)
        for cost in (np.zeros(1), np.ones(1)):
            self.assertAlmostEqual(float(loss(prediction, gain, cost, 'parent_loss')),
                                   .5 * .02 ** 2 + .25 * .05)
        zero = np.zeros((1, 4)).view(Array)
        self.assertEqual(loss(zero, zero, np.ones(4), 'parent_loss'), 0)

    def test_recorded_prediction_check_uses_e6_not_parent_head(self):
        class Head:
            def __init__(self):
                self.value = .1

            def load_state_dict(self, state, strict):
                self.value = state['value']
                assert strict

            def __call__(self, tensor):
                return np.full(tensor.shape[:-1] + (1,), self.value).view(Array)

        main = function(TREE, 'main')
        start = next(i for i, node in enumerate(main.body)
                     if ast.dump(node) == ast.dump(
                         ast.parse('baseline, max_errors = {}, {}').body[0]))
        end = next(i for i, node in enumerate(main.body)
                   if isinstance(node, ast.Delete))
        head = Head()
        tensors = {part: np.zeros((1, 4, 955)) for part in ('fit', 'holdout')}
        namespace = dict(
            copy=copy, np=np, torch=SimpleNamespace(no_grad=nullcontext), head=head,
            overlay=dict(head=dict(value=.8)), tensors=tensors,
            data={part: dict(recorded=np.full((1, 4), .8)) for part in tensors},
            measure=lambda part: (dict(parent_value=head.value), None))
        execute(main.body[start:end + 1], namespace)
        self.assertEqual(head.value, .1)
        self.assertEqual(namespace['max_errors'], dict(fit=0., holdout=0.))
        self.assertEqual(namespace['baseline']['fit']['parent_value'], .1)
        namespace['overlay'] = dict(head=dict(value=.1))
        with self.assertRaises(AssertionError):
            execute(main.body[start:end + 1], namespace)

    def test_manifest_requires_train_and_excludes_dev_test(self):
        main = function(TREE, 'main')
        start = next(i for i, node in enumerate(main.body)
                     if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id == 'manifest_path')
        end = next(i for i, node in enumerate(main.body)
                   if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == 'overlay')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'manifest.json'
            plan = dict(training_manifest='manifest.json', expected_sequences=2,
                        fit_sequences=['fit'], holdout_sequences=['held'])
            for dev, test, train, accepted in (
                ([], [], ['fit', 'held'], True),
                (['held'], [], ['fit', 'held'], False),
                ([], ['held'], ['fit', 'held'], False),
                ([], [], ['fit'], False),
            ):
                with self.subTest(dev=dev, test=test, train=train):
                    write_json(path, dict(train=[dict(name=n) for n in train],
                                          dev=[dict(name=n) for n in dev],
                                          test_sequences=test))
                    plan['training_manifest_sha256'] = sha(path)
                    namespace = dict(ROOT=root, plan=plan, sha=sha, json=json)
                    if accepted:
                        execute(main.body[start:end], namespace)
                    else:
                        with self.assertRaises(AssertionError):
                            execute(main.body[start:end], namespace)

    def test_formal_coverage_gate_100_15_states_over_10_3_sequences(self):
        gate = next(node for node in function(TREE, 'main').body
                    if isinstance(node, ast.If)
                    and 'minimum_recovery_states' in ast.dump(node))
        plan = dict(mode='formal', minimum_recovery_states=dict(fit=100, holdout=15),
                    minimum_recovery_sequences=dict(fit=10, holdout=3))
        baseline = dict(fit=dict(sampled_recoverable_states=100, recovery_sequences=10),
                        holdout=dict(sampled_recoverable_states=15, recovery_sequences=3))
        namespace = dict(plan=plan, baseline=baseline, data=baseline)
        execute([gate], namespace)
        for part in baseline:
            for key in baseline[part]:
                with self.subTest(part=part, key=key):
                    baseline[part][key] -= 1
                    with self.assertRaises(AssertionError):
                        execute([gate], namespace)
                    baseline[part][key] += 1


class PlannerTests(unittest.TestCase):
    def test_planner_keeps_matched100_partitions_and_parent_loss_budget(self):
        tree = ast.parse((ROOT / 'tools/prepare_refreshed_gain.py').read_text())
        main = function(tree, 'main')
        start = next(i for i, node in enumerate(main.body)
                     if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id == 'names')
        end = next(i for i, node in enumerate(main.body) if isinstance(node, ast.For))
        namespace = dict(random=random, manifest=dict(
            train=[dict(name='sequence_%03d' % i) for i in range(882)]))
        execute(main.body[start:end], namespace)
        expected = sorted(row['name'] for row in namespace['manifest']['train'])
        random.Random(2027).shuffle(expected)
        self.assertEqual(namespace['selected'], expected[:100])
        self.assertEqual(namespace['held'],
                         set(expected[16:20] + expected[20:100][4::5]))
        self.assertEqual(len(namespace['fit']), 80)
        self.assertFalse(set(namespace['fit']) & namespace['held'])
        inherit = next(node for node in main.body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id == 'plan')
        control = dict(
            parent_evaluation_plan='parent_plan.json',
            parent_checkpoint_sha256='parent', variant='parent_loss', seed=2027,
            batch_states=2866, lr=.00112, min_epochs=8, max_epochs=30, patience=4,
            min_delta=.0001, minimum_recovery_states=dict(fit=100, holdout=15),
            minimum_recovery_sequences=dict(fit=10, holdout=3))
        namespace['control'] = control
        execute([inherit], namespace)
        self.assertEqual(namespace['plan'], control)
        gate = next(node for node in main.body if isinstance(node, ast.If))
        namespace['args'] = SimpleNamespace(mode='smoke')
        smoke_body = ast.If(test=gate.test, body=gate.body, orelse=[])
        ast.copy_location(smoke_body, gate)
        execute([smoke_body], namespace)
        self.assertEqual(namespace['plan'], dict(control, min_epochs=2, max_epochs=2))

    def test_launcher_dispatches_only_requested_stage_and_gpu(self):
        path = ROOT / 'tools/launch_refreshed_gain.py'
        main = function(ast.parse(path.read_text()), 'main')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'reports').mkdir()
            for mode in ('smoke', 'formal'):
                with self.subTest(mode=mode):
                    plan = dict(mode=mode, launch_allowed=True)
                    write_json(root / (
                        'reports/REFRESHED_GAIN_PARENT_LOSS_%s_PLAN_20261030.json'
                        % mode.upper()), plan)
                    calls = []
                    args = SimpleNamespace(mode=mode, gpu='3')
                    parser = SimpleNamespace(
                        add_argument=lambda *a, **k: None, parse_args=lambda: args)
                    namespace = dict(
                        ROOT=root, json=json, print=lambda *a, **k: None,
                        argparse=SimpleNamespace(ArgumentParser=lambda: parser),
                        launch=lambda *a: calls.append(a))
                    execute([main], namespace)
                    namespace['main']()
                    self.assertEqual(calls, [(
                        root / (
                            'reports/REFRESHED_GAIN_PARENT_LOSS_%s_PLAN_20261030.json'
                            % mode.upper()),
                        'refreshed_query_gain_parent_loss_%s_20261030' % mode, '3')])

    def test_launcher_measures_free_memory_without_starting_real_processes(self):
        tree = ast.parse((ROOT / 'tools/launch_refreshed_gain.py').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'reports').mkdir()
            plan_path = root / 'reports/plan.json'
            write_json(plan_path, dict(mode='smoke'))
            calls = []
            memory = ['99999', '99999', '99999', '1023']

            def check_output(command, **kwargs):
                self.assertIn('--query-gpu=memory.free', command)
                return '\n'.join(memory)

            def popen(command, **kwargs):
                calls.append(command)
                self.assertTrue(kwargs['start_new_session'])
                return SimpleNamespace(pid=42)

            namespace = execute([function(tree, 'launch')], dict(
                ROOT=root, json=json, sha=sha,
                subprocess=SimpleNamespace(check_output=check_output, Popen=popen,
                                           STDOUT=-2),
                os=SimpleNamespace(environ={}), time=SimpleNamespace(time=lambda: 0)))
            launch = namespace['launch']
            with self.assertRaises(AssertionError):
                launch(plan_path, 'smoke_fixture', '3')
            self.assertEqual(calls, [])
            memory[3] = '1024'
            result = launch(plan_path, 'smoke_fixture', '3')
            self.assertEqual(result['observed_free_mib'], 1024)
            self.assertEqual(result['plan_sha256'], sha(plan_path))
            self.assertEqual(len(calls), 1)
            command = calls[0]
            self.assertEqual(command[command.index('--gpu') + 1], '3')
            self.assertEqual(command[command.index('--plan') + 1], 'reports/plan.json')
            self.assertIn(str(root / 'scripts/train_refreshed_query_gain.py'), command)


if __name__ == '__main__':
    unittest.main()
