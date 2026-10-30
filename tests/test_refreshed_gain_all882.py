"""Fixtures for expanded coverage and exact smoke-to-formal admission."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'all882', ROOT / 'tools/prepare_refreshed_gain_all882.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_partition_preserves_matched100_and_remaining782_membership(self):
        manifest = dict(train=[dict(name='sequence_%03d' % i) for i in range(882)])
        names, fit, held = MODULE.partition(manifest)
        self.assertEqual(len(fit), 706)
        self.assertEqual(len(held), 176)
        self.assertFalse(set(fit) & set(held))
        self.assertEqual(set(held) & set(names[:100]),
                         set(names[16:20] + names[20:100][4::5]))
        self.assertEqual(set(held) & set(names[100:]), set(names[100:][4::5]))

    def cache_fixture(self):
        names, fit, held = MODULE.partition(
            dict(train=[dict(name='sequence_%03d' % i) for i in range(882)]))
        for phase, selected in (('matched100', names[:100]), ('remaining782', names[100:])):
            for shard in range(8):
                run = 'policy_refresh_e6_%s_s%d_20261030_full' % (phase, shard)
                subset = selected[shard::8]
                config = dict(native_command=['collector'], selected_sequences=subset,
                              plan=dict(phase=phase, shard=shard, expected_frames=len(subset),
                                        fit_sequences=[n for n in subset if n in fit],
                                        holdout_sequences=[n for n in subset if n in held]))
                config['selected_sequences'] = config['plan']['fit_sequences'] + config['plan']['holdout_sequences']
                directory = self.root / 'runs' / run
                receipt = directory / 'COMPLETE.json'
                native = self.root / 'runs/pipeline_tasks' / ('collect_' + run) / 'status.json'
                write(directory / 'config.json', config)
                write(receipt, dict(status='CACHE_COMPLETE', sequences=len(subset), frames=len(subset)))
                write(native, dict(state='COMPLETE', exit_code=0, command=['collector']))
                write(self.root / 'reports' / (run + '_audit.json'), dict(
                    status='VERIFIED_CACHE_DIAGNOSTIC', complete_planned_coverage=True,
                    smoke=False, native_sha256=MODULE.sha(native), receipt_sha256=MODULE.sha(receipt)))
        return names, set(held)

    def test_all_sixteen_caches_required_with_fixed_partitions(self):
        names, held = self.cache_fixture()
        self.assertEqual(len(MODULE.collect_entries(self.root, names, held)), 16)
        with self.assertRaises(AssertionError):
            MODULE.collect_entries(self.root, names, set())
        missing = self.root / 'reports/policy_refresh_e6_remaining782_s7_20261030_full_audit.json'
        missing.unlink()
        with self.assertRaises(FileNotFoundError):
            MODULE.collect_entries(self.root, names, held)

    def test_native_or_receipt_tampering_rejected(self):
        names, held = self.cache_fixture()
        native = self.root / 'runs/pipeline_tasks/collect_policy_refresh_e6_remaining782_s7_20261030_full/status.json'
        write(native, dict(state='RUNNING', command=['collector']))
        with self.assertRaises(AssertionError):
            MODULE.collect_entries(self.root, names, held)

    def smoke_fixture(self):
        plan = dict(MODULE.RECIPE, mode='formal', formal_launch_allowed=True,
                    caches=['all16'], source_sha256={'trainer': 'bound'},
                    fit_sequences=['fit'], holdout_sequences=['held'])
        smoke = copy.deepcopy(plan)
        smoke.update(mode='smoke', formal_launch_allowed=False, min_epochs=2, max_epochs=2)
        path = self.root / ('reports/' + MODULE.PREFIX + '_SMOKE_PLAN_20261030.json')
        write(path, smoke)
        directory = self.root / 'runs' / (MODULE.RUN_PREFIX + '_smoke_20261030')
        task = 'train_smoke'
        config = dict(plan=smoke, plan_sha256=MODULE.sha(path), task=task,
                      native_command=['trainer'],
                      cache_policy_prediction_max_abs_error=dict(fit=0., holdout=0.))
        result = dict(status='SANITY_PASS', strict_reload_bit_equal=True, epochs=2,
                      updates=4, plan_sha256=MODULE.sha(path),
                      changed_state_tensors=[
                          'query_gain.0.weight', 'query_gain.0.bias',
                          'query_gain.2.weight', 'query_gain.2.bias'],
                      baseline=dict(
                          fit=dict(sampled_recoverable_states=100, recovery_sequences=10),
                          holdout=dict(sampled_recoverable_states=15, recovery_sequences=3)))
        write(directory / 'config.json', config)
        write(directory / 'COMPLETE.json', result)
        write(self.root / 'runs/pipeline_tasks' / task / 'status.json',
              dict(state='COMPLETE', exit_code=0, command=['trainer']))
        return plan, directory, result

    def test_formal_binds_exact_smoke_inputs_and_preserves_recovery_gate(self):
        plan, directory, result = self.smoke_fixture()
        MODULE.bind_smoke(self.root, copy.deepcopy(plan))
        changed = copy.deepcopy(plan)
        changed['caches'] = ['different16']
        with self.assertRaises(AssertionError):
            MODULE.bind_smoke(self.root, changed)
        result['baseline']['holdout']['sampled_recoverable_states'] = 14
        write(directory / 'COMPLETE.json', result)
        with self.assertRaises(AssertionError):
            MODULE.bind_smoke(self.root, copy.deepcopy(plan))


if __name__ == '__main__':
    unittest.main()
