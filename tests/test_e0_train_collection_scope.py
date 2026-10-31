"""Check exact paired TRAIN shard/partition admission without importing CUDA."""
import ast
from pathlib import Path
import unittest

source = Path(__file__).resolve().parents[1] / 'scripts/collect_e0_reliability_train.py'
module = ast.parse(source.read_text())
module.body = [next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'shard_sequences')]
namespace = {}
exec(compile(module, str(source), 'exec'), namespace)
shard_sequences = namespace['shard_sequences']


class E0TrainScopeTest(unittest.TestCase):
    def test_eight_shards_cover_each_training_sequence_once(self):
        names = ['s%03d' % i for i in range(882)]
        parts = [shard_sequences(names, names[:706], names[706:], {'dev', 'test'}, i, 8)
                 for i in range(8)]
        self.assertEqual(sorted(s for part in parts for s in part), names)
        self.assertEqual([len(p) for p in parts], [111, 111, 110, 110, 110, 110, 110, 110])

    def test_leakage_duplicate_partition_and_wrong_shards_rejected(self):
        names = ['s%03d' % i for i in range(882)]
        with self.assertRaises(AssertionError):
            shard_sequences(names, names[:706], names[706:], {names[0]}, 0, 8)
        with self.assertRaises(AssertionError):
            shard_sequences(names, names[:706], names[705:881], set(), 0, 8)
        with self.assertRaises(AssertionError):
            shard_sequences(names, names[:706], names[706:], set(), 0, 4)


if __name__ == '__main__':
    unittest.main()
