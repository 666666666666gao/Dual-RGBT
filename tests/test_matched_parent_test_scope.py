"""Exercise the evaluator's real split-admission function without importing CUDA."""
import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/eval_matched_parent_test.py'
scope_module = ast.parse(SOURCE.read_text())
scope_module.body = [next(node for node in scope_module.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'admitted_scope')]
namespace = {}
exec(compile(scope_module, str(SOURCE), 'exec'), namespace)
admitted_scope = namespace['admitted_scope']


class MatchedParentScopeTest(unittest.TestCase):
    def test_test_requires_full_explicit_exploratory_admission(self):
        self.assertEqual(admitted_scope('test', 0, True, True), (245, 220703))
        for smoke, exploratory, allowed in [(11, True, True), (0, False, True), (0, True, False)]:
            with self.assertRaises(AssertionError):
                admitted_scope('test', smoke, exploratory, allowed)

    def test_train_smoke_only_and_no_dev(self):
        self.assertEqual(admitted_scope('train_smoke', 11, True, True), (1, 11))
        for split, frames in [('dev', 0), ('dev', 11), ('train_smoke', 0)]:
            with self.assertRaises(AssertionError):
                admitted_scope(split, frames, True, True)


if __name__ == '__main__':
    unittest.main()
