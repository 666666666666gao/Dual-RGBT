"""Checkpoint retention must keep the newest five and leave other runs alone."""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


class RetentionTests(unittest.TestCase):
    def test_newest_five_only_and_no_overwrite(self):
        source = Path(__file__).resolve().parents[1] / 'scripts/train_size_tail.py'
        module = ast.parse(source.read_text())
        function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'save_epoch')
        namespace = {'torch': SimpleNamespace(save=lambda state, path: path.write_text(str(state['epoch'])))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'new_run'
            run.mkdir()
            protected = root / 'epoch_001.pth'
            protected.write_text('parent')
            for epoch in range(1, 8):
                namespace['save_epoch'](run, {'epoch': epoch})
            self.assertEqual([p.name for p in sorted(run.glob('*.pth'))],
                             ['epoch_%03d.pth' % e for e in range(3, 8)])
            self.assertEqual(protected.read_text(), 'parent')
            with self.assertRaises(AssertionError):
                namespace['save_epoch'](run, {'epoch': 7})


if __name__ == '__main__':
    unittest.main()
