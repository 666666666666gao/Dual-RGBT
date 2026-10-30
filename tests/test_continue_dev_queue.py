"""Exercise controller ownership and refill with short-lived fake task wrappers."""
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(sys.platform == 'linux', 'Controller runs on Linux GPU hosts')
class QueueTests(unittest.TestCase):
    def test_adoption_four_gpu_fill_and_refill(self):
        source = Path(__file__).resolve().parents[1] / 'tools/continue_dev_queue.py'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tools').mkdir()
            shutil.copyfile(str(source), str(root / 'tools/continue_dev_queue.py'))
            (root / 'scripts').mkdir()
            (root / 'conda-envs/dual_rgbt/bin').mkdir(parents=True)
            (root / 'conda-envs/dual_rgbt/bin/python').symlink_to(sys.executable)
            (root / 'scripts/run_task_four_gpu.py').write_text(
                "import sys,pathlib,json\n"
                "r=pathlib.Path.cwd(); task=sys.argv[sys.argv.index('--task')+1]\n"
                "p=r/'runs/pipeline_tasks'/task; p.mkdir(parents=True)\n"
                "(p/'status.json').write_text(json.dumps(dict(state='COMPLETE',exit_code=0)))\n")
            path = root / 'state.json'
            entries = [dict(task='task%d' % i, run='run%d' % i, plan='plan.json', cost_weight=0.0) for i in range(5)]
            native = root / 'runs/pipeline_tasks/adopted'
            native.mkdir(parents=True)
            (native / 'status.json').write_text(json.dumps(dict(state='COMPLETE', exit_code=0)))
            path.write_text(json.dumps(dict(status='RUNNING', queue=entries,
                active={'0': dict(entry=dict(task='adopted'), pid=999999999)}, completed=[], failed=[])))
            samples = []
            real_sleep = time.sleep

            def sleep(seconds):
                self.assertEqual(seconds, 180)
                samples.append(json.loads(path.read_text()))
                real_sleep(.05)

            with patch.object(sys, 'argv', ['controller', '--state', str(path)]), \
                 patch.object(subprocess, 'check_output', return_value='0\n0\n0\n0\n'), \
                 patch.object(time, 'sleep', side_effect=sleep):
                runpy.run_path(str(root / 'tools/continue_dev_queue.py'), run_name='__main__')
            final = json.loads(path.read_text())
            self.assertEqual(final['status'], 'COMPLETE')
            self.assertEqual(len(samples[0]['active']), 4)
            self.assertEqual(len(final['completed']), 6)
            self.assertEqual(len({x['entry']['task'] for x in final['completed']}), 6)
            self.assertFalse(final['failed'])


if __name__ == '__main__':
    unittest.main()
