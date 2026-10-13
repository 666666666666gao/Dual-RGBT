"""Paired numerical admission of the crop API using the original C sanity.

Two sequential sandbox children run the unchanged ten-frame/two-update train
sanity, including its strict reload witness. Only the second child's in-memory
crop function changes. Neither sandbox checkpoint is a production candidate.
"""
import argparse
import csv
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'scripts'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child-mode', choices=('original', 'six_channel'))
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    policy = 'runs/stage_c_complete_20261013/latest.pth'
    expected_sha = '94be6239f7f6864cc193fff7c21d67c9479924c9432039c6ff16cd0d1223f15f'
    assert hashlib.sha256((root / policy).read_bytes()).hexdigest() == expected_sha
    names = dict(original='crop_api_original_sanity_20261013',
                 six_channel='crop_api_six_channel_sanity_20261013')
    if args.child_mode:
        import dual_rgbt.data as data
        if args.child_mode == 'six_channel':
            from profile_crop_api import six_channel_crop
            data.crop_pair = six_channel_crop
        import dual_rgbt.tracker as tracking
        snapshots = []

        class RecordedTracker(tracking.Tracker):
            def record(self, route=None):
                snapshots.append(dict(
                    frame=self.frame,
                    state_sha256=hashlib.sha256(self.state.tobytes()).hexdigest(),
                    templates_sha256=hashlib.sha256(self.templates.cpu().numpy().tobytes()).hexdigest(),
                    history_sha256=hashlib.sha256(self.history.cpu().numpy().tobytes()).hexdigest(),
                    route={k: route[k] for k in ('modality', 'action', 'confidence',
                                                'primary_tokens', 'auxiliary_tokens')} if route else None))

            def initialize(self, image, initial_box):
                elapsed = super().initialize(image, initial_box)
                self.record()
                return elapsed

            def track(self, image):
                prediction, route = super().track(image)
                self.record(route)
                return prediction, route

        tracking.Tracker = RecordedTracker
        sys.argv = ['train_trajectory_policy.py', '--policy', policy,
                    '--run', names[args.child_mode], '--sanity']
        runpy.run_path(str(root / 'scripts/train_trajectory_policy.py'), run_name='__main__')
        (root / 'runs' / names[args.child_mode] / 'causal_snapshots.json').write_text(
            json.dumps(snapshots, indent=2))
        return

    output = root / 'reports/crop_runtime_check.json'
    assert not output.exists()
    assert all(not (root / 'runs' / name).exists() for name in names.values())
    receipts = []
    for mode, name in names.items():
        command = [sys.executable, str(Path(__file__).resolve()), '--child-mode', mode]
        started = time.time()
        with (root / 'reports' / (name + '.log')).open('wb') as log:
            child = subprocess.Popen(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT)
            receipt = dict(mode=mode, command=command, pid=child.pid, started=started,
                           completed=None, exit_code=None)
            receipts.append(receipt)
            (root / 'reports/crop_runtime_children.json').write_text(json.dumps(receipts, indent=2))
            exit_code = child.wait()
        receipt.update(completed=time.time(), exit_code=exit_code)
        (root / 'reports/crop_runtime_children.json').write_text(json.dumps(receipts, indent=2))
        assert exit_code == 0, receipt

    import torch
    states, rows, sanity, snapshots = [], [], [], []
    for name in names.values():
        run = root / 'runs' / name
        states.append(torch.load(run / 'latest.pth', map_location='cpu'))
        with (run / 'trajectory_steps.csv').open(newline='') as handle:
            rows.append(list(csv.DictReader(handle)))
        sanity.append(json.loads((run / 'SANITY.json').read_text()))
        snapshots.append(json.loads((run / 'causal_snapshots.json').read_text()))
    assert rows[0] == rows[1] and len(rows[0]) == 2
    assert snapshots[0] == snapshots[1] and len(snapshots[0]) == 11
    assert [r['frame'] for r in snapshots[0]] == list(range(11))
    assert all(r['status'] == 'PASS' and r['steps'] == 2 and r['tracked_frames'] == 10
               and r['saved_state_reload_bit_equal'] and r['physical_gpu'] == '1' for r in sanity)
    assert states[0]['net'].keys() == states[1]['net'].keys()
    assert all(torch.equal(states[0]['net'][key], states[1]['net'][key]) for key in states[0]['net'])
    assert states[0]['step'] == states[1]['step'] == 2
    report = dict(status='PASS', input_checkpoint_sha256=expected_sha, physical_gpu='1',
                  children=receipts, sanity=sanity, cross_mode_supervision_rows_equal=True,
                  cross_mode_causal_snapshots_equal=11,
                  cross_mode_net_tensors_equal=len(states[0]['net']),
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                  for p in ('scripts/check_crop_runtime.py', 'scripts/profile_crop_api.py',
                                            'scripts/profile_interior_crops.py', 'dual_rgbt/model.py',
                                            'scripts/train_trajectory_policy.py', 'dual_rgbt/data.py',
                                            'dual_rgbt/tracker.py')},
                  scope='Two sequential ten-frame train sandboxes with two real optimizer updates '
                        'and native strict reload each; not full trajectories, accuracy, speed '
                        'or production adoption. Sandbox weights are not selected for evaluation.')
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
