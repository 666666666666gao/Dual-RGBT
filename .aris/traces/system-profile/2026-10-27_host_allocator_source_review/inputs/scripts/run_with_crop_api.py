"""Run an existing train/track entry with the admitted crop API.

The original entry and network stay unchanged. The alias is set before either
tracker or training imports bind crop_pair. Current original jobs do not use it.
"""
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'scripts'))


def main():
    entry = sys.argv[1]
    assert entry in ('train_trajectory_policy.py', 'run_tracking.py', 'compare_tracking_latency.py')
    import dual_rgbt.data as data
    from profile_crop_api import six_channel_crop
    data.crop_pair = six_channel_crop
    sys.argv = [entry] + sys.argv[2:]
    runpy.run_path(str(root / 'scripts' / entry), run_name='__main__')


if __name__ == '__main__':
    main()
