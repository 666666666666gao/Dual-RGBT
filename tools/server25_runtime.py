"""Control the explicitly authorized additional server25 without changing server26."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/gaob/Dual-RGBT'
PYTHON = REMOTE + '/conda-envs/dual_rgbt/bin/python'
SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
       '-o', 'ConnectTimeout=10', '2025', 'python3 -']


def remote(code):
    result = subprocess.run(SSH, input=code, text=True, encoding='utf-8',
                            capture_output=True, timeout=120)
    if result.returncode:
        sys.stderr.write(result.stderr)
    result.check_returncode()
    return json.loads(result.stdout)
