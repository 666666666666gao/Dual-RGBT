"""Run the exploratory LasHeR test evaluator with the verified server26 copy."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
import dual_rgbt.data as data
import eval_cached_query_gain_test as evaluation

ORIGINAL_PREFIX = '/home/gaob/Dual-RGBT/data/LasHeR/test/'
LOCAL_PREFIX = '/data2/gb/Dual-RGBT/data/LasHeR/test/'
original_read_pair = data.read_pair


def read_pair(sequence, frame):
    pair = {}
    for key in ('rgb', 'tir'):
        path = sequence[key][frame]
        assert path.startswith(ORIGINAL_PREFIX)
        pair[key] = [LOCAL_PREFIX + path[len(ORIGINAL_PREFIX):]]
    return original_read_pair(pair, 0)


def main():
    plan_path = sys.argv[sys.argv.index('--plan') + 1]
    plan = json.loads((ROOT / plan_path).read_text())
    assert plan['exploratory_test'] is True
    assert plan['runtime_provider']['ssh_alias'] == '2026'
    assert plan['runtime_provider']['project'] == str(ROOT)
    assert plan['source_sha256']['tools/run_cached_gain_server26_test.py'] == sha(Path(__file__))
    receipt_path = ROOT / plan['dataset_mapping_receipt']
    assert sha(receipt_path) == plan['dataset_mapping_receipt_sha256']
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'TEST_MAPPING_VERIFIED'
    assert receipt['original_prefix'] == ORIGINAL_PREFIX
    assert receipt['local_prefix'] == LOCAL_PREFIX
    assert (receipt['sequences'], receipt['frames']) == (245, 220703)
    evaluation.read_pair = read_pair
    evaluation.main()


if __name__ == '__main__':
    main()
