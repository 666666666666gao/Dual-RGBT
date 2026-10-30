"""Use the sealed parent evaluator with an explicit server26 image mapping."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import sha
import dual_rgbt.data as data
import eval_cached_query_gain as evaluation

ORIGINAL_PREFIX = '/data2/wangwj/dataset/LasHeR-0428/trainingset/'
LOCAL_PREFIX = '/data2/gb/Dual-RGBT/data/LasHeR-0428/trainingset/'
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
    assert plan['runtime_provider']['project'] == str(ROOT)
    assert plan['runtime_provider']['ssh_alias'] == '2026'
    assert plan['source_sha256']['tools/run_cached_gain_server26.py'] == sha(Path(__file__))
    validation = ROOT / plan['dataset_mapping_receipt']
    assert sha(validation) == plan['dataset_mapping_receipt_sha256']
    record = json.loads(validation.read_text())
    assert record['status'] == 'MAPPING_VERIFIED'
    assert record['original_prefix'] == ORIGINAL_PREFIX and record['local_prefix'] == LOCAL_PREFIX
    assert record['frames'] == 54646 and record['sequences'] == 97
    evaluation.read_pair = read_pair
    evaluation.main()


if __name__ == '__main__':
    main()
