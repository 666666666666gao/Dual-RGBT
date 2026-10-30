"""Run the size-tail test evaluator with the verified server26 image mapping."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_inputs import sha
from run_cached_gain_server26_test import read_pair, ORIGINAL_PREFIX, LOCAL_PREFIX
import eval_size_tail_test as evaluation


def main():
    plan_path = sys.argv[sys.argv.index("--plan") + 1]
    plan = json.loads((ROOT / plan_path).read_text())
    assert plan["mode"] == "size_tail_diagnostic"
    assert plan["exploratory_test"] is True
    assert plan["runtime_provider"] == dict(ssh_alias="2026", project=str(ROOT))
    assert plan["source_sha256"]["tools/run_size_tail_server26_test.py"] == sha(Path(__file__))
    receipt_path = ROOT / plan["dataset_mapping_receipt"]
    assert sha(receipt_path) == plan["dataset_mapping_receipt_sha256"]
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "TEST_MAPPING_VERIFIED"
    assert receipt["original_prefix"] == ORIGINAL_PREFIX
    assert receipt["local_prefix"] == LOCAL_PREFIX
    assert (receipt["sequences"], receipt["frames"]) == (245, 220703)
    evaluation.read_pair = read_pair
    evaluation.main()


if __name__ == "__main__":
    main()
