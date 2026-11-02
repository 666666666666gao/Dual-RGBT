"""Verify full frame joins and fixed shard/native identities before deployment."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
from full_current_e0_supplement_contract import native_command, partitions, verify_native
from current_e0_supplement_contract import verify_arrays
from test_e0_current_supplement import SupplementTest


class FullSupplementTest(unittest.TestCase):
    def test_entire_trajectory_and_later_sample_join(self):
        fixture = SupplementTest()
        fixture.setUp()
        arrays = fixture.values
        for key in ("frame", "e0_features", "e0_coverage", "normalized_box", "geometry",
                    "current_box", "previous_box", "observation", "route"):
            arrays[key] = np.concatenate([arrays[key], arrays[key]])
        arrays["frame"] = np.asarray([1, 26, 51, 76], np.int64)
        arrays["predictions"] = fixture.original["predictions"].copy()
        arrays["frame_seconds"] = np.ones(100, np.float64)
        result = verify_arrays(arrays, fixture.sequence, fixture.original, count=100)
        self.assertEqual(result["candidate_sets"], 4)
        self.assertEqual(result["frames"], 100)
        for key, index in (("frame", 2), ("current_box", (3, 0)),
                           ("predictions", (99, 0))):
            with self.subTest(key=key):
                values = copy.deepcopy(arrays)
                values[key][index] += 1
                with self.assertRaises(AssertionError):
                    verify_arrays(values, fixture.sequence, fixture.original, count=100)

    def fixture(self):
        names = ["train_%03d" % i for i in range(882)]
        lengths = [527] * 189 + [520] * 165 + [519] * 528
        sequences = {name: dict(rgb=[None] * n, valid_frames=list(range(n)))
                     for name, n in zip(names, lengths)}
        collection = dict(sequence_names=names, fit_sequences=names[:706],
                          holdout_sequences=names[706:])
        plan = dict(collection, expected_frames=459435, expected_candidate_sets=18711,
                    shards=8, jobs=[dict(shard=i, assigned_gpu=str(i % 2),
                                        task="collect_s%d" % i, run="runs/collect_s%d" % i,
                                        sequence_names=names[i::8]) for i in range(8)])
        return plan, collection, sequences

    def test_full_partition_and_eight_shard_coverage(self):
        plan, collection, sequences = self.fixture()
        partitions(plan, collection, sequences)
        for change in ("duplicate_sequence", "held_overlap", "wrong_shard_names",
                       "duplicate_task", "wrong_gpu", "wrong_frames", "wrong_sets"):
            with self.subTest(change=change):
                bad = copy.deepcopy(plan)
                if change == "duplicate_sequence":
                    bad["sequence_names"][-1] = bad["sequence_names"][0]
                elif change == "held_overlap":
                    bad["holdout_sequences"][0] = bad["fit_sequences"][0]
                elif change == "wrong_shard_names":
                    bad["jobs"][0]["sequence_names"] = bad["jobs"][1]["sequence_names"]
                elif change == "duplicate_task":
                    bad["jobs"][1]["task"] = bad["jobs"][0]["task"]
                elif change == "wrong_gpu":
                    bad["jobs"][0]["assigned_gpu"] = "2"
                elif change == "wrong_frames":
                    bad["expected_frames"] -= 1
                else:
                    bad["expected_candidate_sets"] -= 1
                with self.assertRaises(AssertionError):
                    partitions(bad, collection, sequences)

    def test_full_native_task_host_command_gpu_and_shard(self):
        plan, _, _ = self.fixture()
        plan["target_hostname"] = "actual26"
        root = Path("/data2/gb/Dual-RGBT")
        relative = "reports/full/plan.json"
        command = native_command(root, relative, 0)
        native = dict(state="RUNNING", task=plan["jobs"][0]["task"], hostname="actual26",
                      physical_gpu="0", command=command)
        with patch("full_current_e0_supplement_contract.socket.gethostname", return_value="actual26"):
            self.assertEqual(verify_native(root, plan, native, relative, 0, "RUNNING"), command)
            for key, wrong in (("task", "wrong"), ("hostname", "wrong"),
                               ("physical_gpu", "1"), ("command", command[:-1] + ["1"]),
                               ("state", "COMPLETE")):
                with self.subTest(key=key), self.assertRaises(AssertionError):
                    verify_native(root, plan, dict(native, **{key: wrong}), relative, 0, "RUNNING")


if __name__ == "__main__":
    unittest.main()
