"""Check M2 no-match cost, coverage and observed profile-source drift."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m2_compute", ROOT / "tools/summarize_e0_m2_full_compute.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class M2FullComputeTests(unittest.TestCase):
    def write_trace(self, path, count):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "modality", "action", "primary_tokens",
                             "auxiliary_tokens", "m2_invoked", "m2_class_index"])
            for frame in range(1, count + 1):
                invoked = int((frame - 1) % 25 == 0)
                selected = (0 if frame == 1 else 4) if invoked else -1
                writer.writerow([frame, 0, 0, 385, 0, invoked, selected])

    def test_no_match_still_costs_full_candidate_generation(self):
        routes = {(0, 0): dict(primary_tokens=385, auxiliary_tokens=0)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.write_trace(path, 26)
            counts, calls, matches = MODULE.trace_counts(path, 27, routes)
            self.assertEqual((counts[(0, 0)], calls, matches), (26, 2, 1))
            profile = dict(routes=[dict(modality=m, action=a, macs=10)
                                   for m in range(2) for a in range(5)],
                           m2_identity_head=[dict(candidates=24, macs=3,
                                                 independent_formula_match=True)],
                           m2_initial_anchor=dict(macs=7))
            totals = MODULE.compute_totals(
                profile, dict(candidate_primary_macs_per_invocation=100), counts, calls, 1)
            self.assertEqual(totals["m2_candidate_primary_macs"], 200)
            self.assertEqual(totals["m2_identity_head_macs"], 6)
            self.assertEqual(totals["tracked_frame_macs"], 466)
            self.assertEqual(totals["all_frame_macs"], 473)

    def test_missing_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.write_trace(path, 1)
            with self.assertRaises(AssertionError):
                MODULE.trace_counts(path, 3, {(0, 0): dict(primary_tokens=385, auxiliary_tokens=0)})

    def test_skipped_m2_invocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text("frame,modality,action,primary_tokens,auxiliary_tokens,"
                            "m2_invoked,m2_class_index\n1,0,0,385,0,0,-1\n")
            with self.assertRaises(AssertionError):
                MODULE.trace_counts(path, 2, {(0, 0): dict(primary_tokens=385, auxiliary_tokens=0)})

    def test_profile_allows_only_observed_wrapper_difference(self):
        route = json.loads(
            (ROOT / "reports/E0_ROUTE_MAC_CPU_CROSSCHECK_P25_20261101.json").read_bytes())
        candidate = json.loads(
            (ROOT / "reports/E0_M2_CANDIDATE_PRIMARY_MAC_CPU_R1_20261101.json").read_bytes())
        plan = dict(source_sha256=copy.deepcopy(candidate["source_sha256"]),
                    input_sha256=candidate["input_sha256"],
                    parent_checkpoint_sha256=candidate["parent_checkpoint_sha256"])
        _, differences = MODULE.bind_profiles(route, candidate, plan)
        self.assertEqual(differences, ["scripts/run_task_four_gpu.py"])
        route["source_sha256"]["dual_rgbt/model.py"] = "changed"
        with self.assertRaises(AssertionError):
            MODULE.bind_profiles(route, candidate, plan)

    def test_candidate_count_mismatch_is_rejected(self):
        route = json.loads(
            (ROOT / "reports/E0_ROUTE_MAC_CPU_CROSSCHECK_P25_20261101.json").read_bytes())
        candidate = json.loads(
            (ROOT / "reports/E0_M2_CANDIDATE_PRIMARY_MAC_CPU_R1_20261101.json").read_bytes())
        plan = dict(source_sha256=candidate["source_sha256"],
                    input_sha256=candidate["input_sha256"],
                    parent_checkpoint_sha256=candidate["parent_checkpoint_sha256"])
        candidate["candidate_primary_macs_per_invocation"] //= 2
        with self.assertRaises(AssertionError):
            MODULE.bind_profiles(route, candidate, plan)


if __name__ == "__main__":
    unittest.main()
