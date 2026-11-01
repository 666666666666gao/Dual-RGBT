"""Reject observed provenance drift and incomplete compute traces."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "full_compute", ROOT / "tools/summarize_e0_full_compute.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullComputeTests(unittest.TestCase):
    def test_weighted_route_counts_and_missing_frame(self):
        routes = {(0, 0): dict(primary_tokens=385, auxiliary_tokens=0, macs=40),
                  (1, 4): dict(primary_tokens=385, auxiliary_tokens=385, macs=80)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame", "modality", "action", "primary_tokens", "auxiliary_tokens"])
                writer.writerow([1, 0, 0, 385, 0])
                writer.writerow([2, 1, 4, 385, 385])
            counts = MODULE.trace_counts(path, 3, routes)
            self.assertEqual(sum(routes[key]["macs"] * n for key, n in counts.items()), 120)
            with self.assertRaises(AssertionError):
                MODULE.trace_counts(path, 4, routes)

    def test_wrong_auxiliary_token_path_is_rejected(self):
        routes = {(0, 4): dict(primary_tokens=385, auxiliary_tokens=385)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text("frame,modality,action,primary_tokens,auxiliary_tokens\n1,0,4,385,193\n")
            with self.assertRaises(AssertionError):
                MODULE.trace_counts(path, 2, routes)

    def test_changed_cadtrack_source_is_rejected(self):
        names = ["dual_rgbt/model.py", "dual_rgbt/tracker.py",
                 "lib/models/cadtrack/vit_care.py", "lib/models/cadtrack/cadtrack.py",
                 "lib/config/cadtrack/config.py", "experiments/cadtrack/cadtrack.yaml"]
        sources = {name: "reviewed" for name in names}
        profile = dict(source_sha256=sources.copy())
        config = dict(source_sha256=sources.copy(),
                      input_sizes=dict(search=256, template=128, templates=2))
        self.assertEqual(MODULE.bind_sources(profile, config), sorted(names))
        config["source_sha256"]["lib/models/cadtrack/cadtrack.py"] = "changed"
        with self.assertRaises(AssertionError):
            MODULE.bind_sources(profile, config)


if __name__ == "__main__":
    unittest.main()
