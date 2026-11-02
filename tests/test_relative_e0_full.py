"""Check fixed full TRAIN recipe and the exact whitelist of late evidence hashes."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from relative_e0_full_contract import BINDINGS, STAGE, bind_draft, recipe
from relative_e0_smoke_contract import INITIAL_SHA


class RelativeFullTest(unittest.TestCase):
    def test_recipe_does_not_allow_test_or_posthoc_training_changes(self):
        plan = dict(stage=STAGE, phase="full", split="train", test_eligible=False,
                    scientific_result_audit_accepted=False, online_allowed=False,
                    device="cuda", assigned_gpu="0", seed=2027, epochs=12, batch_sets=512,
                    lr=.0003, weight_decay=.0001, initialization_sha256=INITIAL_SHA)
        recipe(plan)
        for key, value in (("split", "test"), ("test_eligible", True), ("seed", 2028),
                           ("epochs", 20), ("lr", .001), ("online_allowed", True)):
            with self.assertRaises(AssertionError):
                recipe(dict(plan, **{key: value}))

    def test_late_binding_changes_only_declared_receipt_hashes(self):
        draft = dict(lr=.0003, task="fixed", source_sha256={"x.py": "fixed"})
        draft.update({key: None for key in BINDINGS})
        draft["launch_allowed"] = False
        plan = dict(draft)
        plan.update({key: "a" * 64 for key in BINDINGS})
        plan["launch_allowed"] = True
        bind_draft(plan, draft)
        for key, value in (("lr", .001), ("task", "replacement"),
                           ("source_sha256", {"x.py": "changed"})):
            with self.assertRaises(AssertionError):
                bind_draft(dict(plan, **{key: value}), draft)
        with self.assertRaises(AssertionError):
            bind_draft(dict(plan, update_smoke_native_sha256=None), draft)


if __name__ == "__main__":
    unittest.main()
