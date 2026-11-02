"""Check diagnostic confusion orientation and recovery-window eligibility."""
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "summary", Path(__file__).resolve().parents[1] /
    "tools/summarize_e0_m1_m2_train_diagnostic.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(frame, before, after, predicted=0, selected=-1, valid=1):
    return dict(frame=str(frame), valid=str(valid), e0_iou=str(before), output_iou=str(after),
                m1_state=str(predicted), m2_class_index=str(selected),
                m2_invoked=str(int(selected >= 0)),
                template_write_proposed="0", template_write_blocked="0")


class SummaryTests(unittest.TestCase):
    def test_labels_and_replacement_counts(self):
        rows = [row(1, .5, .2, 2, 1), row(2, .09, .8, 2, 2),
                row(3, .1, .1), row(4, "", "", 2, 0, valid=0)]
        result = MODULE.aggregate(rows)
        self.assertEqual(result["confusion_gt_rows_predicted_columns"],
                         [[0, 0, 1], [1, 0, 0], [0, 0, 1]])
        self.assertEqual(result["m2_invocations"], 3)
        self.assertEqual(result["no_match"], 1)
        self.assertEqual(result["replacement_better"], 1)
        self.assertEqual(result["replacement_worse"], 1)
        self.assertEqual(result["lost_to_at_least_half"], 1)
        self.assertEqual(result["reliable_to_below_half"], 1)

    def test_recovery_persistence_excludes_unannotated_future(self):
        rows = [row(1, .05, .8, 2, 1)] + [row(i, .8, .8) for i in range(2, 12)]
        rows[3]["output_iou"] = ".2"
        result = MODULE.persistence(rows)
        self.assertEqual(result["1"], dict(eligible=1, sustained=1))
        self.assertEqual(result["5"], dict(eligible=1, sustained=0))
        rows[3]["valid"] = "0"
        self.assertEqual(MODULE.persistence(rows)["5"], dict(eligible=0, sustained=0))
        self.assertEqual(MODULE.persistence(rows[:2])["10"], dict(eligible=0, sustained=0))


if __name__ == "__main__":
    unittest.main()
