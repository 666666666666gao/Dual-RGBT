"""Prepare disabled LasHeR-first evaluation of fixed P1B epoch005 weights."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def prepare(original, audit, audit_sha256):
    assert audit["status"] == "P1B_FIXED_EPOCH005_CPU_AUDIT_PASS"
    assert audit["cuda_initialized"] is False
    assert audit["evaluation_admitted"] is False
    assert audit["executor_preserved"] is audit["all_policy_modules_changed"] is True
    assert audit["branch"] == original["branch"]
    assert audit["host"] == original["host"]
    assert audit["root"] == original["root"]
    assert audit["source_sha256"] == original["source_sha256"]
    assert len(audit["source_sha256"]) == 49
    assert audit["factory"] == original["factory"]
    branch, host = audit["branch"], audit["host"]
    parent = PurePosixPath(original["smoke"]["command"][0]).parents[3]
    storage = PurePosixPath(original["jobs"][0]["run"]).parents[1]
    root = str(storage / ("p1b_evaluation_%s_r2_20261104" % branch))
    review_path = root + "/reports/P1B_%s_EVALUATION_REVIEW_R2_20261104.json" % branch.upper()
    checkpoint = audit["checkpoint"]
    assert PurePosixPath(checkpoint["path"]).name == "epoch_005.pth"

    def job(source, smoke):
        task = ("%s_p1b_e005_lasher_%s_p%s_r2_20261104" %
                ("smoke" if smoke else "eval", branch, host[-2:]))
        run = str(storage / "p1b_eval_runs" / task)
        command = [
            str(parent / "conda-envs/dual_rgbt/bin/python"),
            root + "/scripts/eval_p1b_completed.py",
        ]
        for name, value in (
            ("checkpoint", checkpoint["path"]),
            ("checkpoint-sha256", checkpoint["sha256"]),
            ("factory", audit["factory"]["path"]),
            ("factory-sha256", audit["factory"]["sha256"]),
            ("manifest", source["manifest"]["path"]),
            ("manifest-sha256", source["manifest"]["sha256"]),
            ("source-review", review_path),
            ("source-review-sha256", "PENDING_REVIEW_SHA256"),
            ("task", task), ("run", run), ("cost-weight", "0.05"),
            ("smoke-frames", "11" if smoke else "0"),
        ):
            command.extend(["--" + name, value])
        return dict(
            task=task, run=run, command=command, gpu="0", smoke=smoke,
            manifest=source["manifest"], sequences=1 if smoke else 245,
            frames=11 if smoke else 220703,
            exploratory_test=not smoke, test_eligible=False,
            score_output=root + "/reports/" + task + "_SCORES.json")

    lasher = next(row for row in original["jobs"] if row["benchmark"] == "LasHeR")
    assert lasher["sequences"] == 245 and lasher["frames"] == 220703
    sources = dict(audit["source_sha256"])
    sources["scripts/eval_p1b_completed.py"] = hashlib.sha256(
        (ROOT / "scripts/eval_p1b_completed.py").read_bytes()).hexdigest()
    return dict(
        stage="P1B_fixed_epoch005_evaluation",
        host=host, branch=branch, root=root, source_root=audit["root"],
        launch_allowed=False, checkpoint=checkpoint, factory=audit["factory"],
        source_sha256=sources, fit_audit_sha256=audit_sha256,
        fit_evidence=audit["archived_evidence"], fit_plan_binding=audit["fit_plan_binding"],
        review=dict(path=review_path, sha256="PENDING"),
        toolkit=original["toolkit"], protocol=original["protocol"],
        scoring_interpreter=str(parent / "conda-envs/cpu-scoring/bin/python"),
        smoke=job(original["smoke"], True), jobs=[job(lasher, False)],
        exploratory_test=True, test_eligible=False,
        selection="Fixed recipe-final epoch005; no TEST-based selection.",
        scope="P1B round1 matched policy; not C2 or a second TRAIN refresh round.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    args = parser.parse_args()
    folder = ROOT / "reports/p1b_evaluation_r2_20261104" / args.branch
    assert not folder.exists()
    original = json.loads((ROOT / "reports/p1a_evaluation_r1_20261102" /
                           args.branch / "draft.json").read_bytes())
    raw = (ROOT / "reports/p1b_epoch005_evaluation_audit_r1_20261104" /
           args.branch / "audit.json").read_bytes()
    draft = prepare(original, json.loads(raw), hashlib.sha256(raw).hexdigest())
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "draft.json").write_text(json.dumps(draft, indent=2) + "\n")
    print(json.dumps(dict(
        branch=args.branch, root=draft["root"], launch_allowed=False,
        checkpoint=draft["checkpoint"]), indent=2))


if __name__ == "__main__":
    main()
