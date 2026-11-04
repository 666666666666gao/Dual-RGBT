"""Prepare fixed epoch005 RGBT evaluations on the remaining reviewed GPUs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

from prepare_p1b_evaluation import prepare as prepare_fixed

ROOT = Path(__file__).resolve().parents[1]
GPUS = {("dropmoe", "rgbt210"): "2", ("dropmoe", "rgbt234"): "3",
        ("sot", "rgbt210"): "1", ("sot", "rgbt234"): "1"}
COUNTS = {"rgbt210": (210, 104706), "rgbt234": (234, 116650)}


def prepare(original, audit, audit_sha256, dependencies, benchmark):
    base = prepare_fixed(original, audit, audit_sha256, dependencies)
    branch, host = base["branch"], base["host"]
    gpu = GPUS[branch, benchmark]
    source = next(row for row in original["jobs"]
                  if row["benchmark"].lower() == benchmark)
    assert (source["sequences"], source["frames"]) == COUNTS[benchmark]
    storage = PurePosixPath(base["root"]).parent
    root = str(storage / ("p1b_evaluation_%s_%s_r1_20261104" % (branch, benchmark)))
    review = root + "/reports/P1B_%s_%s_EVALUATION_REVIEW_R1_20261104.json" % (
        branch.upper(), benchmark.upper())

    def job(smoke):
        result = copy.deepcopy(base["smoke"] if smoke else base["jobs"][0])
        task = "%s_p1b_e005_%s_%s_p%s_r1_20261104" % (
            "smoke" if smoke else "eval", benchmark, branch, host[-2:])
        manifest = original["smoke"]["manifest"] if smoke else source["manifest"]
        run = str(storage / "p1b_eval_runs" / task)
        replacements = {
            "--manifest": manifest["path"], "--manifest-sha256": manifest["sha256"],
            "--source-review": review, "--task": task, "--run": run}
        command = list(result["command"])
        command[1] = root + "/scripts/eval_p1b_completed.py"
        for key, value in replacements.items():
            command[command.index(key) + 1] = value
        result.update(
            command=command, task=task, run=run, gpu=gpu, manifest=manifest,
            sequences=1 if smoke else source["sequences"],
            frames=11 if smoke else source["frames"],
            score_output=root + "/reports/" + task + "_SCORES.json")
        return result

    base.update(root=root, benchmark=benchmark,
                review=dict(path=review, sha256="PENDING"),
                smoke=job(True), jobs=[job(False)],
                scheduling="SOT RGBT234 shares GPU1; require prior RGBT210 terminal "
                           "and a fresh empty-GPU check before its smoke/full launch.")
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("benchmark", choices=tuple(COUNTS))
    args = parser.parse_args()
    folder = ROOT / "reports/p1b_rgbt_evaluation_r1_20261104" / args.branch / args.benchmark
    assert not folder.exists()
    original = json.loads((ROOT / "reports/p1a_evaluation_r1_20261102" /
                           args.branch / "draft.json").read_bytes())
    raw = (ROOT / "reports/p1b_epoch005_evaluation_audit_r1_20261104" /
           args.branch / "audit.json").read_bytes()
    dependencies = json.loads((ROOT / "reports/p1b_evaluation_dependencies_r3_20261104" /
                               args.branch / "audit.json").read_bytes())
    draft = prepare(original, json.loads(raw), hashlib.sha256(raw).hexdigest(),
                    dependencies, args.benchmark)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "draft.json").write_text(json.dumps(draft, indent=2) + "\n")
    print(json.dumps(dict(branch=args.branch, benchmark=args.benchmark,
                          root=draft["root"], gpu=draft["smoke"]["gpu"],
                          launch_allowed=False)))


if __name__ == "__main__":
    main()
