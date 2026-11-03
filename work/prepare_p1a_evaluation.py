"""Preserve missing evaluator sources and prepare disabled fixed-epoch controls."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/p1a_evaluation_r1_20261102"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


for branch in ("dropmoe", "sot"):
    folder = FOLDER / branch
    audit = json.loads((folder / "host_audit.json").read_bytes())
    host, root, parent = audit["host"], audit["root"], audit["parent"]
    missing = [name for name, row in audit["source_files"].items() if not row["present"]]
    assert len(missing) == 24
    for name in missing:
        assert audit["source_files"][name]["parent_present"], name
    copied = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         host, "tar", "-C", parent, "-cf", "-", "--"] + missing,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    payload = folder / "source_payload"
    payload.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(copied.stdout)) as archive:
        assert {item.name for item in archive.getmembers()} == set(missing)
        for item in archive.getmembers():
            assert item.isfile()
            raw = archive.extractfile(item).read()
            assert sha(raw) == audit["source_files"][item.name]["parent_sha256"]
            path = payload / item.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    sources = {name: row["sha256"] if row["present"] else row["parent_sha256"]
               for name, row in audit["source_files"].items()}
    assert len(sources) == 49
    inputs = audit["original_inputs"]
    collection = json.loads((ROOT / ("reports/p1b_%s_full_collection_r1_20261102/s0/plan.json" % branch)).read_bytes())
    run_parent = PurePosixPath(collection["run"]).parent.parent / "p1a_eval_runs"
    assert audit["p1a_native"]["state"] == "COMPLETE"
    assert audit["p1a_complete"]["epochs"] == 20
    review_path = root + "/reports/P1A_EPOCH20_" + branch.upper() + "_EVALUATION_SOURCE_REVIEW_R1_20261102.json"
    python = parent + "/conda-envs/dual_rgbt/bin/python"
    checkpoint, factory = inputs["executor"], inputs["factory"]
    suffix = "%s_p%s_r1_20261102" % (branch, host[-2:])

    def job(benchmark, gpu, smoke):
        key = "lasher_train_smoke" if smoke else benchmark.lower() + "_test"
        manifest = audit["manifests"][key]
        assert manifest["present"]
        task = ("smoke_p1a_e020_" if smoke else "eval_p1a_e020_" + benchmark.lower() + "_") + suffix
        run = str(run_parent / task)
        command = [python, root + "/scripts/eval_e0_finetuned.py",
                   "--checkpoint", checkpoint["path"], "--checkpoint-sha256", checkpoint["sha256"],
                   "--factory", factory["path"], "--factory-sha256", factory["sha256"],
                   "--manifest", manifest["path"], "--manifest-sha256", manifest["sha256"],
                   "--source-review", review_path, "--source-review-sha256", "PENDING_REVIEW_SHA256",
                   "--task", task, "--run", run, "--cost-weight", "0.05",
                   "--smoke-frames", "11" if smoke else "0"]
        return dict(task=task, run=run, benchmark=benchmark, gpu=gpu,
                    manifest=manifest, command=command,
                    sequences=1 if smoke else manifest["sequence_count"],
                    frames=11 if smoke else manifest["frame_count"],
                    smoke=smoke, exploratory_test=not smoke, test_eligible=False,
                    score_output=root + "/reports/" + task + "_SCORES.json")

    draft = dict(
        stage="P1A_epoch20_fixed_control_evaluation", host=host, root=root,
        branch=branch, launch_allowed=False, epoch=20, cost_weight=.05,
        review=dict(path=review_path, sha256=None),
        checkpoint=checkpoint, factory=factory, source_sha256=sources,
        original98_inputs=inputs, p1a_native=audit["p1a_native"],
        p1a_complete=audit["p1a_complete"], toolkit=audit["toolkit"],
        protocol=audit["protocol"],
        additive_source_files=missing,
        smoke=job("LasHeR", "0", True),
        jobs=[job(name, str(index), False) for index, name in enumerate(("LasHeR", "RGBT210", "RGBT234"))],
        exploratory_test=True, test_eligible=False,
        requires_actual_train_smoke_before_full=True,
        scope="Fixed recipe-final P1A epoch20 executor-only control before P1B refresh. "
              "Not selected on TEST; do not use its TEST results to alter the preplanned "
              "P1B training. Report all three full benchmarks descriptively. "
              "Existing eight TRAIN collectors and original98 sources remain unchanged.")
    assert not (folder / "draft.json").exists()
    (folder / "draft.json").write_text(json.dumps(draft, indent=2) + "\n")
    print(json.dumps(dict(branch=branch, draft_sha256=sha((folder / "draft.json").read_bytes()),
                          source_entries=len(sources), additive_files=len(missing),
                          jobs=len(draft["jobs"]), checkpoint=checkpoint)), flush=True)
