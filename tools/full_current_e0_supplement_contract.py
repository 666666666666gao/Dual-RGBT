"""Admission and complete TRAIN verification for current-E0 supplements."""
import json
from pathlib import Path
import socket

import numpy as np

from current_e0_supplement_contract import (
    ACCEPTANCE_SHA, PARENT_SHA, sha, verify_arrays, verify_run as verify_smoke)


STAGE = "M2_current_E0_full_evidence_supplement"
SMOKE_PLAN_SHA = "94f4bd85a45faf49f6b10eefe1534f38583885fc8441dc65ac7871010a557113"
SMOKE_RECEIPT_SHA = "87321c5b524bd20591cd931300a91b98038303ecd86a1acad11b37020ff9c240"
SMOKE_NATIVE_SHA = "d835fb2b87bc1859f567987a8a61456c7fb2f0dad86f1bb939ad1c4ee62f82df"


def partitions(plan, collection, sequences):
    names = plan["sequence_names"]
    assert names == collection["sequence_names"]
    assert len(names) == len(set(names)) == 882 and set(names) == set(sequences)
    assert plan["fit_sequences"] == collection["fit_sequences"]
    assert plan["holdout_sequences"] == collection["holdout_sequences"]
    fit, held = set(plan["fit_sequences"]), set(plan["holdout_sequences"])
    assert len(fit) == 706 and len(held) == 176 and fit.isdisjoint(held)
    assert fit | held == set(names)
    assert plan["shards"] == len(plan["jobs"]) == 8
    for shard, job in enumerate(plan["jobs"]):
        assert job["shard"] == shard
        assert job["sequence_names"] == names[shard::8]
        assert job["assigned_gpu"] == str(shard % 2)
    assert len({job["task"] for job in plan["jobs"]}) == 8
    assert len({job["run"] for job in plan["jobs"]}) == 8
    assert sum(len(sequences[n]["rgb"]) for n in names) == plan["expected_frames"] == 459435
    assert sum(len([f for f in sequences[n]["valid_frames"] if 0 < f < len(
        sequences[n]["rgb"])][::25]) for n in names) == plan["expected_candidate_sets"] == 18711


def admission(root, plan):
    assert plan["stage"] == STAGE and plan["phase"] == "full"
    assert plan["target_host"] == "2026" and plan["target_root"] == str(root)
    assert plan["target_hostname"] == socket.gethostname()
    assert plan["split"] == "train" and plan["test_eligible"] is False
    assert plan["training_allowed"] is False and plan["cost_weight"] == 0.
    assert plan["launch_allowed"] is True
    assert plan["checkpoint_sha256"] == PARENT_SHA
    assert plan["acceptance_sha256"] == ACCEPTANCE_SHA
    for name, expected in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        assert sha(root / name) == expected, name
    assert plan["input_sha256"][plan["checkpoint"]] == PARENT_SHA
    assert sha(root / plan["acceptance"]) == ACCEPTANCE_SHA
    accepted = json.loads((root / plan["acceptance"]).read_bytes())
    assert accepted["split"] == "train" and accepted["test_eligible"] is False
    assert sha(root / accepted["collection_plan"]) == accepted["collection_plan_sha256"]
    collection = json.loads((root / accepted["collection_plan"]).read_bytes())
    assert collection["checkpoint_sha256"] == PARENT_SHA
    assert collection["manifest"] == plan["manifest"]
    assert collection["input_sha256"][plan["manifest"]] == plan["input_sha256"][plan["manifest"]]
    manifest = json.loads((root / plan["manifest"]).read_bytes())
    sequences = {row["name"]: row for row in manifest["train"]}
    assert set(sequences).isdisjoint(
        {row["name"] for row in manifest["dev"]} | set(manifest["test_sequences"]))
    partitions(plan, collection, sequences)
    caches = {Path(name).stem: (name, expected)
              for name, expected in accepted["inputs_sha256"].items() if name.endswith(".npz")}
    assert len(caches) == 882 and set(caches) == set(sequences)
    assert plan["cache_sha256"] == {name: expected for name, expected in caches.values()}
    for name, expected in plan["cache_sha256"].items():
        assert sha(root / name) == expected, name
    review_path = root / plan["source_review"]
    assert sha(review_path) == plan["source_review_sha256"]
    review = json.loads(review_path.read_bytes())
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    for key in ("stage", "target_host", "target_root", "source_sha256"):
        assert review[key] == plan[key], key
    draft_path = root / plan["reviewed_draft"]
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(draft_path)
    draft = json.loads(draft_path.read_bytes())
    assert draft["launch_allowed"] is False
    omit = {"launch_allowed", "source_review_sha256"}
    assert {k: v for k, v in plan.items() if k not in omit} == {
        k: v for k, v in draft.items() if k not in omit}
    assert sha(root / plan["smoke_plan"]) == SMOKE_PLAN_SHA
    smoke_plan = json.loads((root / plan["smoke_plan"]).read_bytes())
    assert plan["smoke_run"] == smoke_plan["run"]
    assert sha(root / plan["smoke_acceptance"]) == plan["smoke_acceptance_sha256"]
    smoke = verify_smoke(root, root / plan["smoke_run"])
    assert smoke["receipt_sha256"] == SMOKE_RECEIPT_SHA
    assert smoke["native_sha256"] == SMOKE_NATIVE_SHA
    assert json.loads((root / plan["smoke_acceptance"]).read_bytes()) == smoke
    for name, expected in smoke_plan["source_sha256"].items():
        assert plan["source_sha256"][name] == expected, name
    return sequences, caches


def native_command(root, relative, shard):
    return [str(root / "conda-envs/dual_rgbt/bin/python"),
            str(root / "scripts/collect_e0_current_full.py"),
            "--plan", relative, "--shard", str(shard)]


def verify_native(root, plan, native, relative, shard, state):
    job = plan["jobs"][shard]
    assert job["shard"] == shard
    assert native["state"] == state and native["task"] == job["task"]
    assert native["hostname"] == plan["target_hostname"] == socket.gethostname()
    assert native["physical_gpu"] == job["assigned_gpu"]
    command = native_command(root, relative, shard)
    assert native["command"] == command
    return command


def verify_shard(root, plan, relative, shard, sequences, caches):
    job = plan["jobs"][shard]
    directory = root / job["run"]
    config = json.loads((directory / "config.json").read_bytes())
    receipt = json.loads((directory / "COMPLETE.json").read_bytes())
    native_path = root / "runs/pipeline_tasks" / job["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    command = verify_native(root, plan, native, relative, shard, "COMPLETE")
    assert native["exit_code"] == 0
    assert config["plan"] == plan and config["plan_path"] == relative
    assert config["shard"] == shard and config["run"] == job["run"]
    assert config["task"] == receipt["task"] == job["task"]
    assert config["native_command"] == command
    assert config["plan_sha256"] == receipt["plan_sha256"] == sha(root / relative)
    assert receipt["status"] == "CURRENT_E0_FULL_SUPPLEMENT_SHARD_COMPLETE"
    assert receipt["split"] == "train" and receipt["test_eligible"] is False
    for key in ("all_online_states_bit_equal", "original_predictions_bit_equal",
                "model_unchanged", "predict_restored", "no_gradients"):
        assert receipt[key] is True
    assert receipt["optimizer_updates"] == receipt["backward_calls"] == 0
    assert receipt["frozen_state_tensors"] == 265
    assert receipt["anchor_encoder_calls"] == receipt["candidate_encoder_calls"] == 0
    expected = {p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob("*")
                if p.is_file() and p.name != "COMPLETE.json"}
    assert receipt["artifacts_sha256"] == expected
    assert set(expected) == {"config.json", "progress.json", "current.json"} | {
        name + ".npz" for name in job["sequence_names"]}
    rows = []
    for name in job["sequence_names"]:
        parent, _ = caches[name]
        with np.load(directory / (name + ".npz"), allow_pickle=False) as values, \
                np.load(root / parent, allow_pickle=False) as original:
            rows.append(verify_arrays(values, sequences[name], original, len(sequences[name]["rgb"])))
    assert json.loads((directory / "progress.json").read_bytes()) == rows
    assert receipt["sequences"] == len(rows)
    assert receipt["frames"] == sum(row["frames"] for row in rows)
    assert receipt["candidate_sets"] == sum(row["candidate_sets"] for row in rows)
    assert receipt["supplement_encoder_calls"] == 2 * receipt["candidate_sets"]
    used = {str((directory / name).relative_to(root)): expected[name] for name in expected}
    used[native_path.relative_to(root).as_posix()] = sha(native_path)
    used[(directory / "COMPLETE.json").relative_to(root).as_posix()] = sha(directory / "COMPLETE.json")
    return rows, used


def verify_collection(root, relative):
    plan = json.loads((root / relative).read_bytes())
    sequences, caches = admission(root, plan)
    rows, used = [], {relative: sha(root / relative)}
    for shard in range(8):
        shard_rows, shard_used = verify_shard(root, plan, relative, shard, sequences, caches)
        rows.extend(shard_rows)
        used.update(shard_used)
    assert len(rows) == 882 and {r["sequence"] for r in rows} == set(sequences)
    assert sum(r["frames"] for r in rows) == 459435
    assert sum(r["candidate_sets"] for r in rows) == 18711
    return dict(status="FULL_CURRENT_E0_TRAIN_SUPPLEMENT_ACCEPTED", split="train",
                test_eligible=False, scientific_result_audit_accepted=False,
                plan=relative, plan_sha256=sha(root / relative),
                parent_acceptance_sha256=ACCEPTANCE_SHA,
                sequences=882, frames=459435, candidate_sets=18711,
                fit_sequences=plan["fit_sequences"], holdout_sequences=plan["holdout_sequences"],
                trajectories=rows, artifacts_sha256=used, training_allowed=False,
                scope="Complete original TRAIN frame joins/current-E0 evidence. "
                      "Receipts attest runtime parity; verifier does not rerun models. "
                      "No training, online improvement or independent validation.")
