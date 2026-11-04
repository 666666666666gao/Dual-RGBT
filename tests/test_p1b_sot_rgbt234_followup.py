"""SOT234 cannot deploy or launch before its exact SOT210 predecessor completes."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))
SPEC = importlib.util.spec_from_file_location(
    "sot234_followup", ROOT / "work/operate_p1b_sot_rgbt234_followup.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(tmp_path):
    def save(path, data):
        path.write_text(json.dumps(data))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    run = tmp_path / "run"
    run.mkdir()
    job = dict(task="preceding210", run=str(run), gpu="1",
               command=["python", "eval.py", "PENDING_REVIEW_SHA256"])
    review_path = tmp_path / "review.json"
    review_sha = save(review_path, dict(status="SOURCE_PASS"))
    preceding = dict(host="2026", branch="sot", benchmark="rgbt210", jobs=[job])
    draft_path = tmp_path / "draft.json"
    draft_sha = save(draft_path, preceding)
    native_path = tmp_path / "native.json"
    save(native_path, dict(
        task=job["task"], physical_gpu="1", command=["python", "eval.py", review_sha],
        state="COMPLETE", exit_code=0, pid=999999999))
    save(run / "config.json", dict(
        task=job["task"], physical_gpu="1", native_command=["python", "eval.py", review_sha],
        source_review_sha256=review_sha, split_sequences=210, split_frames=104706))
    save(run / "COMPLETE.json", dict(
        task=job["task"], sequences=210, frames=104706, state_unchanged=True,
        smoke_only=False, optimizer_updates=0, backward_calls=0))
    request = dict(action="full", draft=dict(
        host="2026", branch="sot", smoke=dict(gpu="1"),
        predecessor=dict(
            draft_path=str(draft_path), draft_sha256=draft_sha,
            review_path=str(review_path), review_sha256=review_sha,
            job=job, native_path=str(native_path))))
    return request, native_path, run


@pytest.mark.parametrize("action", ["deploy", "smoke", "full"])
@pytest.mark.parametrize("terminal", [True, False])
def test_predecessor_terminal_gate(tmp_path, monkeypatch, action, terminal):
    request, native_path, _ = fixture(tmp_path)
    request["action"] = action
    if not terminal:
        native = json.loads(native_path.read_bytes())
        native.update(state="RUNNING", exit_code=None)
        native_path.write_text(json.dumps(native))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    if terminal:
        exec(MODULE.GATE, {})
    else:
        with pytest.raises(AssertionError):
            exec(MODULE.GATE, {})


@pytest.mark.parametrize("mutation", ["command", "frames", "draft_hash"])
def test_substituted_predecessor_evidence_rejected(tmp_path, monkeypatch, mutation):
    request, native_path, run = fixture(tmp_path)
    if mutation == "draft_hash":
        request["draft"]["predecessor"]["draft_sha256"] = "wrong"
    else:
        path = native_path if mutation == "command" else run / "COMPLETE.json"
        data = json.loads(path.read_bytes())
        data[mutation] = ["wrong"] if mutation == "command" else 104705
        path.write_text(json.dumps(data))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    with pytest.raises(AssertionError):
        exec(MODULE.GATE, {})


def test_r2_plan_has_fresh_identity_and_exact_predecessor():
    folder = ROOT / "reports/p1b_rgbt_evaluation_r1_20261104/sot"
    old = json.loads((folder / "rgbt234/draft.json").read_bytes())
    previous = json.loads((folder / "rgbt210/draft.json").read_bytes())
    untouched = copy.deepcopy(old)
    result = MODULE.prepare(old, previous, "draft_hash", "review_hash")
    assert old == untouched and result["launch_allowed"] is False
    assert result["checkpoint"] == old["checkpoint"]
    assert result["source_sha256"] == old["source_sha256"]
    assert "_r2_" in result["root"]
    assert result["predecessor"]["job"] == previous["jobs"][0]
    for job in [result["smoke"]] + result["jobs"]:
        assert "_r2_" in job["task"] and job["gpu"] == "1"
        assert job["command"][1].startswith(result["root"] + "/")
        assert job["command"][job["command"].index("--source-review") + 1] == result["review"]["path"]
