"""Exercise retrieval with the real C2 scorer over synthetic CPU artifacts."""
import base64
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import types

import pytest

from test_score_c2_policy_completed import case, scoring, seal, write

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "c2_remote_score", ROOT / "work/score_c2_policy_completed_remote.py")
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)


@pytest.fixture
def ready(case, monkeypatch):
    args, plan, root, run, native_path, _ = case
    monkeypatch.setattr(remote, "ROOT", root)
    monkeypatch.setitem(remote.CPU, "2025", sys.executable)
    helper = root / remote.ENTRY
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_bytes((ROOT / remote.ENTRY).read_bytes())
    plan["sources"][remote.ENTRY] = remote.sha(helper)
    plan.update(scoring_interpreter=sys.executable, score_output=args.output, plan_path=args.plan)
    config = remote.read(run / "config.json")
    config["source_sha256"] = plan["sources"]
    write(run / "config.json", config)
    seal(root, plan, Path(args.plan))
    native = remote.read(native_path)
    native["pid"] = 999999999
    write(native_path, native)
    plan, request = remote.prepare_request(Path(args.plan), Path(plan["review"]["path"]))
    calls = []

    def execute(command, **kwargs):
        assert kwargs["cwd"] == root
        expected = [
            sys.executable, "-B", str(root / remote.SCORER),
            "--plan", args.plan, "--run", args.run,
            "--toolkit", args.toolkit, "--protocol", args.protocol,
            "--protocol-sha256", args.protocol_sha256, "--output", args.output,
        ]
        assert command == expected
        calls.append(command)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            scoring.score(args)
        return types.SimpleNamespace(returncode=0, stdout=output.getvalue().encode())

    monkeypatch.setattr(remote.subprocess, "run", execute)
    return args, plan, request, root, run, native_path, calls


def test_full_score_and_exact_archive(ready):
    args, plan, request, root, _, _, calls = ready
    response = remote.remote_score(request)
    assert len(calls) == 1 and response["scoring_reused"] is False
    destination = root / "archive"
    audit = remote.archive_result(response, request, destination)
    assert audit["task"] == plan["task"]
    assert set(path.name for path in destination.iterdir()) == remote.NAMES | {"audit.json"}
    for name, digest in audit["archived_sha256"].items():
        assert remote.sha(destination / name) == digest
    assert remote.sha(destination / "score.json") == remote.sha(args.output)


def test_existing_score_validation_never_reexecutes_metrics_or_rewrites(ready, monkeypatch):
    args, _, request, root, _, _, calls = ready
    first = remote.remote_score(request)
    output = Path(args.output)
    log = output.with_suffix(".scoring.log")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in (output, output.with_suffix(".npz"), log)}
    monkeypatch.setattr(scoring, "metric_curves", lambda *args: pytest.fail("no rescoring"))
    second = remote.remote_score(request)
    assert second["scoring_reused"] is True and len(calls) == 2
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    assert first["files"] == second["files"]
    remote.archive_result(second, request, root / "recovered_archive")


@pytest.mark.parametrize("key,value", [
    ("state", "RUNNING"), ("exit_code", 1), ("task", "wrong"),
    ("command", ["wrong"]), ("physical_gpu", "1"),
])
def test_bad_native_never_scores(ready, key, value):
    args, _, request, _, _, native_path, calls = ready
    native = remote.read(native_path)
    native[key] = value
    write(native_path, native)
    with pytest.raises(AssertionError):
        remote.remote_score(request)
    assert not calls and not Path(args.output).exists()


def test_live_wrapper_never_scores(ready, monkeypatch):
    args, _, request, _, _, native_path, calls = ready
    native = remote.read(native_path)
    native["pid"] = __import__("os").getpid()
    write(native_path, native)
    # /proc exists on the execution hosts; simulate that exact lookup on Windows.
    original = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: True if str(path).replace("\\", "/").startswith(
        "/proc/") else original(path))
    with pytest.raises(AssertionError):
        remote.remote_score(request)
    assert not calls and not Path(args.output).exists()


@pytest.mark.parametrize("partial", ["curves", "log"])
def test_preserves_partial_failure_before_scoring(ready, partial):
    args, _, request, _, _, _, calls = ready
    output = Path(args.output)
    path = output.with_suffix(".npz" if partial == "curves" else ".scoring.log")
    path.write_bytes(b"partial-failure-evidence")
    with pytest.raises(AssertionError):
        remote.remote_score(request)
    assert not calls and path.read_bytes() == b"partial-failure-evidence"


def test_failed_scorer_log_preserved_without_archive(ready, monkeypatch):
    args, _, request, _, _, _, _ = ready
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=1, stdout=b"actual scorer failure fixture"))
    with pytest.raises(AssertionError, match="actual scorer failure"):
        remote.remote_score(request)
    assert Path(args.output).with_suffix(".scoring.log").read_bytes() == b"actual scorer failure fixture"
    with pytest.raises(AssertionError):
        remote.remote_score(request)


@pytest.mark.parametrize("change", ["plan", "review", "source"])
def test_changed_pinned_bytes_never_scores(ready, change):
    _, plan, request, root, _, _, calls = ready
    target = {"plan": Path(plan["plan_path"]), "review": Path(plan["review"]["path"]),
              "source": root / remote.ENTRY}[change]
    target.write_bytes(b"changed")
    with pytest.raises(AssertionError):
        remote.remote_score(request)
    assert not calls


@pytest.mark.parametrize("change", ["cpu", "smoke", "output", "review_binding"])
def test_invalid_reviewed_plan_rejected(ready, change):
    _, plan, _, _, run, _, calls = ready
    plan = copy.deepcopy(plan)
    review = remote.read(plan["review"]["path"])
    if change == "cpu":
        plan["scoring_interpreter"] = "/nn/python"
    elif change == "smoke":
        plan.update(smoke=True, smoke_frames=11)
    elif change == "output":
        plan["score_output"] = str(run / "score.json")
    else:
        review["admitted_plan_bindings"] = []
    with pytest.raises(AssertionError):
        remote.authenticate(plan, review)
    assert not calls


@pytest.mark.parametrize("corruption", ["digest", "filename", "plan", "review", "task"])
def test_corrupt_transfer_rejected_before_local_directory(ready, corruption):
    _, _, request, root, _, _, _ = ready
    response = remote.remote_score(request)
    if corruption == "digest":
        response["files"]["score.json"]["sha256"] = "wrong"
    elif corruption == "filename":
        response["files"]["../escape.json"] = response["files"].pop("score.json")
    elif corruption in ("plan", "review"):
        name = corruption + ".json"
        raw = b"{}"
        response["files"][name] = dict(raw=base64.b64encode(raw).decode(),
                                       sha256=remote.hashlib.sha256(raw).hexdigest())
    else:
        response["task"] = "wrong"
    destination = root / "archive"
    with pytest.raises(AssertionError):
        remote.archive_result(response, request, destination)
    assert not destination.exists()


def test_existing_local_archive_never_overwritten(ready):
    _, _, request, root, _, _, _ = ready
    response = remote.remote_score(request)
    destination = root / "archive"
    destination.mkdir()
    marker = destination / "failure"
    marker.write_bytes(b"preserve")
    with pytest.raises(AssertionError):
        remote.archive_result(response, request, destination)
    assert marker.read_bytes() == b"preserve"
