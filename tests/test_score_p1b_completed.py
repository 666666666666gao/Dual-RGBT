"""Existing scores must remain bound to the exact current run artifacts."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p1b_scorer", ROOT / "work/score_p1b_completed.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def contract(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "prediction.txt").write_text("1,2,3,4\n")
    native = tmp_path / "native.json"
    native.write_text('{"state":"COMPLETE"}\n')
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"toolkit_code_sha256":{"metric.py":"toolkit"}}\n')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sources = {name: name for name in (
        "tools/score_e0_finetuned.py", "tools/joint_candidate_results.py",
        "scripts/score_tracking.py", "scripts/joint_evaluation_contract.py",
        "scripts/benchmark_inputs.py")}
    d = dict(source_sha256=sources,
             protocol=dict(path=str(protocol), sha256=sha(protocol)))
    job = dict(manifest=dict(sha256="manifest"))
    score = dict(
        source_sha256=sources, source_review_sha256="review",
        manifest_sha256="manifest", toolkit_protocol_sha256=sha(protocol),
        toolkit_code_sha256={"metric.py": "toolkit"},
        scoring_script_sha256=sources["tools/score_e0_finetuned.py"],
        metric_implementation_sha256={name: value for name, value in sources.items()
                                     if name != "tools/score_e0_finetuned.py"},
        native_status_sha256=sha(native),
        artifacts_sha256={"prediction.txt": sha(run / "prediction.txt")})
    tree = ast.parse(MODULE.REMOTE)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "validate_score_provenance")
    namespace = dict(d=d, job=job, request=dict(review_sha256="review"),
                     native_path=native, run=run, sha=sha,
                     read=lambda path: json.loads(Path(path).read_bytes()))
    exec(compile(ast.Module(body=[function], type_ignores=[]), "contract", "exec"), namespace)
    return namespace["validate_score_provenance"], score, run, native


def test_unchanged_score_provenance_passes(tmp_path):
    validate, score, _, _ = contract(tmp_path)
    validate(score)


@pytest.mark.parametrize("target", ["prediction", "native", "extra_file"])
def test_changed_current_evidence_rejected(tmp_path, target):
    validate, score, run, native = contract(tmp_path)
    path = native if target == "native" else run / (
        "prediction.txt" if target == "prediction" else "extra.txt")
    path.write_text("changed")
    with pytest.raises(AssertionError):
        validate(score)


@pytest.mark.parametrize("key", [
    "source_sha256", "source_review_sha256", "manifest_sha256",
    "toolkit_protocol_sha256", "toolkit_code_sha256", "scoring_script_sha256",
    "metric_implementation_sha256", "native_status_sha256", "artifacts_sha256",
])
def test_stale_saved_provenance_rejected(tmp_path, key):
    validate, score, _, _ = contract(tmp_path)
    score = copy.deepcopy(score)
    score[key] = "stale"
    with pytest.raises(AssertionError):
        validate(score)
