"""The RGBT scorer retains existing-score provenance and exact dataset gates."""
import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rgbt_scorer", ROOT / "work/score_p1b_rgbt_completed.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BASE_SPEC = importlib.util.spec_from_file_location(
    "lasher_scorer", ROOT / "work/score_p1b_completed.py")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)


def test_preserves_full_current_artifact_and_source_provenance_validation():
    def function_tree(source):
        return ast.dump(next(
            node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "validate_score_provenance"))
    assert function_tree(MODULE.REMOTE) == function_tree(BASE.REMOTE)
    compile(MODULE.REMOTE, "rgbt_remote", "exec")


@pytest.mark.parametrize("benchmark,count,frames", [
    ("rgbt210", 210, 104706), ("rgbt234", 234, 116650),
])
@pytest.mark.parametrize("wrong", [False, True])
def test_wrong_dataset_counts_rejected_before_scoring(benchmark, count, frames, wrong):
    tree = ast.parse(MODULE.REMOTE)
    start = next(index for index, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "expected"
                         for target in node.targets))
    scope = dict(d=dict(benchmark=benchmark),
                 job=dict(sequences=count, frames=frames + int(wrong)))
    code = compile(ast.Module(body=tree.body[start:start + 2], type_ignores=[]),
                   "exact_counts", "exec")
    if wrong:
        with pytest.raises(AssertionError):
            exec(code, scope)
    else:
        exec(code, scope)
        assert scope["expected"] == (count, frames)
