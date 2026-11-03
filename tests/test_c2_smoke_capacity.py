"""Check the exact remote operator's capacity and serial predecessor gates."""
import ast
import json
from pathlib import Path

import pytest


def remote_gates():
    source = Path(__file__).resolve().parents[1] / "work/operate_c2_executor_smoke.py"
    tree = ast.parse(source.read_text())
    remote = next(node.value.s for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "REMOTE"
                          for target in node.targets))
    tree = ast.parse(remote)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(Path=Path, json=json)
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("free,passed", [(20000, False), (20479, False), (20480, True)])
def test_actual_gib_boundary(free, passed):
    gate = remote_gates()["check_gpu_capacity"]
    if passed:
        gate(dict(host="2025"), ["1", str(free), "250"])
    else:
        with pytest.raises(AssertionError):
            gate(dict(host="2025"), ["1", str(free), "250"])


def test_p25_power_limit_preserved():
    with pytest.raises(AssertionError):
        remote_gates()["check_gpu_capacity"](
            dict(host="2025"), ["1", "24000", "350"])


@pytest.mark.parametrize("state,exit_code,passed", [
    ("RUNNING", None, False), ("FAILED", 1, False), ("COMPLETE", 0, True)])
def test_serial_next_variant_requires_previous_native_exit(tmp_path, state, exit_code, passed):
    task = "smoke_zero_fixture"
    command = ["python", "train.py"]
    path = tmp_path / "runs/pipeline_tasks" / task / "status.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(
        task=task, command=command, state=state, exit_code=exit_code, pid=2147483647)))
    loaded = []

    def load_plan(path):
        loaded.append(path)
        return dict(gpu="1", command=command)

    gate = remote_gates()["check_predecessor"]
    args = (dict(variant="shared", gpu="1"), tmp_path, dict(shared=task), load_plan)
    if passed:
        gate(*args)
    else:
        with pytest.raises(AssertionError):
            gate(*args)
    assert loaded == [str(tmp_path / "reports" / task / "plan.json")]


def test_first_pair_has_no_predecessor():
    remote_gates()["check_predecessor"](dict(variant="zero"), None, {}, None)
