"""Exercise the actual trainer admission without importing the NN runtime."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest


def load_functions(root):
    source = Path(__file__).resolve().parents[1] / "scripts/train_c2_executor.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("sha", "load_plan")]
    namespace = dict(Path=Path, hashlib=hashlib, json=json, ROOT=root,
                     VARIANTS=dict.fromkeys(("zero", "film", "shared", "routed")))
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


def write_json(path, value):
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


@pytest.fixture
def admitted(tmp_path):
    scope = load_functions(tmp_path)
    (tmp_path / "source.py").write_text("pass\n")
    result = write_json(tmp_path / "cpu.json", dict(
        status="PASS", stage="P2_C2_executor_cpu_check", cuda_initialized=False,
        strict_state_dict_reload=True, cpu_optimizer_updates=[{}, {}]))
    plan = dict(
        stage="P2_C2_executor_training", root=str(tmp_path), launch_allowed=True,
        smoke=True, variant="routed", rank=8, epochs=20, samples=20000,
        batch_size=8, workers=4, seed=2027, milestones=[10], backbone_lr=1e-5,
        head_lr=1e-4, weight_decay=1e-4, gradient_clip=.1,
        test_enabled=False, dev_enabled=False, inputs=dict(cpu_result=result),
        task="smoke_fixture", run=str(tmp_path / "run"),
        command=["python", str(tmp_path / "source.py"), "--plan", str(tmp_path / "plan.json")],
        sources={"source.py": scope["sha"](tmp_path / "source.py")})
    binding = hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items()
         if key not in ("launch_allowed", "review")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan["review"] = write_json(tmp_path / "review.json", dict(
        stage=plan["stage"], status="SOURCE_PASS", blocking_issues=0,
        sources=plan["sources"], admitted_plan_bindings=[binding]))
    return scope, plan, tmp_path / "plan.json"


def test_exact_concrete_smoke_admission(admitted):
    scope, plan, path = admitted
    write_json(path, plan)
    assert scope["load_plan"](path) == plan


@pytest.mark.parametrize("key,value", [
    ("launch_allowed", False), ("smoke", "true"), ("test_enabled", True),
    ("dev_enabled", True), ("rank", 16), ("batch_size", 16),
    ("epochs", 5), ("head_lr", 1e-3), ("seed", 2028),
    ("root", "/different/isolated/root"), ("variant", "other"),
])
def test_rejects_disabled_or_changed_contract(admitted, key, value):
    scope, plan, path = admitted
    plan[key] = value
    write_json(path, plan)
    with pytest.raises(AssertionError):
        scope["load_plan"](path)


def test_rejects_source_change(admitted):
    scope, plan, path = admitted
    (path.parent / "source.py").write_text("raise RuntimeError\n")
    write_json(path, plan)
    with pytest.raises(AssertionError):
        scope["load_plan"](path)


def test_source_only_review_does_not_admit_smoke(admitted):
    scope, plan, path = admitted
    review_path = Path(plan["review"]["path"])
    review = json.loads(review_path.read_bytes())
    review["admitted_plan_bindings"] = []
    plan["review"] = write_json(review_path, review)
    write_json(path, plan)
    with pytest.raises(AssertionError):
        scope["load_plan"](path)


def test_full_requires_real_smoke_receipts(admitted):
    scope, plan, path = admitted
    plan = copy.deepcopy(plan)
    plan["smoke"] = False
    write_json(path, plan)
    with pytest.raises(KeyError, match="smoke_receipt"):
        scope["load_plan"](path)


def full_fixture(admitted):
    scope, smoke, path = admitted
    for key in ("factory", "executor", "manifest"):
        smoke["inputs"][key] = write_json(path.parent / (key + ".json"), {})
    run = Path(smoke["run"])
    run.mkdir()
    checkpoint = run / "epoch_001.pth"
    checkpoint.write_bytes(b"fixture-checkpoint")
    config = write_json(path.parent / "smoke_config.json", smoke)
    receipt = write_json(path.parent / "smoke_receipt.json", dict(
        status="PASS", task=smoke["task"], updates=10,
        config_sha256=config["sha256"], route_counts=dict.fromkeys(map(str, range(10)), 1),
        strict_reload=True, policy_preserved=True,
        artifacts={checkpoint.name: scope["sha"](checkpoint)}))
    native = write_json(path.parent / "smoke_native.json", dict(
        state="COMPLETE", exit_code=0, task=smoke["task"], command=smoke["command"]))
    plan = copy.deepcopy(smoke)
    plan.update(smoke=False, task="full_fixture", run=str(path.parent / "full_run"))
    plan["inputs"].update(smoke_receipt=receipt, smoke_native=native, smoke_config=config)
    binding = hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items()
         if key not in ("launch_allowed", "review")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan["review"] = write_json(path.parent / "full_review.json", dict(
        stage=plan["stage"], status="SOURCE_PASS", blocking_issues=0,
        sources=plan["sources"], admitted_plan_bindings=[binding]))
    return scope, plan, path


def test_complete_bound_smoke_admits_full_contract(admitted):
    scope, plan, path = full_fixture(admitted)
    write_json(path, plan)
    assert scope["load_plan"](path) == plan


@pytest.mark.parametrize("field,value", [("state", "FAILED"), ("exit_code", 1)])
def test_full_rejects_failed_native_even_with_valid_file_hash(admitted, field, value):
    scope, plan, path = full_fixture(admitted)
    native_path = Path(plan["inputs"]["smoke_native"]["path"])
    native = json.loads(native_path.read_bytes())
    native[field] = value
    plan["inputs"]["smoke_native"] = write_json(native_path, native)
    write_json(path, plan)
    with pytest.raises(AssertionError):
        scope["load_plan"](path)


def test_full_rejects_changed_smoke_checkpoint(admitted):
    scope, plan, path = full_fixture(admitted)
    (path.parent / "run/epoch_001.pth").write_bytes(b"changed")
    write_json(path, plan)
    with pytest.raises(AssertionError):
        scope["load_plan"](path)


@pytest.mark.parametrize("key,value", [("route_counts", {"0": 1}), ("artifacts", {})])
def test_full_rejects_incomplete_smoke_witness(admitted, key, value):
    scope, plan, path = full_fixture(admitted)
    receipt_path = Path(plan["inputs"]["smoke_receipt"]["path"])
    receipt = json.loads(receipt_path.read_bytes())
    receipt[key] = value
    plan["inputs"]["smoke_receipt"] = write_json(receipt_path, receipt)
    write_json(path, plan)
    with pytest.raises(AssertionError):
        scope["load_plan"](path)
