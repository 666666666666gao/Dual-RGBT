"""Prepare disabled C2 TRAIN smoke plans without accessing a server."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
NEW_SOURCES = (
    "dual_rgbt/c2_executor.py", "dual_rgbt/c2_data.py",
    "dual_rgbt/route_lora.py", "dual_rgbt/missing_modality.py",
    "scripts/train_c2_executor.py")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    args = parser.parse_args()
    branch = args.branch
    collection = json.loads((ROOT / (
        "reports/p1b_%s_full_collection_r1_20261102/s0/plan.json" % branch)).read_bytes())
    parent = "/home/gaob/Dual-RGBT" if branch == "dropmoe" else "/data2/gb/Dual-RGBT"
    storage = "/data2/gaob/Dual-RGBT" if branch == "dropmoe" else parent
    host = collection["host"]
    root = storage + "/c2_training_%s_r1_20261103" % branch
    folder = ROOT / "reports/c2_training_smoke_r1_20261103" / branch
    assert not folder.exists()
    cpu_raw = (ROOT / "reports/c2_executor_cpu_r2_20261103/result.json").read_bytes()
    cpu = json.loads(cpu_raw)
    assert cpu["status"] == "PASS" and cpu["cuda_initialized"] is False
    assert len(cpu["cpu_optimizer_updates"]) == 2
    sources = dict(collection["sources"])
    assert len(sources) == 98
    for name in NEW_SOURCES:
        assert name not in sources
        sources[name] = digest((ROOT / name).read_bytes())
    review = root + "/reports/P2_C2_%s_SMOKE_ADMISSION_R1_20261103.json" % branch.upper()
    inputs = {key: collection["inputs"][key] for key in ("factory", "executor", "manifest")}
    inputs["cpu_result"] = dict(path=root + "/evidence/c2_cpu_result_r2.json",
                                sha256=digest(cpu_raw))
    folder.mkdir(parents=True)
    bindings = []
    for variant, gpu in (("zero", "1"), ("film", "2"), ("shared", "1"), ("routed", "2")):
        task = "smoke_c2_%s_%s_r1_p%s_20261103" % (branch, variant, host[-2:])
        remote_plan = root + "/reports/" + task + "/plan.json"
        plan = dict(
            stage="P2_C2_executor_training", host=host, root=root,
            source_snapshot=collection["root"], source_collection_plan=(
                "reports/p1b_%s_full_collection_r1_20261102/s0/plan.json" % branch),
            launch_allowed=False, smoke=True, variant=variant, rank=8,
            epochs=20, samples=20000, batch_size=8, workers=4, seed=2027,
            milestones=[10], backbone_lr=1e-5, head_lr=1e-4,
            weight_decay=1e-4, gradient_clip=.1, dev_enabled=False, test_enabled=False,
            task=task, gpu=gpu, run=storage + "/c2_runs/" + task,
            sources=sources, inputs=inputs,
            review=dict(path=review, sha256="PENDING_REVIEW_SHA256"),
            command=[parent + "/conda-envs/dual_rgbt/bin/python", "-B",
                     root + "/scripts/train_c2_executor.py", "--plan", remote_plan])
        binding = {key: value for key, value in plan.items()
                   if key not in ("review", "launch_allowed")}
        binding_sha = digest(json.dumps(
            binding, sort_keys=True, separators=(",", ":")).encode())
        destination = folder / variant
        destination.mkdir()
        raw = (json.dumps(plan, indent=2) + "\n").encode()
        (destination / "plan.json").write_bytes(raw)
        bindings.append(dict(variant=variant, gpu=gpu, task=task, binding_sha256=binding_sha,
                             draft_sha256=digest(raw),
                             path=str(PurePosixPath("reports/c2_training_smoke_r1_20261103") /
                                      branch / variant / "plan.json")))
    (folder / "bindings.json").write_text(json.dumps(dict(
        status="DISABLED_DRAFTS", branch=branch, host=host, root=root,
        source_count=len(sources), cpu_result_sha256=digest(cpu_raw),
        new_sources=list(NEW_SOURCES), review=review, plans=bindings,
        launch_allowed=False, no_remote_actions=True), indent=2) + "\n")
    print(json.dumps(dict(branch=branch, root=root, sources=len(sources), bindings=bindings)))


if __name__ == "__main__":
    main()
