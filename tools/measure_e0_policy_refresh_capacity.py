"""Measure two real FIT updates on an admitted completed TRAIN prefix."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def binding(spec):
    payload = {key: value for key, value in spec.items()
               if key not in ("review", "launch_allowed")}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    assert spec["stage"] == "P1B_direct_E0_capacity"
    assert spec["launch_allowed"] is True
    assert spec["batch_size"] == 128 and spec["updates"] == 2
    assert spec["prefix_sequences"] == 8 and spec["minimum_free_mib"] == 20000
    assert os.environ["CUDA_VISIBLE_DEVICES"] == spec["gpu"]
    assert Path.cwd().resolve() == Path(spec["root"]).resolve()
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert sha(__file__) == spec["tool_sha256"]
    assert sha(spec["review"]["path"]) == spec["review"]["sha256"]
    review = json.loads(Path(spec["review"]["path"]).read_text())
    assert review["stage"] == spec["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"]["tools/measure_e0_policy_refresh_capacity.py"] == spec["tool_sha256"]
    assert binding(spec) in review["admitted_spec_bindings"]
    output = Path(spec["output"])
    assert not output.exists()
    plan_path = Path(spec["collection_plan"]["path"])
    assert sha(plan_path) == spec["collection_plan"]["sha256"]
    sys.path.insert(0, spec["root"])
    import torch
    from torch.utils.data._utils.collate import default_collate
    from dual_rgbt.e0_policy_refresh import (
        POLICY_PREFIXES, configure_policy_refresh, load_completed_executor,
        load_plan, policy_refresh_loss, rebuild_state, train_sequences)
    from lib.test.utils.hann import hann2d

    plan = load_plan(plan_path, "collect")
    assert plan["root"] == spec["root"] and plan["host"] == spec["host"]
    assert plan["branch"] == spec["branch"] and not plan["smoke"]
    assert plan["round"] == 1 and plan["shard"] == 0 and plan["shards"] == 4
    progress = json.loads((Path(plan["run"]) / "progress.json").read_text())
    assert progress["sequences"] >= spec["prefix_sequences"]
    sequences = train_sequences(plan["inputs"]["manifest"]["path"])[::4][:8]
    samples, artifacts = [], []
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for sequence in sequences:
        path = Path(plan["run"]) / (sequence["name"] + ".pth")
        raw = path.read_bytes()
        cache = torch.load(io.BytesIO(raw), map_location="cpu")
        assert cache["sequence"] == sequence["name"]
        assert cache["plan_sha256"] == spec["collection_plan"]["sha256"]
        expected = [frame for frame in sequence["valid_frames"]
                    if frame > 0 and frame % plan["label_stride"] == 0]
        assert [record["frame"] for record in cache["records"]] == expected
        artifacts.append(dict(sequence=sequence["name"],
                              sha256=hashlib.sha256(raw).hexdigest()))
        for record in cache["records"]:
            if len(samples) < spec["batch_size"]:
                samples.append(rebuild_state(sequence, record))
    assert len(samples) == spec["batch_size"]
    batch = default_collate(samples)
    del samples
    assert all(torch.isfinite(tensor).all() for tensor in batch)
    free_mib = int(subprocess.check_output([
        "nvidia-smi", "--id=" + spec["gpu"], "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], text=True).strip())
    assert free_mib >= spec["minimum_free_mib"]
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits"], text=True).strip()
    model = load_completed_executor(plan)
    before = {key: tensor.cpu().clone() for key, tensor in model.state_dict().items()}
    groups = configure_policy_refresh(model)
    parameters = [parameter for group in groups for parameter in group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    batch = [tensor.cuda() for tensor in batch]
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    updates = []
    for step in range(spec["updates"]):
        started = time.perf_counter()
        result = policy_refresh_loss(model, *batch, window)
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        for module in (model.preview, model.quality, model.gain):
            assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                       for parameter in module.parameters())
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert torch.isfinite(norm)
        optimizer.step()
        torch.cuda.synchronize()
        updates.append(dict(step=step + 1, seconds=time.perf_counter() - started,
                            loss=float(result["loss"]), gradient_norm=float(norm)))
    after = model.state_dict()
    assert all(torch.isfinite(tensor).all() for tensor in after.values())
    for key, tensor in before.items():
        if not key.startswith(POLICY_PREFIXES):
            assert torch.equal(tensor, after[key].cpu()), key
    output.write_text(json.dumps(dict(
        stage=spec["stage"], status="COMPLETE", spec_sha256=sha(args.spec),
        spec_binding_sha256=binding(spec), tool_sha256=sha(__file__),
        branch=plan["branch"], batch_size=len(batch[0]), updates=updates,
        learning_rates=[group["lr"] for group in groups], weight_decay=1e-4,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
        free_mib_before_cuda=free_mib, artifacts=artifacts,
        executor_preserved=True, checkpoint_written=False,
        full_collection_acceptance=False, full_fit_admitted=False,
        benchmark_evidence=False, compute_processes_before_cuda=processes,
        scope="Two updates on one repeated real TRAIN batch; GPU kernel capacity only, not full loader throughput or convergence."
    ), indent=2) + "\n")
    print(output.read_text(), flush=True)


if __name__ == "__main__":
    main()
