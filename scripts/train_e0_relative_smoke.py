"""Five admitted real TRAIN prefix updates, with strict save/reload checks."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from current_e0_supplement_contract import sha
from dual_rgbt.e0_relative_identity import E0RelativeIdentity
from relative_e0_smoke_contract import (
    admission, load_prefix_data, native_command, verify_native)
from relative_e0_training import batch_values, forward, update_epoch


def run_smoke(root, plan, plan_path):
    smoke = admission(root, plan)
    native = json.loads((root / "runs/pipeline_tasks" / plan["task"] / "status.json").read_bytes())
    verify_native(root, plan, native, plan_path, "RUNNING")
    assert native["pid"] == os.getppid()
    assert native["command"] == [sys.executable] + sys.argv
    assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["assigned_gpu"]
    plan_sha = sha(root / plan_path)
    data = load_prefix_data(root, smoke)
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    head = E0RelativeIdentity().to(plan["device"])
    initial = torch.load(root / plan["initialization"], map_location="cpu")["net"]
    head.load_state_dict(initial, strict=True)
    before = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    assert all(torch.equal(value, initial[key]) for key, value in before.items())
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=plan["lr"], weight_decay=plan["weight_decay"])
    directory = root / plan["run"]
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(dict(
        plan=plan, plan_path=plan_path, plan_sha256=plan_sha,
        native_command=native_command(root, plan_path)), indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started, losses = time.time(), []
    for _ in range(5):
        result = update_epoch(head, optimizer, data, 6)
        assert result["optimizer_updates"] == 1
        losses.append(result["loss"])
    after = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    assert torch.equal(after["no_match"], before["no_match"])
    assert any(not torch.equal(value, before[key]) for key, value in after.items())
    torch.save(dict(net=after, plan_sha256=plan_sha), directory / "smoke.pth")
    restored = E0RelativeIdentity().to(plan["device"]).eval()
    restored.load_state_dict(torch.load(directory / "smoke.pth", map_location="cpu")["net"],
                             strict=True)
    probe = batch_values(data, slice(None), plan["device"])
    head.eval()
    with torch.no_grad():
        assert torch.equal(forward(head, probe), forward(restored, probe))
    torch.cuda.synchronize()
    assert sha(root / plan_path) == plan_sha
    admission(root, plan)
    receipt = dict(
        status="RELATIVE_CURRENT_E0_REAL_UPDATE_SMOKE_COMPLETE", split="train",
        task=plan["task"], plan_sha256=plan_sha, candidate_sets=6,
        optimizer_updates=5, losses=losses, parameters_changed=True,
        frozen_no_match_unchanged=True, finite_gradients=True, strict_reload_bit_equal=True,
        full_training_allowed=False, online_allowed=False, test_eligible=False,
        scientific_result_audit_accepted=False, protected_e0_loaded=False,
        total_parameters=sum(p.numel() for p in head.parameters()),
        supervised_sets=result["supervised_sets"], wall_seconds=time.time() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        artifacts_sha256={name: sha(directory / name) for name in ("config.json", "smoke.pth")})
    (directory / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    run_smoke(ROOT, json.loads((ROOT / args.plan).read_bytes()), args.plan)


if __name__ == "__main__":
    main()
