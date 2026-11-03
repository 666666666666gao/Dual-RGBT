"""Prepare disabled batch-scaled FIT metadata from actual full acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_e0_policy_refresh_full_fit import prepare


def prepare_scaled(acceptance, target_acceptance, task, gpu, batch_size, workers):
    plan, metadata = prepare(acceptance, target_acceptance, task, gpu, batch_size, workers)
    trainer = "tools/train_e0_policy_refresh_scaled.py"
    plan.update(
        supplemental_sources={
            trainer: hashlib.sha256((ROOT / trainer).read_bytes()).hexdigest()},
        lr_scale=batch_size / 128,
        preview_lr=1e-5 * (batch_size / 128),
        quality_gain_lr=1e-4 * (batch_size / 128))
    plan["command"][1] = plan["root"] + "/" + trainer
    payload = {key: value for key, value in plan.items()
               if key not in ("review", "launch_allowed")}
    metadata.update(
        plan_binding_sha256=hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        supplemental_sources=plan["supplemental_sources"],
        lr_scale=plan["lr_scale"], preview_lr=plan["preview_lr"],
        quality_gain_lr=plan["quality_gain_lr"])
    return plan, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", required=True)
    parser.add_argument("--target-acceptance", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan, metadata = prepare_scaled(
        args.acceptance, args.target_acceptance, args.task, args.gpu,
        args.batch_size, args.workers)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "binding.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
