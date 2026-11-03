"""Bind successful native TRAIN smokes into disabled full C2 plans."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("--variants", nargs="+", required=True,
                        choices=("zero", "film", "shared", "routed"))
    args = parser.parse_args()
    source = ROOT / "reports/c2_training_smoke_r2_20261103" / args.branch
    folder = ROOT / "reports/c2_training_full_r1_20261103" / args.branch
    assert len(args.variants) == len(set(args.variants))
    bindings = []
    for variant in args.variants:
        runtime = source / variant / "runtime"
        audit = json.loads((runtime / "audit.json").read_bytes())
        assert audit["status"] == "ARCHIVED_SMOKE_PASS"
        for name, digest in audit["archived_sha256"].items():
            assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == digest
        plan = json.loads((runtime / "plan.json").read_bytes())
        native = json.loads((runtime / "native.json").read_bytes())
        receipt = json.loads((runtime / "receipt.json").read_bytes())
        assert native["state"] == "COMPLETE" and native["exit_code"] == 0
        assert native["command"] == plan["command"]
        assert receipt["route_counts"] == {str(route): 1 for route in range(10)}
        assert receipt["updates"] == 10 and receipt["strict_reload"]
        assert set(receipt["artifacts"]) == {"epoch_001.pth"}
        smoke_run = PurePosixPath(plan["run"])
        task = "train_c2_%s_%s_full_r1_p%s_20261103" % (
            args.branch, variant, plan["host"][-2:])
        root = PurePosixPath(plan["root"])
        plan.update(smoke=False, launch_allowed=False, task=task,
                    run=str(smoke_run.parent / task))
        paths = dict(smoke_receipt=smoke_run / "SMOKE.json",
                     smoke_config=smoke_run / "config.json",
                     smoke_native=root / "runs/pipeline_tasks" / native["task"] / "status.json")
        for key, remote in paths.items():
            local = dict(smoke_receipt="receipt.json", smoke_config="config.json",
                         smoke_native="native.json")[key]
            plan["inputs"][key] = dict(
                path=str(remote), sha256=audit["archived_sha256"][local])
        review = "reports/P2_C2_%s_FULL_ADMISSION_R1_20261103.json" % args.branch.upper()
        plan["review"] = dict(path=str(root / review), sha256="PENDING_REVIEW_SHA256")
        plan["command"][-1] = str(root / "reports" / task / "plan.json")
        binding = {key: value for key, value in plan.items()
                   if key not in ("launch_allowed", "review")}
        digest = hashlib.sha256(json.dumps(
            binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        destination = folder / variant
        destination.mkdir(parents=True, exist_ok=False)
        raw = (json.dumps(plan, indent=2) + "\n").encode()
        (destination / "plan.json").write_bytes(raw)
        item = dict(task=task, variant=variant, gpu=plan["gpu"], launch_allowed=False,
                    binding_sha256=digest, draft_sha256=hashlib.sha256(raw).hexdigest(),
                    review=review, smoke_task=native["task"],
                    smoke_receipt_sha256=audit["archived_sha256"]["receipt.json"])
        (destination / "binding.json").write_text(json.dumps(item, indent=2) + "\n")
        bindings.append(item)
    print(json.dumps(dict(status="PREPARED_DISABLED", branch=args.branch, plans=bindings)))


if __name__ == "__main__":
    main()
