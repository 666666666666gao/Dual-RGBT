"""Authenticate the fixed P1B terminal checkpoint, then run the E0 evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_inputs import sha


def validate_fit(review, checkpoint, checkpoint_sha256):
    evidence = review["p1b_fit_evidence"]
    documents = {}
    for name, item in evidence.items():
        assert sha(item["remote_path"]) == item["sha256"], name
        documents[name] = json.loads(Path(item["remote_path"]).read_bytes())
    plan = documents["plan.json"]
    complete = documents["complete.json"]
    native = documents["native.json"]
    fit_review = documents["fit_review.json"]
    assert plan == documents["config.json"]
    assert plan["stage"] == "P1B_direct_E0" and plan["mode"] == "fit"
    assert plan["branch"] == review["branch"] and plan["host"] == review["target_host"]
    assert plan["round"] == 1 and plan["smoke"] is False
    assert plan["launch_allowed"] is True
    assert plan["epochs"] == 5 and plan["batch_size"] == 512
    assert native["state"] == "COMPLETE" and native["exit_code"] == 0
    assert native["task"] == plan["task"] and native["physical_gpu"] == plan["gpu"]
    assert native["command"] == plan["command"]
    assert native["command"][-2:] == ["--plan", evidence["plan.json"]["remote_path"]]
    assert sha(plan["review"]["path"]) == plan["review"]["sha256"]
    assert fit_review["stage"] == plan["stage"]
    assert fit_review["status"] == "SOURCE_PASS" and fit_review["blocking_issues"] == 0
    assert fit_review["sources"] == plan["sources"]
    binding = hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items()
         if key not in ("launch_allowed", "review")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert binding == review["p1b_fit_plan_binding"]
    assert binding in fit_review["admitted_plan_bindings"]
    for name, digest in dict(plan["sources"], **plan["supplemental_sources"]).items():
        assert sha(Path(plan["root"]) / name) == digest, name
    for item in plan["inputs"].values():
        assert sha(item["path"]) == item["sha256"]
    assert documents["precision.json"] == dict(
        matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False)
    assert complete["stage"] == "P1B_direct_E0_fit" and complete["status"] == "COMPLETE"
    assert complete["smoke"] is False and complete["closed_loop_round"] == 1
    assert complete["branch"] == plan["branch"] and complete["sources"] == plan["sources"]
    assert complete["plan_sha256"] == evidence["plan.json"]["sha256"]
    assert (complete["epochs"], complete["steps"], complete["records"]) == (5, 890, 91091)
    assert complete["executor_preserved"] is complete["all_policy_modules_changed"] is True
    assert complete["strict_reload"] is True and complete["benchmark_evidence"] is False
    assert complete["executor_sha256"] == plan["inputs"]["executor"]["sha256"]
    assert set(complete["artifacts"]) == {"epoch_%03d.pth" % epoch for epoch in range(1, 6)}
    for name, digest in complete["artifacts"].items():
        assert sha(Path(plan["run"]) / name) == digest, name
    assert checkpoint == str(Path(plan["run"]) / "epoch_005.pth")
    assert sha(checkpoint) == checkpoint_sha256 == complete["artifacts"]["epoch_005.pth"]
    saved = torch.load(checkpoint, map_location="cpu")
    original = torch.load(plan["inputs"]["executor"]["path"], map_location="cpu")
    assert saved["config"] == plan and saved["epoch"] == 5 and saved["step"] == 890
    assert saved["executor_sha256"] == complete["executor_sha256"]
    assert set(saved["net"]) == set(original["net"])
    prefixes = ("preview.", "quality.", "gain.")
    for name, tensor in saved["net"].items():
        assert torch.isfinite(tensor).all(), name
        if not name.startswith(prefixes):
            assert torch.equal(tensor, original["net"][name]), name
    for prefix in prefixes:
        assert any(not torch.equal(tensor, saved["net"][name])
                   for name, tensor in original["net"].items() if name.startswith(prefix))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("source-review", "source-review-sha256", "checkpoint", "checkpoint-sha256"):
        parser.add_argument("--" + name, required=True)
    args, _ = parser.parse_known_args()
    assert sha(args.source_review) == args.source_review_sha256
    review = json.loads(Path(args.source_review).read_bytes())
    assert review["stage"] == "E0_finetuned_evaluation"
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["target_root"] == str(ROOT)
    assert review["checkpoint"] == dict(path=args.checkpoint, sha256=args.checkpoint_sha256)
    assert all(sha(ROOT / name) == digest for name, digest in review["source_sha256"].items())
    torch.set_num_threads(1)
    validate_fit(review, args.checkpoint, args.checkpoint_sha256)
    torch.backends.cudnn.benchmark = False
    from eval_e0_finetuned import main as evaluate
    evaluate()
    run = Path(sys.argv[sys.argv.index("--run") + 1])
    (run / "precision.json").write_text(json.dumps(dict(
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark)) + "\n")


if __name__ == "__main__":
    main()
