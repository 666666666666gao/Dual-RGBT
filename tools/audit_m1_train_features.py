"""Diagnose the frozen M1 head and normalization using accepted TRAIN caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    root = Path(args.root)
    output = root / args.output
    assert not output.exists()
    sys.path[:0] = [str(root), str(root / "scripts")]
    import torch
    from train_e0_reliability import load_collection, fit_normalization, metrics
    from dual_rgbt.e0_reliability import E0Reliability, localization_targets

    torch.set_num_threads(1)
    run = root / "runs/protected_e0_offline_m1_20261031"
    config = json.loads((run / "config.json").read_bytes())
    plan = config["plan"]
    native = root / "runs/pipeline_tasks/train_protected_e0_offline_m1_r5_20261101/status.json"
    status = json.loads(native.read_bytes())
    assert status["state"] == "COMPLETE" and status["exit_code"] == 0
    receipt = json.loads((run / "COMPLETE.json").read_bytes())
    assert all(sha(run / name) == digest for name, digest in receipt["artifacts_sha256"].items())
    assert all(sha(root / name) == digest for name, digest in plan["source_sha256"].items())
    assert sha(root / plan["collection_acceptance"]) == plan["collection_acceptance_sha256"]
    data, used = load_collection(root, plan)
    checkpoint = torch.load(run / "best.pth", map_location="cpu")
    mean, scale = fit_normalization(data["fit"]["features"])
    assert torch.equal(mean, checkpoint["feature_mean"])
    assert torch.equal(scale, checkpoint["feature_scale"])
    head = E0Reliability().eval().requires_grad_(False)
    head.load_state_dict(checkpoint["net"], strict=True)
    # Layout follows the reviewed E0Observation concatenation, not inferred correlations.
    groups = dict(preview=(0, 128), history=(128, 136), deep_roi=(136, 904),
                  roi_coverage=(904, 905), score_stats=(905, 908),
                  box=(908, 912), modality=(912, 914))
    summaries = {}
    with torch.no_grad():
        for part, values in data.items():
            normalized = (values["features"] - mean) / scale
            target = localization_targets(values["iou"])
            probabilities = head(normalized).softmax(1)
            predicted = probabilities.argmax(1)
            reliable = target == 0
            interrupted = reliable & (predicted != 0)
            false_lost = reliable & (predicted == 2)
            summaries[part] = dict(
                metrics=metrics(head, normalized, values["iou"], 4096),
                reliable_states=int(reliable.sum()),
                reliable_interrupted=int(interrupted.sum()),
                reliable_predicted_lost=int(false_lost.sum()),
                reliable_predicted_lost_probability_ge_09=int(
                    (false_lost & (probabilities[:, 2] >= .9)).sum()),
                feature_groups={name: dict(
                    mean_shift_rms=float(normalized[:, start:end].mean(0).square().mean().sqrt()),
                    standardized_rms=float(normalized[:, start:end].square().mean().sqrt()),
                    max_abs=float(normalized[:, start:end].abs().max()),
                    channels_with_any_abs_gt_10=int(
                        (normalized[:, start:end].abs().max(0).values > 10).sum()),
                    saved_scale_min=float(scale[start:end].min()),
                    saved_scale_max=float(scale[start:end].max()))
                    for name, (start, end) in groups.items()})
    history = json.loads((run / "history.json").read_bytes())
    for part in summaries:
        expected = history[checkpoint["epoch"] - 1][part]
        actual = summaries[part]["metrics"]
        assert actual["confusion_true_rows_predicted_columns"] == expected["confusion_true_rows_predicted_columns"]
    result = dict(status="M1_ACCEPTED_TRAIN_FEATURE_DIAGNOSTIC",
        checkpoint_epoch=checkpoint["epoch"], checkpoint_sha256=sha(run / "best.pth"),
        native_sha256=sha(native), config_sha256=sha(run / "config.json"),
        collection_acceptance_sha256=plan["collection_acceptance_sha256"],
        cache_input_count=len(used), normalization_exactly_reproduced=True,
        source_sha256=plan["source_sha256"],
        observation_source_sha256=sha(root / "dual_rgbt/e0_observation.py"),
        script_sha256=sha(__file__), partitions=summaries,
        cuda_initialized=torch.cuda.is_initialized(), test_eligible=False,
        scientific_result_audit_accepted=False,
        scope="Descriptive TRAIN fit/component-holdout feature and prediction audit. "
              "Complete caches revalidated by original admitted loader; best head "
              "normalization and confusion reproduced. Upstream E0 saw component "
              "holdout. No intervention, causal attribution, threshold choice, "
              "new optimizer updates, TEST input or independent generalization claim.")
    assert result["cuda_initialized"] is False
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
