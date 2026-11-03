"""Read-only CPU diagnostics on matched completed TRAIN cache prefixes."""
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    root, branch, per_shard = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    assert branch in ("sot", "dropmoe") and per_shard > 0
    host = "p25" if branch == "dropmoe" else "p26"
    folder = root / "reports" / ("p1b_" + branch + "_full_collection_r1_20261102")
    total = dict(records=0, outside_crop=0, query_selected=0,
                 beneficial_query_any_modality=0, positive_query_chosen_modality=0,
                 costly_query_opportunity_chosen_modality=0)
    action_counts = [0] * 5
    quality_sum = gain_sum = best_net_gain_sum = regret_sum = 0.
    iou_sum = 0.
    evidence = []
    reference, token_cost = None, None
    for shard in range(4):
        plan_path = folder / ("s" + str(shard)) / "plan.json"
        plan_raw = plan_path.read_bytes()
        plan = json.loads(plan_raw)
        manifest_raw = Path(plan["inputs"]["manifest"]["path"]).read_bytes()
        assert hashlib.sha256(manifest_raw).hexdigest() == plan["inputs"]["manifest"]["sha256"]
        manifest = json.loads(manifest_raw)
        if reference is None:
            reference = plan
            executor_raw = Path(plan["inputs"]["executor"]["path"]).read_bytes()
            assert hashlib.sha256(executor_raw).hexdigest() == plan["inputs"]["executor"]["sha256"]
            token_cost = torch.load(io.BytesIO(executor_raw), map_location="cpu")["net"]["token_cost"]
            assert token_cost.shape == (5,) and torch.isfinite(token_cost).all()
            del executor_raw
        for key in ("inputs", "sources", "cost_weight", "label_stride",
                    "root", "branch", "shards", "round"):
            assert plan[key] == reference[key], key
        selected = manifest["train"][shard::4][:per_shard]
        progress = json.loads((Path(plan["run"]) / "progress.json").read_text())
        assert progress["sequences"] >= per_shard
        native = json.loads((root / "runs/pipeline_tasks" / plan["task"] / "status.json").read_text())
        assert native["task"] == plan["task"] and native["command"] == plan["command"]
        assert plan["host"] == ("2025" if host == "p25" else "2026")
        for sequence in selected:
            path = Path(plan["run"]) / (sequence["name"] + ".pth")
            raw = path.read_bytes()
            cache = torch.load(io.BytesIO(raw), map_location="cpu")
            assert cache["sequence"] == sequence["name"]
            assert cache["plan_sha256"] == hashlib.sha256(plan_raw).hexdigest()
            expected_frames = [frame for frame in sequence["valid_frames"]
                               if frame > 0 and frame % plan["label_stride"] == 0]
            assert [record["frame"] for record in cache["records"]] == expected_frames
            for record in cache["records"]:
                quality = record["quality_targets"].numpy().reshape(2)
                gain = record["gain_targets"].numpy().reshape(2, 4)
                target = np.asarray(record["target"])
                assert all(np.isfinite(value).all() for value in (quality, gain, target))
                modality, action = record["route"]["modality"], record["route"]["action"]
                assert modality in (0, 1) and action in range(5)
                options = torch.cat((torch.zeros(1, dtype=token_cost.dtype),
                                     torch.from_numpy(gain[modality]))).sub(
                                         plan["cost_weight"] * token_cost).numpy()
                best_net_gain_sum += max(options)
                regret_sum += max(options) - options[action]
                quality_sum += float(quality.sum())
                gain_sum += float(gain.sum())
                centre = target[:2] + target[2:] / 2
                total["outside_crop"] += int(not ((centre >= 0) & (centre < 1)).all())
                total["records"] += 1
                total["query_selected"] += int(action > 0)
                total["beneficial_query_any_modality"] += int(gain.max() > 0)
                total["positive_query_chosen_modality"] += int(gain[modality].max() > 0)
                total["costly_query_opportunity_chosen_modality"] += int(max(options) > 0)
                action_counts[action] += 1
                predicted = np.asarray(record["prediction"])
                truth = np.asarray(sequence["boxes"][record["frame"]])
                intersection = np.maximum(
                    np.minimum(predicted[:2] + predicted[2:], truth[:2] + truth[2:])
                    - np.maximum(predicted[:2], truth[:2]), 0).prod()
                union = predicted[2:].prod() + truth[2:].prod() - intersection
                assert union > 0
                iou_sum += float(intersection / union)
            evidence.append(dict(shard=shard, sequence=sequence["name"],
                                 records=len(cache["records"]),
                                 cache_sha256=hashlib.sha256(raw).hexdigest(),
                                 plan_sha256=hashlib.sha256(plan_raw).hexdigest()))
    count = total["records"]
    assert count > 0 and not torch.cuda.is_initialized()
    print(json.dumps(dict(
        stage="P1B_direct_E0_matched_TRAIN_prefix_diagnostic",
        branch=branch, root=str(root), sequences=len(evidence),
        per_shard=per_shard, counts=total, action_counts=action_counts,
        mean_quality_target=quality_sum / (count * 2),
        mean_query_gain=gain_sum / (count * 8),
        mean_best_net_gain_chosen_modality=best_net_gain_sum / count,
        mean_chosen_action_net_regret=regret_sum / count,
        mean_saved_prediction_iou=iou_sum / count,
        all_checked_targets_finite=True, cuda_initialized=False,
        token_cost=token_cost.tolist(),
        executor_sha256=reference["inputs"]["executor"]["sha256"],
        manifest_sha256=reference["inputs"]["manifest"]["sha256"],
        full_collection_acceptance=False, benchmark_evidence=False,
        scope="Matched ordered TRAIN prefix only; descriptive, not full-population accuracy or fresh-policy counterfactuals.",
        evidence=evidence), indent=2))


if __name__ == "__main__":
    main()
