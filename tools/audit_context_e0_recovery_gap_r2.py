"""Attribute rejected context-head recovery errors on accepted TRAIN tensors."""
import json
import os
from pathlib import Path
import sys

ROOT = Path("/home/gaob/Dual-RGBT/snapshots/context_advantage_r2_20261102")
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import torch
from context_e0_execution import exported_data
from context_e0_training import evaluate
from current_e0_supplement_contract import sha
from dual_rgbt.e0_context_advantage import E0ContextAdvantage, advantage_loss

torch.set_num_threads(1)
plan_path = ROOT / "reports/e0_context_advantage_r2_20261102/full_plan.json"
plan = json.loads(plan_path.read_bytes())
accepted_path = ROOT / "reports/E0_M2_CONTEXT_FULL_TRAIN_ACCEPTANCE_R2_20261102.json"
accepted = json.loads(accepted_path.read_bytes())
assert accepted["plan_sha256"] == sha(plan_path)
assert accepted["best_qualifying_epoch"] is None and accepted["retain_protected_e0"]
for name, digest in plan["source_sha256"].items():
    assert sha(ROOT / name) == digest, name
data = exported_data(ROOT, plan)
weights = ROOT / plan["run"] / "latest.pth"
assert sha(weights) == accepted["artifacts_sha256"]["latest.pth"]
head = E0ContextAdvantage().cpu().eval().requires_grad_(False)
head.load_state_dict(torch.load(weights, map_location="cpu")["net"], strict=True)
parts = {}
with torch.no_grad():
    for part, values in data.items():
        assert evaluate(head, values, 512) == accepted["retained_cpu_replay"]["latest.pth"][part]
        logits = torch.cat([head(values["features"][start:start + 512],
                                 values["observed"][start:start + 512])
                            for start in range(0, len(values["features"]), 512)])
        chosen = logits.argmax(1)
        scores = logits[:, 1:]
        observed = values["observed"]
        current = values["current_iou"]
        candidate = values["candidate_iou"]
        delta = candidate - current[:, None]
        lost, reliable = current.lt(.1), current.ge(.5)
        good = observed & candidate.ge(.5)
        oracle = lost & good.any(1)
        selected_iou = candidate.gather(1, (chosen - 1).clamp_min(0)[:, None]).squeeze(1)
        recovered = chosen.gt(0) & selected_iou.ge(.5)
        best_good = scores.masked_fill(~good, float("-inf")).max(1).values
        loss, terms = advantage_loss(
            logits, observed, current, candidate,
            values["advantage_weight"], values["protection_weight"])
        groups = {}
        for name, mask in (
                ("reliable_degrading", observed & reliable[:, None] & delta.lt(0)),
                ("reliable_improving", observed & reliable[:, None] & delta.gt(0)),
                ("lost_recovering", good & lost[:, None]),
                ("lost_other", observed & lost[:, None] & ~good)):
            groups[name] = dict(
                actions=int(mask.sum()),
                positive_predicted_advantages=int((mask & scores.gt(0)).sum()),
                mean_target_delta=float(delta[mask].mean()) if mask.any() else None,
                mean_predicted_advantage=float(scores[mask].mean()) if mask.any() else None)
        parts[part] = dict(
            sets=len(current), observed_actions=int(observed.sum()),
            reliable_sets=int(reliable.sum()), lost_sets=int(lost.sum()),
            recoverable_lost_sets=int(oracle.sum()),
            recoverable_lost_keep=int((oracle & chosen.eq(0)).sum()),
            recoverable_lost_wrong_candidate=int((oracle & chosen.gt(0) & ~recovered).sum()),
            recoverable_lost_recovered=int((oracle & recovered).sum()),
            recoverable_lost_no_positive_recovery_advantage=int((oracle & best_good.le(0)).sum()),
            recoverable_lost_positive_recovery_but_outranked=int(
                (oracle & best_good.gt(0) & ~recovered).sum()),
            reliable_harmful=int((reliable & chosen.gt(0) & selected_iou.lt(.5)).sum()),
            reliable_harmful_original_unknown=int((
                reliable & chosen.gt(0) & selected_iou.lt(.5)
                & values["labels"].gather(1, chosen[:, None]).squeeze(1).eq(-1)).sum()),
            action_groups=groups,
            full_partition_weighted_objective=dict(
                total=float(loss), **{key: float(value) for key, value in terms.items()}))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(
    status="CONTEXT_REJECTED_HEAD_TRAIN_RECOVERY_GAP_ATTRIBUTED",
    plan_sha256=sha(plan_path), acceptance_sha256=sha(accepted_path),
    dataset_sha256=plan["dataset_sha256"], checkpoint_sha256=sha(weights),
    split="train", optimizer_updates=0, cuda_initialized=False,
    online_allowed=False, test_eligible=False, parts=parts,
    scope="Actual rejected latest head on fixed accepted TRAIN tensors. GT groups and "
          "oracle counts are diagnostic only, never online inputs or action rules. "
          "No threshold, checkpoint, or training recipe is selected by this audit.")), flush=True)
