"""Prespecified zero-update TRAIN diagnostics for the recovery-objective run."""
import json
import os
from pathlib import Path
import sys

ROOT = Path("/home/gaob/Dual-RGBT/snapshots/context_recovery_r1_20261102")
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import torch
from context_e0_training import evaluate
from context_recovery_execution import data_for_plan
from current_e0_supplement_contract import sha
from dual_rgbt.e0_context_advantage import E0ContextAdvantage

torch.set_num_threads(1)
plan_path = ROOT / "reports/e0_context_recovery_r1_20261102/full_plan.json"
plan = json.loads(plan_path.read_bytes())
accepted_path = ROOT / "reports/E0_M2_CONTEXT_RECOVERY_FULL_ACCEPTANCE_R1_20261102.json"
accepted = json.loads(accepted_path.read_bytes())
assert accepted["status"] == "RECOVERY_CONTEXT_FULL_EXECUTION_VERIFIED"
assert accepted["plan_sha256"] == sha(plan_path)
for name, digest in plan["source_sha256"].items():
    assert sha(ROOT / name) == digest, name
data = data_for_plan(ROOT, plan)
weights = ROOT / plan["run"] / "latest.pth"
assert sha(weights) == accepted["artifacts_sha256"]["latest.pth"]
head = E0ContextAdvantage().cpu().eval().requires_grad_(False)
head.load_state_dict(torch.load(weights, map_location="cpu")["net"], strict=True)
parts = {}
with torch.no_grad():
    for part, values in data.items():
        actual = evaluate(head, values, 512)
        assert actual == accepted["retained_cpu_replay"]["latest.pth"][part]
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
        groups = {}
        for name, mask in (
                ("reliable_degrading", observed & reliable[:, None] & delta.lt(0)),
                ("reliable_improving", observed & reliable[:, None] & delta.gt(0)),
                ("lost_recovering", good & lost[:, None]),
                ("lost_other", observed & lost[:, None] & ~good)):
            assert mask.any()
            groups[name] = dict(
                actions=int(mask.sum()),
                positive_predicted_advantages=int((mask & scores.gt(0)).sum()),
                mean_target_delta=float(delta[mask].mean()),
                mean_predicted_advantage=float(scores[mask].mean()))
        parts[part] = dict(
            actual_metrics=actual,
            recoverable_lost_sets=int(oracle.sum()),
            recoverable_lost_keep=int((oracle & chosen.eq(0)).sum()),
            recoverable_lost_wrong_candidate=int((oracle & chosen.gt(0) & ~recovered).sum()),
            recoverable_lost_recovered=int((oracle & recovered).sum()),
            recoverable_lost_no_positive_recovery_advantage=int((oracle & best_good.le(0)).sum()),
            recoverable_lost_positive_recovery_but_outranked=int(
                (oracle & best_good.gt(0) & ~recovered).sum()),
            action_groups=groups)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(
    status="RECOVERY_CONTEXT_PRESPECIFIED_TRAIN_DIAGNOSTICS",
    plan_sha256=sha(plan_path), acceptance_sha256=sha(accepted_path),
    dataset_sha256=plan["dataset_sha256"], checkpoint_sha256=sha(weights),
    split="train", optimizer_updates=0, cuda_initialized=False,
    online_allowed=False, test_eligible=False, parts=parts,
    scope="Latest fixed run only. GT diagnostic groups never enter online inputs or "
          "actions. No threshold, epoch, or recipe is selected by this audit.")), flush=True)
