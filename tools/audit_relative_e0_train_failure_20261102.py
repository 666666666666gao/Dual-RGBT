"""Read-only attribution of an already rejected head on accepted TRAIN states."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path("/data2/gb/Dual-RGBT")
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import torch
from dual_rgbt.e0_relative_identity import E0RelativeIdentity
from relative_e0_train_data import load_verified_data

torch.set_num_threads(1)
plan_path = ROOT / "reports/e0_relative_full_training_r1_20261102/plan.json"
plan = json.loads(plan_path.read_bytes())
acceptance_path = ROOT / "reports/E0_M2_RELATIVE_FULL_TRAIN_ACCEPTANCE_R1_20261102.json"
accepted = json.loads(acceptance_path.read_bytes())
assert accepted["retain_protected_e0"] and accepted["best_qualifying_epoch"] is None
assert accepted["plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
for name, digest in plan["source_sha256"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
data = load_verified_data(ROOT, plan["collection_plan"], plan["collection_acceptance"])
head = E0RelativeIdentity().cpu().eval()
weights = ROOT / plan["run"] / "latest.pth"
assert hashlib.sha256(weights.read_bytes()).hexdigest() == accepted["artifacts_sha256"]["latest.pth"]
head.load_state_dict(torch.load(weights, map_location="cpu")["net"], strict=True)
parts = {}
with torch.no_grad():
    for part, fields in data.items():
        logits = torch.cat([
            head(*(fields[key][start:start + 512] for key in
                   ("anchor", "current", "current_coverage", "candidates", "coverage")))
            for start in range(0, len(fields["labels"]), 512)])
        selected = logits.argmax(1)
        current = fields["current_iou"]
        candidate = fields["candidate_iou"]
        selected_iou = candidate.gather(1, (selected - 1).clamp_min(0)[:, None]).squeeze(1)
        unknown = fields["labels"].gather(1, selected[:, None]).squeeze(1).eq(-1)
        reliable = current.ge(.5)
        lost = current.lt(.1)
        replacement = selected.gt(0)
        harmful = replacement & reliable & selected_iou.lt(.5)
        available = fields["coverage"].eq(1).any(2)
        recoverable = (available & candidate.ge(.5)).any(1) & fields["current_coverage"].eq(1).any(1)
        known_logits = logits.masked_fill(fields["labels"].eq(-1), float("-inf"))
        known_logits[:, 0] = logits[:, 0]
        masked_selected = known_logits.argmax(1)
        masked_iou = candidate.gather(1, (masked_selected - 1).clamp_min(0)[:, None]).squeeze(1)
        margin = logits[:, 1:].max(1).values - logits[:, 0]
        groups = {}
        for name, mask in (("reliable_harmful", harmful),
                           ("reliable_safe_replacement", replacement & reliable & ~harmful),
                           ("lost_recovered", replacement & lost & selected_iou.ge(.5))):
            values = margin[mask]
            groups[name] = dict(count=int(mask.sum()),
                               margin_quantiles=torch.quantile(values, torch.tensor([0., .25, .5, .75, 1.])).tolist()
                               if len(values) else [])
        parts[part] = dict(
            sets=len(selected), reliable_sets=int(reliable.sum()), lost_sets=int(lost.sum()),
            replacements=int(replacement.sum()),
            reliable_harmful=int(harmful.sum()),
            harmful_selected_original_unknown=int((harmful & unknown).sum()),
            harmful_selected_known_negative=int((harmful & ~unknown).sum()),
            original_unknown_replacements=int((replacement & unknown).sum()),
            lost_recovered=int((replacement & lost & selected_iou.ge(.5)).sum()),
            oracle_available_lost_recovery_upper_bound=int((lost & recoverable).sum()),
            label_masked_counterfactual=dict(
                gt_dependent=True, online_allowed=False,
                reliable_harmful=int((masked_selected.gt(0) & reliable & masked_iou.lt(.5)).sum()),
                lost_recovered=int((masked_selected.gt(0) & lost & masked_iou.ge(.5)).sum())),
            margin_groups=groups)
        assert parts[part]["reliable_harmful"] == accepted["retained_cpu_replay"]["12"][part]["reliable_to_below_half"]
        assert parts[part]["lost_recovered"] == accepted["retained_cpu_replay"]["12"][part]["lost_to_at_least_half"]
assert not torch.cuda.is_initialized()
print(json.dumps(dict(
    status="REJECTED_RELATIVE_HEAD_TRAIN_ATTRIBUTION_COMPLETE",
    plan_sha256=accepted["plan_sha256"],
    acceptance_sha256=hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
    checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
    optimizer_updates=0, cuda_initialized=False, online_allowed=False,
    test_eligible=False, split="train", parts=parts,
    scope="Descriptive failure attribution, no new training, tuning, or checkpoint selection. "
          "Original-label masking and oracle counts use TRAIN GT and cannot be online rules. "
          "Margin quantiles are diagnostic only; no threshold is selected.")), flush=True)
