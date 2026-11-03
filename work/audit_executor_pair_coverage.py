"""Measure the current V2 search-frame domain using TRAIN metadata only."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import hashlib,json
from pathlib import Path
root=Path("/home/gaob/Dual-RGBT/snapshots/p1b_dropmoe_p25_r1_20261102")
plan=json.loads((root/"reports/p1b_dropmoe_full_collection_r1_20261102/s0/plan.json").read_text())
item=plan["inputs"]["manifest"]
raw=Path(item["path"]).read_bytes()
assert hashlib.sha256(raw).hexdigest()==item["sha256"]
rows=json.loads(raw)["train"]
assert len(rows)==882
evidence=[]
for row in rows:
    valid=row["valid_frames"]
    assert valid==sorted(set(valid)) and valid
    first=valid[0]
    current=[frame for frame in valid if first<frame<=first+200]
    causal=[frame for previous,frame in zip(valid,valid[1:]) if frame-previous<=200]
    excluded=[frame for frame in causal if frame>first+200]
    evidence.append(dict(sequence=row["name"],valid_frames=len(valid),
                         current_search_frames=len(current),
                         full_timeline_causal_search_frames=len(causal),
                         excluded_later_search_frames=len(excluded),
                         first_valid_frame=first,last_valid_frame=valid[-1]))
summary={field:sum(row[field] for row in evidence) for field in (
    "valid_frames","current_search_frames","full_timeline_causal_search_frames",
    "excluded_later_search_frames")}
summary["sequences_with_excluded_later_search_frames"]=sum(
    row["excluded_later_search_frames"]>0 for row in evidence)
summary["search_frame_domain_coverage"]=(
    summary["current_search_frames"]/summary["full_timeline_causal_search_frames"])
print(json.dumps(dict(
    stage="P1A_V2_TRAIN_search_domain_audit",host="2025",
    manifest=item,source={"path":str(root/"dual_rgbt/data.py"),
                         "sha256":hashlib.sha256((root/"dual_rgbt/data.py").read_bytes()).hexdigest()},
    sequences=len(rows),max_interval=200,summary=summary,per_sequence=evidence,
    metadata_only=True,actual_sampled_training_frames_reconstructed=False,
    gpu_execution=False,benchmark_evidence=False),indent=2))
'''
output = ROOT / "reports/P1A_V2_TRAIN_SEARCH_DOMAIN_AUDIT_20261103.json"
assert not output.exists()
result = subprocess.run(
    ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "2025",
     "/home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python", "-c", shlex.quote(REMOTE)],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
report = json.loads(result.stdout)
assert report["sequences"] == 882
output.write_bytes(result.stdout)
print(json.dumps(report["summary"], indent=2))
