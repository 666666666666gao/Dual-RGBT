"""Bind the existing R7 identity collector to the verified host25 mirror."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FOLDER = "reports/e0_m2_identity_p25_r1_20261101"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    smoke = json.loads((ROOT / "reports/E0_M2_IDENTITY_SMOKE_DRAFT_R7_20261031.json").read_bytes())
    full = json.loads((ROOT / "reports/E0_M2_IDENTITY_FULL_DRAFT_R7_20261031.json").read_bytes())
    new = {"scripts/collect_e0_identity_train.py", "tests/test_collect_e0_identity_train.py"}
    payload = json.dumps(dict(source=smoke["source_sha256"], inputs=smoke["input_sha256"]))
    code = """
import json,hashlib
from pathlib import Path
r=Path('/data2/gb/Dual-RGBT')
d=json.loads(__PAYLOAD__)
print(json.dumps({kind:{name:hashlib.sha256((r/name).read_bytes()).hexdigest()
                       for name in names} for kind,names in d.items()}))
""".replace("__PAYLOAD__", repr(payload))
    result = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "2025",
         "cd /data2/gb/Dual-RGBT && conda-envs/dual_rgbt/bin/python -"],
        input=code.encode(), stdout=subprocess.PIPE, check=True)
    actual = json.loads(result.stdout)
    assert actual["inputs"] == smoke["input_sha256"]
    assert all(actual["source"][name] == value for name, value in smoke["source_sha256"].items()
               if name not in new)
    assert all(sha(ROOT / name) == smoke["source_sha256"][name] for name in new)
    folder = ROOT / FOLDER
    folder.mkdir(exist_ok=False)
    run = "protected_e0_m2_identity_smoke_r1_p25_20261101"
    task = "collect_protected_e0_m2_identity_smoke_r1_p25_20261101"
    for plan in (smoke, full):
        plan.update(source_review="reports/E0_M2_IDENTITY_COLLECTOR_SOURCE_REVIEW_P25_20261101.json",
                    target_host="2025", target_root="/data2/gb/Dual-RGBT",
                    smoke_plan="reports/E0_M2_IDENTITY_SMOKE_PLAN_P25_R1_20261101.json")
        for job in plan["jobs"]:
            if plan["phase"] == "smoke":
                job.update(run=run, task=task, assigned_gpu="2")
            else:
                shard = job["shard"]
                job.update(run="protected_e0_m2_identity_full_r1_p25_20261101_s%d" % shard,
                           task="collect_protected_e0_m2_identity_full_r1_p25_20261101_s%d" % shard,
                           assigned_gpu=str(2 + shard % 2))
        if plan["phase"] == "full":
            plan.update(smoke_run=run, smoke_task=task)
        name = Path(plan["reviewed_draft"]).name
        (folder / name).write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(source_count=len(smoke["source_sha256"]),
                         input_count=len(smoke["input_sha256"]),
                         drafts={p.name: sha(p) for p in folder.glob("*.json")}), indent=2))


if __name__ == "__main__":
    main()
