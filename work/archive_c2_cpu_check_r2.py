"""Archive the exact successful isolated C2 CPU check."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import base64, hashlib, json
from pathlib import Path
root = Path("/data2/gaob/Dual-RGBT/c2_cpu_check_r2_20261103")
names = ["invocation.json", "source_review.json", "run.log", "result.json",
         "dual_rgbt/c2_executor.py", "tools/check_c2_executor_cpu.py",
         "dual_rgbt/route_lora.py", "dual_rgbt/missing_modality.py"]
result = json.loads((root / "result.json").read_text())
assert result["status"] == "PASS" and result["cuda_initialized"] is False
assert result["strict_state_dict_reload"] and result["frozen_policy_equal"]
assert len(result["variants"]) == 4 and len(result["cpu_optimizer_updates"]) == 2
invocation = json.loads((root / "invocation.json").read_text())
files = {}
for name in names:
    raw = (root / name).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if name in invocation["files"]:
        assert digest == invocation["files"][name]
    files[name] = dict(raw=base64.b64encode(raw).decode(), sha256=digest)
assert files["source_review.json"]["sha256"] == invocation["review_sha256"]
print(json.dumps(dict(status="PASS", stage=result["stage"], files=files,
                     gpu_execution=False, native_training=False, benchmark_evidence=False)))
'''
response = subprocess.run(
    ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "2025",
     "/home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python", "-c", shlex.quote(REMOTE)],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
report = json.loads(response.stdout)
folder = ROOT / "reports/c2_executor_cpu_r2_20261103"
folder.mkdir(exist_ok=False)
files = report.pop("files")
for name, item in files.items():
    raw = base64.b64decode(item["raw"])
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    target = folder / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
report["archived_sha256"] = {name: item["sha256"] for name, item in files.items()}
(folder / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
