"""Archive failed R2 smoke evidence and original factory pickle dependencies."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = (
    "lib/train/__init__.py", "lib/train/admin/__init__.py",
    "lib/train/admin/settings.py", "lib/train/admin/environment.py",
    "lib/train/admin/local.py", "lib/train/admin/multigpu.py",
    "lib/train/admin/tensorboard.py", "lib/train/admin/stats.py",
)
REMOTE = r'''
import base64, hashlib, json, pickletools, sys, zipfile
from pathlib import Path
r = json.loads(sys.stdin.read())
d = r["draft"]
root = Path(d["root"])
native = root / "runs/pipeline_tasks" / d["smoke"]["task"]
status = json.loads((native / "status.json").read_bytes())
assert status["state"] == "FAILED" and status["exit_code"] == 1
assert "ModuleNotFoundError: No module named 'lib.train'" in (native / "run.log").read_text()
factory = Path(d["factory"]["path"])
assert hashlib.sha256(factory.read_bytes()).hexdigest() == d["factory"]["sha256"]
with zipfile.ZipFile(str(factory)) as archive:
    payload = archive.read(next(n for n in archive.namelist() if n.endswith("data.pkl")))
globals_ = sorted(set(arg for op, arg, pos in pickletools.genops(payload)
                      if op.name == "GLOBAL"))
files = {}
for name, path in list((name, Path(d["source_root"]) / name)
                       for name in r["dependencies"]) + [
        ("failure/status.json", native / "status.json"),
        ("failure/run.log", native / "run.log")]:
    raw = path.read_bytes()
    files[name] = dict(raw=base64.b64encode(raw).decode(),
                      sha256=hashlib.sha256(raw).hexdigest())
print(json.dumps(dict(factory=d["factory"], pickle_globals=globals_, files=files)))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    args = parser.parse_args()
    folder = ROOT / "reports/p1b_evaluation_dependencies_r3_20261104" / args.branch
    assert not folder.exists()
    draft = json.loads((ROOT / "reports/p1b_evaluation_r2_20261104" /
                        args.branch / "draft.json").read_bytes())
    result = subprocess.run([
        "ssh", "-T", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
        draft["host"], draft["smoke"]["command"][0], "-B", "-c", shlex.quote(REMOTE)],
        input=json.dumps(dict(draft=draft, dependencies=DEPENDENCIES)),
        text=True, capture_output=True, check=True)
    audit = json.loads(result.stdout)
    files = audit.pop("files")
    folder.mkdir(parents=True, exist_ok=False)
    for name, item in files.items():
        raw = base64.b64decode(item["raw"])
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    audit["source_sha256"] = {name: files[name]["sha256"] for name in DEPENDENCIES}
    audit["failure_sha256"] = {name: item["sha256"] for name, item in files.items()
                               if name.startswith("failure/")}
    (folder / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
