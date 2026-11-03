"""Deploy and execute the one precisely reviewed additive CPU profile."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/p1b_route_profile_r1_20261102"
draft = json.loads((FOLDER / "draft.json").read_bytes())
review_name = draft["review"]["path"]
raw = (ROOT / review_name).read_bytes()
review = json.loads(raw)
digest = hashlib.sha256(raw).hexdigest()
assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
assert review["stage"] == draft["stage"]
assert review["target_root"] == draft["root"]
assert review["reviewed_draft"]["sha256"] == hashlib.sha256(
    (FOLDER / "draft.json").read_bytes()).hexdigest()
assert review["admitted_operation"]["command_template"] == draft["command"]
assert review["checkpoint"] == draft["checkpoint"]
assert review["factory"] == draft["factory"]
assert len(review["source_sha256"]) == 99
for name, expected in review["source_sha256"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
assert not (FOLDER / "execution.json").exists()
command = [digest if item == "PENDING_REVIEW_SHA256" else item
           for item in draft["command"]]
execution = dict(draft_sha256=review["reviewed_draft"]["sha256"],
                 review_sha256=digest, command=command,
                 environment=draft["environment"])
(FOLDER / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
payload = io.BytesIO()
with tarfile.open(fileobj=payload, mode="w") as archive:
    for name in (draft["tool"], review_name):
        archive.add(str(ROOT / name), arcname=name)
ssh = ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", draft["host"]]
check = "test ! -e {tool} && test ! -e {output} && test ! -e {review}".format(
    tool=shlex.quote(draft["root"] + "/" + draft["tool"]),
    output=shlex.quote(draft["root"] + "/" + draft["output"]),
    review=shlex.quote(draft["root"] + "/" + review_name))
subprocess.run(ssh + [check], check=True)
subprocess.run(ssh + ["tar", "-C", draft["root"], "-xf", "-"],
               input=payload.getvalue(), check=True)
env = " ".join(shlex.quote(key + "=" + value)
               for key, value in draft["environment"].items())
remote = "cd " + shlex.quote(draft["root"]) + " && env " + env + " " + \
         " ".join(shlex.quote(item) for item in command)
with (FOLDER / "run.log").open("xb") as log:
    result = subprocess.run(ssh + [remote], stdout=log, stderr=subprocess.STDOUT)
(FOLDER / "exit.json").write_text(json.dumps(dict(exit_code=result.returncode)) + "\n")
result.check_returncode()
output = subprocess.check_output(ssh + ["cat", draft["root"] + "/" + draft["output"]])
(FOLDER / "measurement.json").write_bytes(output)
measured = json.loads(output)
print(json.dumps(dict(output_sha256=hashlib.sha256(output).hexdigest(),
                      measurement=measured)), flush=True)
