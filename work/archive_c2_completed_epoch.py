"""Archive a completed C2 epoch using the existing read-only runtime witness."""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess

from archive_c2_full_start import REMOTE, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("dropmoe", "sot"))
    parser.add_argument("variant", choices=("zero", "film"))
    parser.add_argument("--epoch", type=int, default=1)
    args = parser.parse_args()
    assert 1 <= args.epoch <= 20
    folder = ROOT / "reports/c2_training_full_r1_20261103" / args.branch / args.variant
    archive = folder / "epoch_{:03d}_witness".format(args.epoch)
    assert not archive.exists(), str(archive)
    plan = json.loads((folder / "plan.json").read_bytes())
    response = subprocess.run(
        ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
         plan["host"], plan["command"][0], "-B", "-c", shlex.quote(REMOTE)],
        input=json.dumps(dict(root=plan["root"], task=plan["task"])),
        encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    report = json.loads(response.stdout)
    if report["status"] == "NOT_READY":
        print(json.dumps(report))
        return
    files = report.pop("files")
    raw = {name: base64.b64decode(item["raw"]) for name, item in files.items()}
    assert all(hashlib.sha256(raw[name]).hexdigest() == item["sha256"]
               for name, item in files.items())
    prefix = "EPOCH_COMPLETE {} ".format(args.epoch)
    completed = [line for line in raw["run.log"].decode().splitlines()
                 if line.startswith(prefix)]
    if not completed:
        print(json.dumps(dict(status="NOT_READY", task=plan["task"], epoch=args.epoch)))
        return
    assert len(completed) == 1
    rows = [row for row in csv.DictReader(io.StringIO(raw["steps.csv"].decode()))
            if int(row["epoch"]) == args.epoch]
    assert len(rows) == 2500
    assert [int(row["step"]) for row in rows] == list(
        range((args.epoch - 1) * 2500 + 1, args.epoch * 2500 + 1))
    counts = Counter(str(int(row["modality"]) + 2 * int(row["action"])) for row in rows)
    assert dict(counts) == {str(route): 250 for route in range(10)}
    report.update(
        status="COMPLETED_EPOCH_WITH_LIVE_TRAINING", completed_epoch=args.epoch,
        completed_epoch_claimed=True, epoch_updates=len(rows),
        epoch_seconds=float(completed[0].split()[2]), epoch_route_counts=dict(counts),
        epoch_mean_train_loss=sum(float(row["loss"]) for row in rows) / len(rows),
        convergence_claimed=False, benchmark_evidence=False,
        archived_sha256={name: item["sha256"] for name, item in files.items()})
    archive.mkdir()
    for name, value in raw.items():
        (archive / name).write_bytes(value)
    (archive / "observation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("compute", "gpu_state", "archived_sha256")}))


if __name__ == "__main__":
    main()
