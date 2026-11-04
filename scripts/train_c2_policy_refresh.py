"""Fit the C2 preview, quality and gain policy on accepted TRAIN states."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dual_rgbt.c2_policy_loss import configure_policy_refresh, policy_refresh_loss
from dual_rgbt.c2_policy_refresh import file_sha, load_completed_executor
from collect_c2_policy_refresh import rebuild_state, train_sequences
from lib.test.utils.hann import hann2d

LOSS_FIELDS = ("loss", "quality_loss", "gain_loss", "spatial_loss")
POLICY_PREFIXES = ("preview.", "quality.", "gain.")


def plan_binding(plan):
    return hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items()
         if key not in ("launch_allowed", "review")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert plan["stage"] == "P2_C2_policy_refresh_fit"
    assert plan["mode"] == "fit"
    assert plan["host"] in ("2025", "2026")
    assert plan["variant"] in ("zero", "film")
    assert plan["launch_allowed"] is True and plan["smoke"] is False
    assert plan["epochs"] >= 5 and plan["batch_size"] > 0
    assert plan["workers"] >= 0 and plan["gradient_clip"] > 0
    assert plan["preview_lr"] > 0 and plan["quality_gain_lr"] > 0
    assert plan["test_enabled"] is False and plan["dev_enabled"] is False
    assert plan["task"] and Path(plan["task"]).name == plan["task"]
    assert plan["run"].startswith(plan["run_root"] + "/")
    assert plan["command"][-2:] == ["--plan", str(path)]
    assert plan["command"][-3] == str(ROOT / "scripts/train_c2_policy_refresh.py")
    assert plan["executor_binding"]["stage"] == "P2_C2_policy_refresh"
    assert plan["executor_binding"]["host"] == plan["host"]
    assert plan["executor_binding"]["variant"] == plan["variant"]
    assert plan["executor_binding"]["c2_task_id"] == plan["c2_task_id"]
    assert plan["collection_acceptance"]["path"]
    assert len(plan["collection_acceptance"]["sha256"]) == 64

    review = json.loads(Path(plan["review"]["path"]).read_text(encoding="utf-8"))
    assert file_sha(plan["review"]["path"]) == plan["review"]["sha256"]
    assert review["stage"] == plan["stage"]
    assert review["status"] == "SOURCE_PASS" and review["blocking_issues"] == 0
    assert review["sources"] == plan["sources"]
    for name, digest in plan["sources"].items():
        assert file_sha(ROOT / name) == digest, name
    digest = plan_binding(plan)
    assert digest in review["admitted_plan_bindings"]
    return plan


def validate_collection_acceptance(plan):
    item = plan["collection_acceptance"]
    assert file_sha(item["path"]) == item["sha256"]
    acceptance = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
    assert acceptance["status"] == "PASS"
    assert acceptance["stage"] == "P2_C2_policy_refresh_full_collection_acceptance"
    assert acceptance["host"] == plan["host"]
    assert acceptance["variant"] == plan["variant"]
    assert acceptance["c2_task_id"] == plan["c2_task_id"]
    assert acceptance["executor_binding"] == plan["executor_binding"]
    assert acceptance["sequences"] == 882
    assert acceptance["tracked_frames"] > 0 and acceptance["records"] > 0
    assert acceptance["benchmark_evidence"] is False
    assert acceptance["policy_fit_admitted"] is False
    assert len(acceptance["collections"]) == 4
    assert len(acceptance["native_and_artifact_evidence"]) == 4
    return acceptance


class StateDataset(Dataset):
    def __init__(self, plan, acceptance):
        self.sequences = {
            row["name"]: row for row in train_sequences(
                plan["executor_binding"]["inputs"]["manifest"]["path"])
        }
        assert len(self.sequences) == 882
        self.files = []
        self.index = []
        seen_names = set()

        for shard, (collection, evidence) in enumerate(zip(
                acceptance["collections"],
                acceptance["native_and_artifact_evidence"])):
            collection_plan = collection["plan"]
            plan_path = Path(collection_plan["command"][-1])
            receipt_path = Path(collection["path"])
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            assert file_sha(plan_path) == collection["plan_sha256"]
            assert json.loads(plan_path.read_text(encoding="utf-8")) == collection_plan
            assert file_sha(receipt_path) == collection["sha256"]
            assert receipt["status"] == "COMPLETE" and receipt["smoke"] is False
            assert receipt["plan_sha256"] == collection["plan_sha256"]
            assert receipt["benchmark_evidence"] is False
            assert receipt["executor_sha256"] == plan["executor_binding"]["inputs"][
                "executor"]["sha256"]
            assert collection_plan["host"] == plan["host"]
            assert collection_plan["variant"] == plan["variant"]
            assert collection_plan["c2_task_id"] == plan["c2_task_id"]
            assert collection_plan["executor_binding"] == plan["executor_binding"]
            assert collection_plan["shard"] == shard
            assert collection_plan["shards"] == 4
            assert collection_plan["label_stride"] == plan["label_stride"]
            assert collection_plan["cost_weight"] == plan["cost_weight"]
            assert evidence["plan_path"] == str(plan_path)
            assert evidence["plan_sha256"] == collection["plan_sha256"]
            assert evidence["artifact_hashes"] == receipt["artifacts"]

            expected_names = [
                row["name"] for row in list(self.sequences.values())[shard::4]
            ]
            assert receipt["sequence_names"] == expected_names
            artifact_names = []
            actual_records = 0
            for filename, digest in receipt["artifacts"].items():
                artifact_path = receipt_path.parent / filename
                assert artifact_path.name == filename
                assert file_sha(artifact_path) == digest
                saved = torch.load(artifact_path, map_location="cpu")
                name = saved["sequence"]
                assert name in expected_names and name not in artifact_names
                assert name not in seen_names
                assert saved["plan_sha256"] == collection["plan_sha256"]
                sequence = self.sequences[name]
                valid = set(sequence["valid_frames"])
                expected_frames = [
                    frame for frame in range(1, len(sequence["boxes"]))
                    if frame in valid and frame % plan["label_stride"] == 0
                ]
                records = saved["records"]
                assert [record["frame"] for record in records] == expected_frames
                assert saved["tracked_frames"] == len(sequence["boxes"]) - 1
                artifact_names.append(name)
                seen_names.add(name)
                file_index = len(self.files)
                self.files.append((name, artifact_path))
                self.index.extend(
                    (file_index, index) for index in range(len(records)))
                actual_records += len(records)
            assert artifact_names == expected_names
            assert actual_records == receipt["records"]
            assert actual_records == evidence["records"]
            assert receipt["tracked_frames"] == evidence["tracked_frames"]

        assert len(seen_names) == 882
        assert len(self.index) == acceptance["records"]
        assert self.index
        self.cached_index = None
        self.cached = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        file_index, record_index = self.index[index]
        name, path = self.files[file_index]
        if self.cached_index != file_index:
            self.cached = torch.load(path, map_location="cpu")
            self.cached_index = file_index
        record = self.cached["records"][record_index]
        templates, search, history = rebuild_state(
            self.sequences[name], record)
        return (
            templates,
            torch.from_numpy(np.asarray(search, dtype=np.float32)),
            history,
            torch.from_numpy(np.asarray(record["target"], dtype=np.float32)),
        )


def fit(model, loader, plan, run, window):
    before = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    groups = configure_policy_refresh(model)
    assert len(groups) == 2
    groups[0]["lr"] = plan["preview_lr"]
    groups[1]["lr"] = plan["quality_gain_lr"]
    parameters = [parameter for group in groups for parameter in group["params"]]
    optimizer = torch.optim.AdamW(
        groups, weight_decay=plan["weight_decay"])
    step = 0
    artifacts = {}
    device = next(model.parameters()).device
    with (run / "steps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, ["epoch", "step", *LOSS_FIELDS])
        writer.writeheader()
        for epoch in range(1, plan["epochs"] + 1):
            for batch in loader:
                batch = [
                    tensor.to(device, non_blocking=True) for tensor in batch
                ]
                result = policy_refresh_loss(model, *batch, window)
                optimizer.zero_grad(set_to_none=True)
                result["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, plan["gradient_clip"])
                assert torch.isfinite(norm)
                optimizer.step()
                step += 1
                writer.writerow(dict(
                    epoch=epoch, step=step,
                    **{key: float(result[key]) for key in LOSS_FIELDS}))
                if step % 25 == 0:
                    handle.flush()
                    print(epoch, step, float(result["loss"]), flush=True)

            state = model.state_dict()
            assert all(torch.isfinite(value).all() for value in state.values())
            for key, value in before.items():
                if not key.startswith(POLICY_PREFIXES):
                    assert torch.equal(value, state[key].detach().cpu()), key
            for prefix in POLICY_PREFIXES:
                assert any(
                    not torch.equal(value, state[key].detach().cpu())
                    for key, value in before.items() if key.startswith(prefix)
                ), prefix

            if epoch > plan["epochs"] - 5:
                path = run / ("epoch_{:03d}.pth".format(epoch))
                torch.save(dict(
                    net=state, optimizer=optimizer.state_dict(), epoch=epoch,
                    step=step, config=plan,
                    executor_sha256=plan["executor_binding"]["inputs"][
                        "executor"]["sha256"]), path)
                saved = torch.load(path, map_location="cpu")
                assert all(torch.equal(value.detach().cpu(), saved["net"][key])
                           for key, value in state.items())
                expected = model.predict(
                    batch[0][:1], batch[1][:1], batch[2][:1],
                    plan["cost_weight"], window)
                model.load_state_dict(saved["net"], strict=True)
                actual = model.predict(
                    batch[0][:1], batch[1][:1], batch[2][:1],
                    plan["cost_weight"], window)
                for key in ("modality", "action"):
                    assert expected[key] == actual[key]
                for key in ("box", "gains"):
                    assert torch.equal(expected[key], actual[key])
                artifacts[path.name] = file_sha(path)
    assert step == len(loader) * plan["epochs"]
    return dict(
        status="COMPLETE", stage=plan["stage"], smoke=False,
        epochs=plan["epochs"], steps=step, records=len(loader.dataset),
        retained_epochs=list(range(plan["epochs"] - 4, plan["epochs"] + 1)),
        artifacts=artifacts, executor_preserved=True,
        all_policy_modules_changed=True, strict_reload=True,
        collection_acceptance=plan["collection_acceptance"],
        executor_binding=plan["executor_binding"],
        executor_sha256=plan["executor_binding"]["inputs"]["executor"]["sha256"],
        benchmark_evidence=False, test_enabled=False, dev_enabled=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    assert Path.cwd().resolve() == Path(plan["root"]).resolve()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["gpu"]
    acceptance = validate_collection_acceptance(plan)
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dataset = StateDataset(plan, acceptance)
    assert len(dataset) == acceptance["records"]
    loader = DataLoader(
        dataset, batch_size=plan["batch_size"], shuffle=False,
        num_workers=plan["workers"], pin_memory=True)
    device = torch.device("cuda")
    model = load_completed_executor(plan["executor_binding"], device=device)
    run = Path(plan["run"])
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(plan, indent=2) + "\n")
    (run / "precision.json").write_text(json.dumps(dict(
        matmul_allow_tf32=False, cudnn_allow_tf32=False,
        cudnn_benchmark=False)) + "\n")
    window = hann2d(torch.tensor([16, 16]), centered=True).to(device)
    result = fit(model, loader, plan, run, window)
    result["plan_sha256"] = file_sha(args.plan)
    (run / "COMPLETE.json").write_text(
        json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
