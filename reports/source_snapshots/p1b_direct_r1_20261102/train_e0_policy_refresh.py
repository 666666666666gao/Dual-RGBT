"""Offline direct-E0 P1B fit with freshly recomputed counterfactual targets."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_rgbt.e0_policy_refresh import (
    POLICY_PREFIXES, configure_policy_refresh, file_sha, load_completed_executor,
    load_plan, policy_refresh_loss, rebuild_state, train_sequences)
from lib.test.utils.hann import hann2d


class StateDataset(Dataset):
    def __init__(self, plan, sequences):
        self.sequences = {s["name"]: s for s in sequences}
        self.files, self.index = [], []
        names, collection_plans = [], []
        for collection in plan["collections"]:
            terminal_path = Path(collection["path"])
            assert file_sha(terminal_path) == collection["sha256"]
            receipt = json.loads(terminal_path.read_text())
            assert receipt["status"] == "COMPLETE"
            assert receipt["smoke"] == plan["smoke"]
            assert receipt["executor_sha256"] == plan["inputs"]["executor"]["sha256"]
            collection_plan_path = terminal_path.parent / "config.json"
            collection_plan = json.loads(collection_plan_path.read_text())
            # config.json is pretty-printed, so bind the original plan separately.
            assert receipt["plan_sha256"] == collection["plan_sha256"]
            assert collection_plan == collection["plan"]
            assert collection_plan["mode"] == "collect"
            for key in ("inputs", "sources", "seed", "cost_weight", "label_stride",
                        "round", "branch"):
                assert collection_plan[key] == plan[key], key
            collection_plans.append(collection_plan)
            names.extend(receipt["sequence_names"])
            actual_records, actual_frames, file_names = 0, 0, []
            for filename, digest in receipt["artifacts"].items():
                path = terminal_path.parent / filename
                assert file_sha(path) == digest
                saved = torch.load(path, map_location="cpu")
                assert saved["plan_sha256"] == receipt["plan_sha256"]
                name = saved["sequence"]
                assert name in self.sequences and name not in file_names
                file_names.append(name)
                expected = self.sequences[name]
                last = min(len(expected["boxes"]), 11) if plan["smoke"] else len(expected["boxes"])
                valid = set(expected["valid_frames"])
                frames = [f for f in range(1, last)
                          if f in valid and f % plan["label_stride"] == 0]
                assert [r["frame"] for r in saved["records"]] == frames
                assert saved["tracked_frames"] == last - 1
                actual_records += len(saved["records"])
                actual_frames += saved["tracked_frames"]
                file_index = len(self.files)
                self.files.append((name, path))
                self.index.extend((file_index, i) for i in range(len(saved["records"])))
            assert set(file_names) == set(receipt["sequence_names"])
            assert receipt["records"] == actual_records
            assert receipt["tracked_frames"] == actual_frames
        assert len(names) == len(set(names))
        if not plan["smoke"]:
            assert set(names) == set(self.sequences)
            shards = collection_plans[0]["shards"]
            assert sorted(p["shard"] for p in collection_plans) == list(range(shards))
            assert all(p["shards"] == shards for p in collection_plans)
        assert self.index
        self.cached_index, self.cached = None, None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        file_index, record_index = self.index[index]
        name, path = self.files[file_index]
        if self.cached_index != file_index:
            self.cached = torch.load(path, map_location="cpu")
            self.cached_index = file_index
        return rebuild_state(self.sequences[name], self.cached["records"][record_index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan, "fit")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["gpu"]
    assert plan["epochs"] >= 5 and plan["batch_size"] > 0
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    torch.backends.cudnn.benchmark = False
    sequences = train_sequences(plan["inputs"]["manifest"]["path"])
    dataset = StateDataset(plan, sequences)
    loader = DataLoader(dataset, batch_size=plan["batch_size"], shuffle=False,
                        num_workers=plan["workers"], pin_memory=True)
    run = Path(plan["run"])
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(plan, indent=2) + "\n")
    model = load_completed_executor(plan)
    before = {key: value.cpu().clone() for key, value in model.state_dict().items()}
    groups = configure_policy_refresh(model)
    parameters = [p for group in groups for p in group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    step, artifacts = 0, {}
    with (run / "steps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, [
            "epoch", "step", "loss", "quality_loss", "gain_loss", "spatial_loss"])
        writer.writeheader()
        epochs = 1 if plan["smoke"] else plan["epochs"]
        for epoch in range(1, epochs + 1):
            for batch in loader:
                batch = [tensor.cuda(non_blocking=True) for tensor in batch]
                result = policy_refresh_loss(model, *batch, window)
                optimizer.zero_grad(set_to_none=True)
                result["loss"].backward()
                if plan["smoke"]:
                    for module in (model.preview, model.quality, model.gain):
                        assert any(p.grad is not None and p.grad.abs().sum() > 0
                                   for p in module.parameters())
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                assert torch.isfinite(norm)
                optimizer.step()
                step += 1
                writer.writerow(dict(epoch=epoch, step=step, **{
                    key: float(result[key]) for key in (
                        "loss", "quality_loss", "gain_loss", "spatial_loss")}))
                if step % 25 == 0 or plan["smoke"]:
                    handle.flush()
                    print(epoch, step, float(result["loss"]), flush=True)
                if plan["smoke"] and step == 2:
                    break
            after = model.state_dict()
            for key, tensor in before.items():
                if not key.startswith(POLICY_PREFIXES):
                    assert torch.equal(tensor, after[key].cpu()), key
            for prefix in POLICY_PREFIXES:
                assert any(not torch.equal(tensor, after[key].cpu())
                           for key, tensor in before.items() if key.startswith(prefix)), prefix
            if plan["smoke"] or epoch > epochs - 5:
                path = run / ("epoch_{:03d}.pth".format(epoch))
                torch.save(dict(net=after, optimizer=optimizer.state_dict(),
                                epoch=epoch, step=step, config=plan,
                                executor_sha256=plan["inputs"]["executor"]["sha256"]), path)
                saved = torch.load(path, map_location="cpu")
                assert all(torch.equal(tensor.cpu(), saved["net"][key])
                           for key, tensor in after.items())
                expected = model.predict(batch[0][:1], batch[1][:1], batch[2][:1],
                                         plan["cost_weight"], window)
                model.load_state_dict(saved["net"], strict=True)
                actual = model.predict(batch[0][:1], batch[1][:1], batch[2][:1],
                                       plan["cost_weight"], window)
                assert expected["modality"] == actual["modality"]
                assert expected["action"] == actual["action"]
                assert torch.equal(expected["box"], actual["box"])
                assert torch.equal(expected["gains"], actual["gains"])
                artifacts[path.name] = file_sha(path)
    if plan["smoke"]:
        assert step == 2
    else:
        assert step == len(loader) * epochs
    (run / "COMPLETE.json").write_text(json.dumps(dict(
        status="COMPLETE", stage="P1B_direct_E0_fit", smoke=plan["smoke"],
        plan_sha256=file_sha(args.plan), epochs=epochs, steps=step,
        records=len(dataset), artifacts=artifacts, executor_preserved=True,
        all_policy_modules_changed=True, strict_reload=True,
        closed_loop_round=plan["round"], branch=plan["branch"],
        sources=plan["sources"],
        executor_sha256=plan["inputs"]["executor"]["sha256"],
        benchmark_evidence=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
