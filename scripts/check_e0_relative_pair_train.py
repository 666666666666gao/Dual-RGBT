"""Fixed independent E0+M1 / relative-M2 paired TRAIN trajectories; no updates."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tools")]
from relative_e0_pair_contract import PARENT, admission, immutable_inputs, native_command, sha
import dual_rgbt.data as data
from profile_crop_api import six_channel_crop
data.crop_pair = six_channel_crop
from dual_rgbt.data import read_pair
from dual_rgbt.e0_m1_guard import M1TemplateGuard, load_reliability
from dual_rgbt.e0_m1_relative_m2_online import E0M1RelativeM2Tracker
from dual_rgbt.e0_relative_identity import E0RelativeIdentity
from dual_rgbt.model import DualRGBT
from dual_rgbt.tracker import Tracker


def state_digest(base):
    digest = hashlib.sha256()
    for value in (base.state, base.templates.detach().cpu().numpy(),
                  base.history.detach().cpu().numpy(),
                  np.asarray([base.frame, base.since_query], np.int64)):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def iou(box, target):
    box, target = np.asarray(box, np.float64), np.asarray(target, np.float64)
    intersection = np.maximum(
        0, np.minimum(box[:2] + box[2:], target[:2] + target[2:]) -
        np.maximum(box[:2], target[:2])).prod()
    return float(intersection / (box[2:].prod() + target[2:].prod() - intersection))


def model_and_m1(plan):
    model = DualRGBT(str(ROOT / plan["factory_checkpoint"]))
    model.load_state_dict(torch.load(
        ROOT / plan["parent_checkpoint"], map_location="cpu")["net"], strict=True)
    model.cuda().eval().requires_grad_(False)
    m1, mean, scale = load_reliability(ROOT / plan["m1_checkpoint"], PARENT, "cuda")
    return model, m1, mean, scale


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    plan_path = Path(args.plan)
    assert plan_path == ROOT / "reports/e0_relative_pair_train_r1_20261102/plan.json"
    plan = json.loads(plan_path.read_bytes())
    plan_sha = sha(plan_path)
    rows, trained = admission(ROOT, plan)
    native_path = ROOT / "runs/pipeline_tasks" / plan["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["state"] == "RUNNING" and native["pid"] == os.getppid()
    assert native["command"] == [sys.executable] + sys.argv == native_command(ROOT, plan_path)
    assert native["task"] == plan["task"] and native["hostname"] == socket.gethostname()
    assert native["physical_gpu"] == os.environ["CUDA_VISIBLE_DEVICES"] == plan["assigned_gpu"]
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["OPENCV_FOR_THREADS_NUM"] == "1"
    torch.manual_seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert Tracker.__module__ == "dual_rgbt.tracker"
    import dual_rgbt.tracker as tracker_module
    assert tracker_module.crop_pair is six_channel_crop
    control_model, control_m1, mean, scale = model_and_m1(plan)
    control_base = Tracker(control_model, plan["cost_weight"])
    control = M1TemplateGuard(control_base, control_m1, mean, scale, commit_guard=True)
    model, m1, mean, scale = model_and_m1(plan)
    relative = E0RelativeIdentity().cuda().eval().requires_grad_(False)
    checkpoint = torch.load(ROOT / plan["relative_checkpoint"], map_location="cpu")
    assert checkpoint["epoch"] == trained["best_qualifying_epoch"]
    relative.load_state_dict(checkpoint["net"], strict=True)
    intervention = E0M1RelativeM2Tracker(model, plan["cost_weight"], relative, m1, mean, scale)
    modules = dict(control_e0=control_model, control_m1=control_m1,
                   intervention_e0=model, intervention_m1=m1, intervention_m2=relative)
    before = {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
              for name, module in modules.items()}
    run = ROOT / plan["run"]
    run.mkdir(parents=True, exist_ok=False)
    (run / "traces").mkdir()
    config = dict(plan=plan, plan_sha256=plan_sha, native_command=native["command"],
                  training_acceptance_sha256=plan["training_acceptance_sha256"],
                  timing_scope="Synchronized complete initialization and each arm track; decode "
                               "shared/separately recorded, excludes GT/hashing/model load. Concurrent.",
                  arm_parameters=dict(
                      e0_m1_control=sum(p.numel() for m in (control_model, control_m1) for p in m.parameters()),
                      e0_m1_relative_m2=sum(p.numel() for m in (model, m1, relative) for p in m.parameters())))
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    fields = ["frame", "valid", "control_iou", "intervention_iou",
              "control_x", "control_y", "control_w", "control_h",
              "intervention_x", "intervention_y", "intervention_w", "intervention_h",
              "control_m1_state", "intervention_m1_state", "m2_invoked", "m2_class_index",
              "control_template_write_proposed", "control_template_write_blocked",
              "intervention_template_write_proposed", "intervention_template_write_blocked",
              "m2_keep_score", "m2_selected_score", "current_encoder_calls", "candidate_encoder_calls",
              "control_modality", "control_action", "control_confidence", "control_primary_tokens",
              "control_auxiliary_tokens", "intervention_modality", "intervention_action",
              "intervention_confidence", "intervention_primary_tokens", "intervention_auxiliary_tokens",
              "control_state_sha256", "intervention_state_sha256",
              "control_history", "intervention_history", "control_since_query",
              "intervention_since_query",
              "control_seconds", "intervention_seconds", "decode_seconds"]
    timings, progress = [], []
    for row in rows:
        count, valid = len(row["rgb"]), set(row["valid_frames"])
        intervention.set_sequence(row["valid_frames"], count)
        started = time.perf_counter()
        image = read_pair(row, 0)
        decode = time.perf_counter() - started
        started = time.perf_counter()
        control.initialize(image, row["boxes"][0])
        torch.cuda.synchronize()
        control_seconds = time.perf_counter() - started
        started = time.perf_counter()
        intervention.initialize(image, row["boxes"][0])
        torch.cuda.synchronize()
        intervention_seconds = time.perf_counter() - started
        assert state_digest(control_base) == state_digest(intervention.base)
        timings.append([control_seconds, intervention_seconds, decode])
        first_replacement, first_divergence, invocations, replacements = None, None, 0, 0
        with (run / "traces" / (row["name"] + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            for frame in range(1, count):
                started = time.perf_counter()
                image = read_pair(row, frame)
                decode = time.perf_counter() - started
                started = time.perf_counter()
                control_box, control_route = control.track(image)
                torch.cuda.synchronize()
                control_seconds = time.perf_counter() - started
                started = time.perf_counter()
                output, route = intervention.track(image)
                torch.cuda.synchronize()
                intervention_seconds = time.perf_counter() - started
                decision = intervention.last_m2
                invocations += int(decision["invoked"])
                replacements += int(decision["class_index"] > 0)
                if first_replacement is None and decision["class_index"] > 0:
                    first_replacement = frame
                control_state, intervention_state = state_digest(control_base), state_digest(intervention.base)
                if first_replacement is None:
                    assert control_state == intervention_state
                    assert control.record == intervention.guard.record
                if control_state != intervention_state and first_divergence is None:
                    first_divergence = frame
                assert control.observer.calls == intervention.guard.observer.calls == 1
                assert control_base.frame == intervention.base.frame == frame
                assert np.isfinite(control_box).all() and np.isfinite(output).all()
                # Read annotations only after both independent causal decisions.
                target = row["boxes"][frame] if frame in valid else None
                values = dict(
                    frame=frame, valid=int(frame in valid),
                    control_iou=iou(control_box, target) if target is not None else "",
                    intervention_iou=iou(output, target) if target is not None else "",
                    control_m1_state=control.record["m1_state"],
                    intervention_m1_state=intervention.guard.record["m1_state"],
                    m2_invoked=int(decision["invoked"]), m2_class_index=decision["class_index"],
                    m2_keep_score=decision["keep_score"], m2_selected_score=decision["selected_score"],
                    current_encoder_calls=decision["current_encoder_calls"],
                    candidate_encoder_calls=decision["candidate_encoder_calls"],
                    control_state_sha256=control_state, intervention_state_sha256=intervention_state,
                    control_history=json.dumps(control_base.history.cpu().tolist()[0]),
                    intervention_history=json.dumps(intervention.base.history.cpu().tolist()[0]),
                    control_since_query=control_base.since_query,
                    intervention_since_query=intervention.base.since_query,
                    control_seconds=control_seconds, intervention_seconds=intervention_seconds,
                    decode_seconds=decode)
                for prefix, record in (("control", control.record),
                                        ("intervention", intervention.guard.record)):
                    values.update({prefix + "_" + key: record[key] for key in (
                        "template_write_proposed", "template_write_blocked")})
                for prefix, box, routed in (("control", control_box, control_route),
                                            ("intervention", output, route)):
                    values.update({prefix + "_" + key: float(value) for key, value in zip("xywh", box)})
                    values.update({prefix + "_" + key: routed[key] for key in (
                        "modality", "action", "confidence", "primary_tokens", "auxiliary_tokens")})
                writer.writerow(values)
                timings.append([control_seconds, intervention_seconds, decode])
            handle.flush()
        progress.append(dict(sequence=row["name"], frames=count, invocations=invocations,
                             replacements=replacements, first_replacement=first_replacement,
                             first_state_divergence=first_divergence))
        (run / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(json.dumps(progress[-1]), flush=True)
    control.close()
    intervention.close()
    assert control_model.predict == control.observer.original
    assert model.predict == intervention.guard.observer.original
    assert all(torch.equal(value.detach().cpu(), before[name][key])
               for name, module in modules.items() for key, value in module.state_dict().items())
    assert all(p.grad is None for module in modules.values() for p in module.parameters())
    immutable_inputs(ROOT, plan)
    assert sha(plan_path) == plan_sha
    timing = np.asarray(timings, np.float64)
    assert timing.shape == (3076, 3) and np.isfinite(timing).all() and (timing > 0).all()
    np.save(run / "frame_seconds.npy", timing)
    receipt = dict(
        status="RELATIVE_M2_PAIRED_TRAIN_COMPLETE", task=plan["task"], split="train",
        frames=3076, sequences=8, test_eligible=False, benchmark_allowed=False,
        scientific_result_audit_accepted=False, optimizer_updates=0, backward_calls=0,
        plan_sha256=plan_sha, model_weights_unchanged=True, predictor_hooks_restored=True,
        independent_model_instances=True, m2_invocations=sum(x["invocations"] for x in progress),
        replacements=sum(x["replacements"] for x in progress),
        initialization_anchor_encoders=16,
        scope="Fixed paired TRAIN trajectories only. Zero intervention is not recovery evidence. "
              "No automatic benchmark admission or independent final validation.",
        artifacts_sha256={p.relative_to(run).as_posix(): sha(p) for p in run.rglob("*") if p.is_file()})
    (run / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
