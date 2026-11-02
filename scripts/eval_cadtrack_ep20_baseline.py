"""Reproduce the original stripped-CADTrack tracker with the supplied ep20."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from benchmark_inputs import groundtruth, load_manifest, sha
from dual_rgbt.data import read_pair


STAGE = "CADTrack_ep20_original_tracker_evaluation"
BINDINGS = {"launch_allowed", "source_review_sha256", "smoke_receipt_sha256",
            "smoke_native_sha256"}


def admission(plan_path, benchmark, smoke):
    plan = json.loads(plan_path.read_bytes())
    assert plan["stage"] == STAGE and plan["target_root"] == str(ROOT)
    assert plan["launch_allowed"] is True and plan["training_allowed"] is False
    assert plan["test_eligible"] is False
    assert plan["checkpoint_sha256"] == sha(plan["checkpoint"])
    assert plan["checkpoint_sha256"] == (
        "d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8")
    review_path = ROOT / plan["source_review"]
    assert sha(review_path) == plan["source_review_sha256"]
    review = json.loads(review_path.read_bytes())
    assert review["stage"] == STAGE and review["status"] == "SOURCE_PASS"
    assert review["blocking_issues"] == 0 and review["target_root"] == str(ROOT)
    assert review["source_sha256"] == plan["source_sha256"]
    assert all(sha(ROOT / name) == digest for name, digest in plan["source_sha256"].items())
    draft_path = ROOT / plan["reviewed_draft"]
    assert review["reviewed_drafts"][plan["reviewed_draft"]] == sha(draft_path)
    draft = json.loads(draft_path.read_bytes())
    assert draft["launch_allowed"] is False and set(draft) == set(plan)
    assert {k: v for k, v in draft.items() if k not in BINDINGS} == {
        k: v for k, v in plan.items() if k not in BINDINGS}
    job = plan["smoke"] if smoke else plan["jobs"][benchmark]
    manifest_path = Path(job["manifest"])
    assert sha(manifest_path) == job["manifest_sha256"]
    manifest = load_manifest(manifest_path)
    assert manifest["benchmark"] == benchmark
    assert manifest["split"] == ("train_smoke" if smoke else "test")
    if not smoke:
        assert manifest["sequence_count"] == {"LasHeR": 245, "RGBT210": 210,
                                             "RGBT234": 234}[benchmark]
        receipt = Path(plan["smoke"]["run"]) / "SMOKE.json"
        smoke_config_path = Path(plan["smoke"]["run"]) / "config.json"
        native = ROOT / "runs/pipeline_tasks" / plan["smoke"]["task"] / "status.json"
        assert sha(receipt) == plan["smoke_receipt_sha256"]
        assert sha(native) == plan["smoke_native_sha256"]
        success = json.loads(receipt.read_bytes())
        smoke_config = json.loads(smoke_config_path.read_bytes())
        status = json.loads(native.read_bytes())
        assert success["config_sha256"] == sha(smoke_config_path)
        assert success["task"] == smoke_config["task"] == status["task"] == plan["smoke"]["task"]
        assert status["command"] == smoke_config["native_command"]
        assert status["physical_gpu"] == smoke_config["physical_gpu"] == plan["smoke"]["gpu"]
        assert success["stage"] == STAGE and success["checkpoint_sha256"] == plan["checkpoint_sha256"]
        assert success["frames"] == 11 and success["state_unchanged"] is True
        assert success["smoke_only"] is True
        assert success["exploratory_test"] is False and success["test_eligible"] is False
        assert status["state"] == "COMPLETE" and status["exit_code"] == 0
    native_path = ROOT / "runs/pipeline_tasks" / job["task"] / "status.json"
    native = json.loads(native_path.read_bytes())
    assert native["pid"] == os.getppid() and native["state"] == "RUNNING"
    assert native["command"] == [sys.executable] + sys.argv
    assert native["physical_gpu"] == job["gpu"] == os.environ["CUDA_VISIBLE_DEVICES"]
    return plan, job, manifest, native_path, native


def original_tracker(plan):
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.test.tracker.cadtrack import CADTrack
    from lib.test.utils import TrackerParams
    update_config_from_file(str(ROOT / "experiments/cadtrack/cadtrack.yaml"))
    params = TrackerParams()
    params.cfg = cfg
    params.template_factor, params.template_size = cfg.TEST.TEMPLATE_FACTOR, cfg.TEST.TEMPLATE_SIZE
    params.search_factor, params.search_size = cfg.TEST.SEARCH_FACTOR, cfg.TEST.SEARCH_SIZE
    params.checkpoint = plan["checkpoint"]
    params.save_all_boxes, params.debug = False, 0
    assert (params.search_size, params.template_size, cfg.DATA.TEMPLATE.NUMBER) == (256, 128, 2)
    assert (cfg.TEST.UPDATE_INTERVALS, cfg.TEST.UPDATE_THRESHOLD) == (5, .65)
    tracker = CADTrack(params)
    tracker.network.eval().requires_grad_(False)
    return tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--benchmark", choices=("LasHeR", "RGBT210", "RGBT234"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan, job, manifest, native_path, native = admission(plan_path, args.benchmark, args.smoke)
    torch.manual_seed(2027)
    np.random.seed(2027)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    tracker = original_tracker(plan)
    before = {k: v.cpu().clone() for k, v in tracker.network.state_dict().items()}
    run = Path(job["run"])
    run.mkdir(parents=True, exist_ok=False)
    for name in ("predictions", "groundtruth", "traces"):
        (run / name).mkdir()
    selected = manifest["sequences"][:1] if args.smoke else manifest["sequences"]
    config = dict(stage=STAGE, task=job["task"], checkpoint=plan["checkpoint"],
                  checkpoint_sha256=plan["checkpoint_sha256"], manifest=job["manifest"],
                  manifest_sha256=job["manifest_sha256"], benchmark=args.benchmark,
                  split=manifest["split"], split_sequences=manifest["sequence_count"],
                  split_frames=manifest["frame_count"], sequences=[s["name"] for s in selected],
                  source_sha256=plan["source_sha256"], source_review=plan["source_review"],
                  source_review_sha256=plan["source_review_sha256"],
                  plan=args.plan, plan_sha256=sha(plan_path), physical_gpu=job["gpu"],
                  native_status=str(native_path), native_command=native["command"],
                  total_parameters=sum(p.numel() for p in tracker.network.parameters()),
                  shards=1, shard=0, smoke_frames=11 if args.smoke else 0,
                  cost_weight=0., exploratory_test=not args.smoke, test_eligible=False,
                  gpu_name=torch.cuda.get_device_name(0), precision="FP32; TF32 disabled",
                  timing_scope="Original tracker. Decode and initialization included; model "
                               "load excluded. Concurrent host load; not isolated throughput.")
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    progress, timings = [], []
    with torch.no_grad():
        for sequence in selected:
            count = 11 if args.smoke else len(sequence["rgb"])
            predictions = [np.asarray(sequence["initial_box"], dtype=np.float32)]
            torch.cuda.synchronize()
            start = time.perf_counter()
            tracker.initialize(read_pair(sequence, 0), dict(init_bbox=sequence["initial_box"]))
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            with (run / "traces" / (sequence["name"] + ".csv")).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, ["frame", "seconds_including_decode"])
                writer.writeheader()
                for frame in range(1, count):
                    start = time.perf_counter()
                    result = tracker.track(read_pair(sequence, frame))
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    box = np.asarray(result["target_bbox"], dtype=np.float32)
                    assert np.isfinite(box).all() and (box[2:] > 0).all()
                    predictions.append(box.copy())
                    timings.append(elapsed)
                    writer.writerow(dict(frame=frame, seconds_including_decode=elapsed))
            np.savetxt(run / "predictions" / (sequence["name"] + ".txt"),
                       predictions, fmt="%.6f", delimiter=",")
            visible, infrared = groundtruth(sequence, args.benchmark)
            if args.benchmark == "LasHeR":
                np.savetxt(run / "groundtruth" / (sequence["name"] + ".txt"),
                           visible[:count], fmt="%.9g", delimiter=",")
            else:
                folder = run / "groundtruth" / sequence["name"]
                folder.mkdir()
                for modality, boxes in (("visible", visible), ("infrared", infrared)):
                    np.savetxt(folder / (modality + ".txt"), boxes[:count], fmt="%.9g", delimiter=",")
            progress.append(dict(sequence=sequence["name"], frames=count))
            (run / "progress.json").write_text(json.dumps(progress) + "\n")
            print(json.dumps(progress[-1]), flush=True)
    assert all(torch.equal(v.cpu(), before[k]) for k, v in tracker.network.state_dict().items())
    assert all(p.grad is None for p in tracker.network.parameters())
    frames = sum(s["frames"] for s in progress)
    assert frames == (11 if args.smoke else manifest["frame_count"])
    timing = np.asarray(timings)
    assert timing.shape == (frames,) and np.isfinite(timing).all() and (timing > 0).all()
    np.save(run / "frame_seconds.npy", timing)
    receipt = dict(stage=STAGE, task=job["task"], checkpoint_sha256=plan["checkpoint_sha256"],
                   config_sha256=sha(run / "config.json"), sequences=len(progress), frames=frames,
                   seconds=float(timing.sum()), optimizer_updates=0, backward_calls=0,
                   state_unchanged=True, smoke_only=args.smoke,
                   exploratory_test=not args.smoke, test_eligible=False)
    (run / ("SMOKE.json" if args.smoke else "COMPLETE.json")).write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
