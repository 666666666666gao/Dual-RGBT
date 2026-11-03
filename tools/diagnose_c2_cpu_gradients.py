"""Inspect gradient connectivity in the preserved R1 CPU working copy."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

root = Path("/data2/gaob/Dual-RGBT/c2_cpu_check_r1_20261103")
sys.path.insert(0, str(root))
from dual_rgbt.c2_executor import C2DualRGBT, executor_parameter_groups
from dual_rgbt.data import LasHeRPairsV2
from dual_rgbt.losses import executor_loss, localization_loss
from dual_rgbt.model import DualRGBT
from lib.test.utils.hann import hann2d

assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
torch.manual_seed(2027)
invocation = json.loads((root / "invocation.json").read_text())
for name, digest in invocation["files"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
inputs = invocation["inputs"]
baseline = DualRGBT(inputs["factory"]["path"])
baseline.load_state_dict(
    torch.load(inputs["executor"]["path"], map_location="cpu")["net"], strict=True)
del baseline
for missing, lora in (("zero", "none"), ("learned_film", "none"),
                      ("learned_film", "shared"), ("learned_film", "routed")):
    model = C2DualRGBT.from_e0(
        inputs["factory"]["path"], inputs["executor"]["path"], missing, lora)
    executor_parameter_groups(model)
    if lora != "routed":
        del model
executor_parameter_groups(model)
model.train()
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()
item = LasHeRPairsV2(inputs["manifest"]["path"], "train", 1, 2027)[0]
templates, search, target = [
    item[key].unsqueeze(0) for key in ("templates", "search", "target")]
drop_path_outputs = [[] for block in model.backbone.blocks]
handles = []
for index, block in enumerate(model.backbone.blocks):
    def capture(module, inputs, output, index=index):
        drop_path_outputs[index].append(int(torch.count_nonzero(output)))
    handles.append(block.drop_path.register_forward_hook(capture))
output = model.forward_action(
    templates, search, 0, 4, window=hann2d(torch.tensor([16, 16]), centered=True))
for handle in handles:
    handle.remove()
primary_loss, primary_iou = localization_loss(output["primary"]["box"], target)
refined_loss, refined_iou = localization_loss(output["box"], target)
loss = executor_loss(output, target)
assert torch.isfinite(loss)
loss.backward()
gradients = []
for name, module in model.named_modules():
    if module in model.c2_adapters:
        gradients.append(dict(
            name=name,
            a_l1=[float(bank.abs().sum()) for bank in module.a.grad],
            b_l1=[float(bank.abs().sum()) for bank in module.b.grad]))
assert not torch.cuda.is_initialized()
report = dict(
    stage="P2_C2_R1_CPU_gradient_diagnostic", train_sequence=item["sequence"],
    train_frame=item["frame"], loss=float(loss),
    primary_localization=float(primary_loss), refined_localization=float(refined_loss),
    primary_iou=float(primary_iou), refined_iou=float(refined_iou),
    gradients=gradients,
    drop_path_nonzero_output_counts=drop_path_outputs,
    drop_path_call_order=["primary_attention", "primary_mlp",
                          "auxiliary_attention", "auxiliary_mlp"],
    drop_path_rates=[
        0 if isinstance(block.drop_path, torch.nn.Identity) else block.drop_path.drop_prob
        for block in model.backbone.blocks],
    missing_l1=float(model.missing.embedding.grad.abs().sum()),
    film_l1=[{
        name: None if parameter.grad is None else float(parameter.grad.abs().sum())
        for name, parameter in layer.named_parameters()}
        for layer in model.missing.film],
    optimizer_updates=0, cuda_initialized=False, benchmark_evidence=False)
path = root / "gradient_diagnostic.json"
assert not path.exists()
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
