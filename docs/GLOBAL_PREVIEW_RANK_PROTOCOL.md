# Global preview ranking warmup

2026-10-14. SOURCE_PENDING; actual GPU admission and formal run not yet executed.

The completed full B and frozen content diagnostic show that correctly located auxiliary pixels can help, while learned external queries cover only 11/239 target centres for local queries and 49/239 for wide queries in the native main-outside subgroup. The geometry-only nearest-GT-cell diagnostic covers 239/239 at both existing budgets. This motivates fitting the existing preview ranking before full C. These are train diagnostics, not final scores.

The existing localization loss averages sigmoid BCE over a 32x32 map with a small Gaussian target. That establishes a loss-design hypothesis, not proof that imbalance caused failed proposals. Existing template descriptors already average central 4x4 cells and use the initial template. No new descriptor, resolution, region size, model parameter or inference module is introduced.

Train only the six shared preview CNN tensors from the completed full B checkpoint. Normalize the existing Gaussian target to unit mass over valid non-padding global cells, and optimize cross entropy against spatial log-softmax of the existing ten-times-cosine scores. Existing image geometry and pre-encoding query selection remain unchanged. The full executor, quality/gain controllers and all other 259 state tensors stay bit-identical.

The frozen plan specifies seed2027, train882, five epochs of 10,000 pairs, batch8, AdamW lr1e-4/weight_decay1e-4 and 6,250 updates. Sanity uses 16 real pairs and two optimizer updates; its checkpoint must not initialize the formal run. Before and after fitting, evaluate the same 1,752 train states using the original auxiliary role, initial-template descriptor and ROI side recorded in the frozen content CSV. The before peak coordinates must reproduce the original native proposal. Locations and individual rows remain private; export aggregates only.

Real admission must verify finite nonzero preview gradients, six changed tensors, 259 unchanged tensors, Adam update counts and seven-field strict save/reload predictions. Both formal and sanity checkpoints are tagged `localize`; neither is a completed C policy. Shared preview updates can change the inputs seen by the frozen quality/gain heads, so a candidate that improves train ranking still requires B/C label refresh, independent dev calibration and complete test evaluation.

Source reference: [SiamFC-PyTorch BalancedLoss](https://github.com/huanglianghua/siamfc-pytorch/blob/master/siamfc/losses.py) separately normalizes positive and negative terms. This informs the foreground/background imbalance hypothesis; our unit-mass spatial ranking objective is a different implementation. [OSTrack target-conditioned candidate elimination](https://github.com/botaoye/OSTrack/blob/main/lib/utils/ce_utils.py) uses target template masks. Our current central descriptor already has a target-centred restriction; those masks do not establish that our proposal ranking works. Neither repository has been run as a benchmark here.

Actual admission commands, in `/home/gaob/Dual-RGBT` with the existing conda Python:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 /home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python scripts/train_global_preview_rank.py --plan reports/global_preview_rank_plan.json --run global_preview_rank_sanity_20261014 --sanity
```

After its genuine COMPLETE/exit0 and all admission assertions, start once from the same original full B:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 /home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python scripts/train_global_preview_rank.py --plan reports/global_preview_rank_plan.json --run global_preview_rank_formal_20261014
```

Physical GPU0's original full C and its observation owner remain unchanged. A localization fit on GPU1 is not a request to preempt its eventual evaluation or to claim the target 78/72/62 is achieved.
