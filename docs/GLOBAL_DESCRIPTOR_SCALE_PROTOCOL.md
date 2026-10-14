# Frozen descriptor-scale diagnostic

2026-10-14. SOURCE_PENDING; no GPU run yet.

The completed train882 preview fit improved main-outside local target-centre coverage from 11/239 to 25/239 and wide coverage from 49/239 to 55/239. Overall wide coverage fell from 760/1752 to 720/1752. A lower loss or nearer GT-cell rank does not establish a useful recovery proposal; full C is deferred while this bottleneck remains.

The initial factor-two template is resized to64, giving a nominal geometric-mean target extent32 pixels before the CNN. The full-image256 preview in the native main-outside sample has median target width/height9.6/15.733 pixels. These observations motivate a scale-matching hypothesis, not a demonstrated cause. The pinned SiamFC implementation sets search/template crop sizes proportionally to their canonical output sizes; see reports/global_descriptor_scale_reference_provenance.json. It has not been reproduced as a benchmark.

Compare only two descriptors using the same frozen post-fit checkpoint ad059943…: template64/central4x4 versus template32/central2x2. Both use the same shared three-convolution CNN, identical global256 current pixels, grid32, old auxiliary roles and local/wide ROI sides. Global features are computed once per state and reused across descriptors. No new global resolution, heavy auxiliary encoder, parameter, controller or tracker is introduced. GT is used only after the peak is selected to compute rank/ROI coverage. Both descriptors still use frame0 templates; appearance ageing and distractor identity are not isolated by this contrast.

Sanity uses the first two frozen train states, four comparisons, 265 unchanged state tensors, no optimizer/backward and strict reload of both score maps and both peak coordinates. Formal uses all1,752 states and3,504 comparisons. Baseline template64 peaks must reproduce the completed preview fit's private after-ranking record. Raw peaks and frame records remain private; publish aggregate groups only. Main-outside membership and numerical rank remain native-derived. A positive result would still need query benefit, B/C label refresh, dev calibration, full test, real cost and module controls.

Run in /home/gaob/Dual-RGBT with the existing home conda. Only after independent SOURCE_PASS:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 /home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python scripts/diagnose_global_descriptor_scale.py --plan reports/global_descriptor_scale_plan.json --run global_descriptor_scale_sanity_20261014 --sanity
```

After actual sanity COMPLETE/exit0 and its frozen/reload assertions, run once:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 /home/gaob/Dual-RGBT/conda-envs/dual_rgbt/bin/python scripts/diagnose_global_descriptor_scale.py --plan reports/global_descriptor_scale_plan.json --run global_descriptor_scale_formal_20261014
```

The original GPU0 full C/owner16936 and its next20:30 scheduled diagnostic remain unchanged. GPU1's preview fit is genuinely terminal; this short frozen test creates no claim about the final three-module method or the target78/72/62.
