# C20 state-write visual supplementary review

**Verdict: WARN; blocking issues: 0.** Deterministic provenance/selection/geometry verification: **PASS**. No mandatory deletion or claim correction was found.

**Attribution:** the original fresh reviewer `/root/audit_c20_state_write_temporal_20261015` continued this limited supplementary review. This is **not a new fresh-context audit**. Requested `gpt-6-astra` / `max`; actual routing **UNATTESTED**; same-family / provisional.

The reviewer viewed all four actual PNG panels and read the collection/drawing code. The new appendix supports its qualified local observations and does not turn the old C20 examples into evidence of E1 performance or template-write causality. The closed numerical audit is unchanged.

## provenance and archive: PASS

- All 32 JPG names match the provenance receipt and the tar member set; every archived member equals its local JPG bytes and recorded SHA256. There are 30 unique payload hashes because the two same-sequence cases reuse their initialization RGB/TIR images.
- Archive SHA matches ssh.stdout.bin and provenance.json. Saved SSH/SCP transports each record exit 0; both stderr files are empty. The reviewer did not run SSH/SCP or re-run the collector.
- remote.py source/file instructions were parsed as literals and compared against original-manifest paths indexed by program frame number. No remote script was executed.
- Original manifest hash matches the comparison manifest lineage. All GT serialization bytes for the 3 selected sequences match float32 conversion followed by six-decimal comma/LF output; provenance GT/prediction values match archived arrays.
- The four actual 2560×1280 PNGs were viewed. The drawing source reads original JPEGs, overlays GT/prediction rectangles, and adjusts axes around the box union with a margin. Review did not synthesize, rerender or edit an image.

**Evidence:** `C:/Users/gb/.codex_tmp/dual_rgbt_collect_state_write_visuals_20261015.py:44-63`; `C:/Users/gb/.codex_tmp/dual_rgbt_collect_state_write_visuals_20261015.py:64-122`; `C:/Users/gb/.codex_tmp/dual_rgbt_state_write_visuals_20261015/provenance.json`; `C:/Users/gb/.codex_tmp/dual_rgbt_state_write_visuals_20261015/ssh.transport.json`; `C:/Users/gb/.codex_tmp/dual_rgbt_state_write_visuals_20261015/scp.transport.json`.

## selection and frame indexing: PASS

- Selection independently matches the four frozen rules applied in saved event order: unique prior-good event; first confidence>=0.95 bad write; first prior-streak>=20 inside event; first prior-streak>=20 outside event.
- Selected program frames are [0,369,370,390], [0,258,425,445], [0,364,655,675], and [0,781,840,860]. All are zero-based manifest indices, and the last panels are exactly anchor+20 for these cases.
- The predecessor to the current low-IoU streak is anchor-streak-1. Only the designated previous_frame_good case has preceding IoU>=0.5; other three preceding panels have IoU in[0.1,0.5). The document correctly warns that before-streak does not establish correct identity or good localization.
- Four cases come from 3 sequences; the last 2 are the same sequence. Selection is illustrative, conditional and nonrepresentative.

**Evidence:** `C:/Users/gb/.codex_tmp/dual_rgbt_collect_state_write_visuals_20261015.py:29-54`; `docs/C20_STATE_WRITE_VISUAL_NOTES.md:5-7`.

## local visual observations: WARN

- previous_frame_good.png supports that the preceding prediction spans two nearby board/folder-like objects and the event prediction favors the right object while GT favors the left. The later panel changes the left appearance/visible region; four frames cannot resolve precise occlusion, annotation-boundary change, or cause. The wording already retains these alternatives.
- high_confidence.png supports the selected red-clothed person versus the adjacent dark-clothed GT person, with broadly similar grayscale human silhouettes in TIR. This is an appearance observation, not calibrated thermal quality, temperature similarity, or a population-level failure rate.
- long_inside.png supports different selected and GT people among nearby similarly dressed people and several thermal silhouettes. Checking identity association is a reasonable hypothesis for this case; the document does not classify all inside-crop errors as distractor errors.
- long_outside.png shows the GT-marked person in the source-image scene and the prediction remaining near the neighboring group. An annotated target being visible in this original image does not establish that the current primary crop can access its centre.
- No required deletion or claim correction was found for these qualified local observations. WARN reflects subjective visual interpretation and selected examples, not a discovered contradiction.

**Evidence:** `docs/C20_STATE_WRITE_VISUAL_NOTES.md:9-14`; `previous_frame_good.png`; `high_confidence.png`; `long_inside.png`; `long_outside.png`.

## modality response and crop distinction: PASS

- The high-confidence event trace has primary modality 1/TIR and action 3/PREVIEW. The saved 0.9520832300186157 response is the final corrected output score for non-STOP action, not an isolated TIR quality score. The statement at line12 is correct.
- The old tracker first predicts/clips the current box and then tests the write condition, so the current frame cannot be evidence that its own later template write caused that same frame box switch.
- Inside/outside classifications independently match the previous-prediction factor 4 search geometry. Six-decimal saved-box reconstruction is approximate, as disclosed. Target-centre-outside does not mean full object absence from the crop, image invisibility, or absence from a later auxiliary query.
- The appendix describes centre accessibility and original-image visibility separately and does not claim a measured query counterfactual.

**Evidence:** `dual_rgbt/model.py:181-195`; `dual_rgbt/tracker.py:34-54`; `scripts/analyze_full_dev_coverage.py:51-65`; `docs/C20_STATE_WRITE_VISUAL_NOTES.md:12-14`.

## e1 geometry projection: PASS

- global_geometry uses a top-left square with side=max(image height,image width); candidate_tracker resamples this region to 256 before the shared 3 stride-2 convolutions. Global preview_state asserts a 32×32 feature map.
- Candidate centres are explicitly mapped with((column,row)+0.5)/32 times broad side. Therefore the declared lattice-pitch projection for target width is32*target_width/max(image height,image width). The two selected small-target events each occupy 0.6333333333333333 grid spacings in width, strictly below 1.
- Candidate separation uses main-search side/4 in original-image coordinates; it is not a learned identity criterion. The inspected material provides no E1 feature maps, rankings, selected candidates, or outcomes for these images.
- The width calculation is geometric only. Overlapping convolutional receptive fields, candidate region extent and bilinear feature sampling prevent interpreting one cell as a hard CNN resolution bound. No E1 forward pass was executed in this review.
- The old C20 source uses its local preview and does not execute the new E1 global-preview path. Line16 explicitly prevents attributing old C20 failures to observed E1 performance; the capacity/separation concern remains a future hypothesis.

**Evidence:** `dual_rgbt_global/policy.py:13-28`; `dual_rgbt_global/candidate_tracker.py:25-35`; `dual_rgbt/model.py:14-23`; `dual_rgbt_global/candidate_policy.py:15-35`; `dual_rgbt_global/candidate_policy.py:59-71`; `docs/C20_STATE_WRITE_VISUAL_NOTES.md:16-18`.

## scope and preservation: WARN

- This is the original fresh reviewer continuing a supplementary review, not another fresh-context independent audit. Requested Astra/max routing remains UNATTESTED; same-family semantic judgment remains provisional.
- The closed 356 numerical-audit inputs, both closed numerical-audit reports, all 46 source seals and all 47 new visual inputs are byte-unchanged. The original numerical JSON retains SHA256 9afaf888a1dbcbd7b6425477187fcc804ffacf83e44e85ee1919fea93f24b2d7.
- No collected or plotted image, GT coordinate or prediction coordinate is embedded in the public review. Raw inputs and detailed arithmetic stay private. Public before/after dictionaries bind the exact reviewed appendix and all specified image/provenance/collector inputs.
- No inference, training, threshold selection, current-status/log intake, process/CIM inspection, SSH or GPU operation occurred. This review does not launch the future controls mentioned by the appendix.

**Evidence:** `docs/C20_STATE_WRITE_VISUAL_NOTES.md:3-7`; `docs/C20_STATE_WRITE_VISUAL_NOTES.md:16-20`; `reports/C20_STATE_WRITE_TEMPORAL_RESULT_AUDIT.json`.

## Bound evidence and limits

The companion JSON contains `audited_input_hashes` and `audited_input_hashes_after` as `path → sha256:HEX` dictionaries. They bind all 47 visual inputs, the prior 356 inputs for preservation, and both closed audit reports: 405 unchanged inputs. Every before/after digest and preserved byte comparison agrees. Four PNGs, 32 original JPG file names, provenance, tar, recorded transports, collector source and generated remote source are included.

Both small-target examples project to approximately 0.6333 of the declared global-grid spacing in width. This public report omits original pixel values and GT/prediction coordinates; detailed calculations and selected-frame checks remain in `C:/Users/gb/.codex_tmp/dual_rgbt_state_write_visual_review_20261015/private_verification.json`.

The four figures use3 sequences and 30 unique original image payloads across 32 named files because one initialization pair is reused. The saved preceding panels for the high-confidence and two long-streak cases are not IoU≥0.5 examples; the appendix correctly limits what those panels mean.

Recorded SSH/SCP receipts each show exit 0 with empty stderr. No collector rerun, new remote operation, model import/forward, training query, process check or threshold tuning occurred. The parent-reported original helper completion was not independently reacquired. PNG rendering was assessed through actual image viewing and source inspection; it was not rerun.

The comparison remains selected and observational. Thermal silhouettes do not establish standalone modality quality. Grid pitch does not establish a CNN hard boundary, effective candidate recall or E1 failure. The future controls in the appendix remain hypotheses and were not executed by this review.

Original numerical JSON remains `sha256:9afaf888a1dbcbd7b6425477187fcc804ffacf83e44e85ee1919fea93f24b2d7`; all 46 sealed source files and the original 356 audited input bytes are unchanged.
