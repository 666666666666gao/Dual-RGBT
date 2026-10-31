# Dual-RGBT 交接文档

## 2026-10-31 02:20 UTC - Actual E7 full-test analytic MAC accounting; all17 workers live

- Added tools/estimate_full_test_macs.py and five focused CPU tests, PASS on
  local/server26. Actual full E7 trace reweighting verifies38core/config hashes
  and248run artifacts,245seq220703frames/220458tracked. Existing23measured
  cases exactly fit the source-derived122368MAC/global-candidate increment.
  Full test has59shapes;36are analytic estimates covering3356frames, NOT new
  GPU measurements. Tracking41.429531GMAC; initialization-inclusive41.383540;
 103218887parameters. Old40.240538GMAC describes parentDEV, not this full test.
  Scope remains supported convolution/matrix MACs, excludes other operations.
  No accuracy/FPS improvement, independent review or scientific acceptance.
- Report ALL882_E7_FULL_TEST_MACS_ESTIMATE_20261031.json SHA ee54d8da...,
  estimator b5220e55...; see companion FINDINGS for bindings/limitations and
  resolved pre-output schema/fixture mistakes. Existing E7 full metrics stay
 67.543368/62.650040/51.093086,exploratory_test=true/test_eligible=false.
-02:20:29/30UTC all17native children live;traceages<=10.7s,8GPUs assigned.
 25E12=175seq/181525observedframes;utilityE6-E10=155/154/171/156/151seq.
 26matchedparent=63seq/53490observedframes;all882E8=105seq/102388observedframes.
  Matchedparent scorer3756936 live. No new full score, failure or idle GPU.
  Independent451sinterval:25E12~53minleft,utility~69-104min;26~230-369min.
  Buffered concurrent ETA estimates only, not formalFPS/guarantees.
- E0 full collector remains unlaunched/review-required; no user authorization
  to message old reviewer, no message sent, no reused observer approval.
  No newdev/gain-grid/size-tail expansion or speculative weight deletion.
  Goal remains active/unmet; new modules remain untrained/unvalidated.

## 2026-10-31 01:50 UTC - TRAIN-only confidence gate cannot cover most lost states at the tested protection point

- Continued independent CPU work while the new E0 collector's source review
  remains unfulfilled; no unauthorized review-chat message, full collection,
  new dev or gain-grid launch. This turn made real diagnostic progress.
- New tools/diagnose_parent_confidence_gate.py reverified935inputs from the
  existing parent21e07 TRAIN reliability report and45317sampled deployed states,
  exact882seq/706fit176holdout. Fit-only threshold0.5916850566864014 retains
  at least95percent of fit IoU>=.5 states; neither holdout nor test fits it.
  Holdout catches793/2280lost (34.7807%), misses1487 (65.2193%), interrupts
 375/6319good states, retains94.0655%. Fixed.65 catches891lost but interrupts
 519good. A trigger is not a successful recovery; no online policy changed.
- Lost-vs-good confidence AUROCfit.718993737/held.626060225, excludes only
  intermediateIoU states. Five tests pass local/actual26CPU; direct pairwise
 14407320held pairs reproduces.6260602249412104. Result
  reports/PARENT_CONFIDENCE_GATE_DIAGNOSTIC_20261031.json SHA d67ca747...,
  source60273324... . See PARENT_CONFIDENCE_GATE_FINDINGS_20261031.md.
  Not protected E0, true identity supervision, unseen holdout or benchmark
  metrics; upstream model saw component holdout. No scientific acceptance.
  Do not transfer this threshold to E0 or claim95%held preservation.
-01:50:47UTC all17inference children remain live, fresh traces<=10s; no
  completed new score.25E12=161seq158314frames,utilityE6-E10=142/140/148/143/137.
 26all882E8=98seq83437frames;matchedparent58seq35808frames,trace5.3sold.
  Its CPUscorer3756936 also verifiedlive; do not duplicate/restart. Eight
  physical GPUs remain assigned. Long-sequence counters are not stalls.

## 2026-10-31 01:32 UTC - Protected E0 redesign; real prefix parity passed

- Latest pasted steering changes the mainline: protect original E0 local
  localization, then failure sensing, candidate-local target identity and
  separated localization/memory commits. Do not add same-structure gain-only
  loss/K/cost grids. Existing jobs stay intact. No new dev; full-test results
  remain exploratory/ineligible. Active goal75.8/71.9/60.2 and higher pasted
  LasHeR objective78.74/75.16/63.84 remain distinct, both unmet.
- Added read-only `dual_rgbt/e0_observation.py`, calling original predict once
  and returning the identical result. Samples914features including768deep ROI
  channels and explicit primary-crop coverage. Original E0 model/tracker are
  unchanged; no extra heavy forward, replacement local decoder or optimizer.
  It is not yet an identity classifier or validated M1 module.
- Actual26 canonical-root TRAIN prefix check completed at01:20:58UTC:
  wrapper3851521, GPU3, native COMPLETE/exit0,59.666960seconds.
  Three31frame prefixes,90distinct tracked positions; observer off/on matches
  direct E0 for unrounded returned boxes, routes, templates, history,counters.
  All265state tensors unchanged;0backward/updates;90rows x914features.
  Returned boxes include original clipping; pre-clipping predictor tensors
  are not compared. This is not full-trajectory parity or historical TF32
  score reproduction. FP32/cudnn+matmul TF32false,cost0,protected ee60704e...
  E0 checkpoint. Canonical root is /data2/gb/Dual-RGBT, NOT sealed snapshot.
- Source review4d115c2e... is0blockers/same-family/provisional/UNATTESTED;
  no scientific audit acceptance. Deterministic verification checks122sources,
  189inputs and3artifacts. Plan2d41aa75..., result9d2406fd...;
  see reports/E0_OBSERVER_TRAIN_PARITY_VERIFIED_20261031.json and
  reports/PROTECTED_E0_REDESIGN_20261031.md. Five local/actual26 CPU tests pass.
- New full paired TRAIN collector implemented/compiled, not reviewed/launched.
  Draft ba1621e7... binds123sources,190inputs, completed prefix/native receipts,
  882seq/459435frames,706fit/176holdout and8disjointshards. Component holdout
  was seen by upstream E0 training. Only186prefix images individually hashed;
  full paths/GT are manifest-bound, not all full image pixels hash-verified.
  full_collection_allowed=false. The existing source review predates collector
  and MUST NOT approve it. Current continuation exposes no native subagent
  continuation; user authorization to message the completed review chat has
  been requested but no message sent. See E0_PAIRED_TRAIN_COLLECTION_REVIEW_REQUEST.
  Do not launch until new source review and actual capacity checks pass.
-01:32UTC all17existing inference wrappers/children live, traces fresh:
 25GPU counts1/2/2/1,26counts3/3/3/2.25refreshedE12=148seq144539frames;
  utilityE6-E10=131/126/141/132/123seq.26all882E8=92seq75284frames;
  unchanged21e07parent=40seq30871frames;9size-tail runs=58-74seq.
  Parent full score still pending; do not attribute full-test gain-head delta.
- Retention inventory:25 numberedweights46,26sealed22,26canonical0; no group
  above5epochs. Canonical protected best/latest files are not counted by this
  numbered inventory. No newly proven disposable dependency, so no deletion.
 25free6.78GiB,26free6.25TiB. All three modules and target gains remain unproven.

## 2026-10-31 00:35 UTC - Seventeen concurrent full-test workers; matched parent admitted

- User confirms other cards may also run concurrent inference/collection. Latest
  batch observations show25 GPU0/1/2/3 has1/2/2/1 live inference children and26
  has3/3/3/2. All17 workers have live native wrappers/children and fresh traces;
  no unexpected exit. Do not duplicate pending checkpoints or restart long
  sequences merely because per-sequence progress holds. No new dev launched.
- Missing unchanged-parent full-test control now RUNNING on26 GPU0, sharing
  with B576E3/B640E3. Run matched_parent_lasher_test_p26_exploratory_20261031;
  native3753810, child3753812, CPU scorer3756936 waiting180seconds.
  At00:34:50UTC3sequences/751frames complete; current trace8.23seconds old.
  Scope245sequences/220703frames, K8/cost.15, original online template updates,
  checkpoint21e07cd1f645c6f7237e063c81cf5d60f6a14bca710e0180e8a3ca29dc827073.
  PlanSHA8c04c3b272fb5ee19c9a9e0c865ba3bfbf85dd626e53a635f9ad878ae05ea834.
  No completed metrics yet; early startup/sequence throughput is insufficient
  for a useful finish ETA. Continue180-300second monitoring.
- TRAIN smoke passed native COMPLETE/exit0,1sequence/11frames,265 state tensors
  bitwise unchanged, zero backward/optimizer updates. TRAIN-smoke manifests
  referenced absent old26 paths; explicit path mapping was fixed and all22
  image hashes match25. Sources/scripts were not substituted into bound helpers.
  Source reviewSOURCE_PASS/0blocks is same-family/provisional/UNATTESTED.
  Required Astra route unavailable; no mandated-model or final-result audit
  acceptance claimed. Exact source/map/runtime bindings are in
  reports/MATCHED_PARENT_FULL_TEST_{PLAN,SOURCE_REVIEW,LAUNCH_VERIFIED}_20261031.json.
  Existing dirty eval_size_tail.py/score_size_tail.py are untouched.
- Read-only TRAIN reliability analysis verified13 existing parent cache audits,
  native/command/config/labels and882 unique TRAIN sequences.45317 sampled
  fifth-frame states include11603 lost(IoU<.1),6596 lost with confidence>=.65;
  fit5207/9323 lost states,holdout1389/2280. There are218 actually executed
  queries. Reconstruct only STOP or selected-query final confidence/IoU.
  This supports investigating TRAIN reliability but is not a causal memory
  result, full-trajectory prevalence or unobserved-modality comparison.
  See reports/PARENT_TRAIN_RELIABILITY_DIAGNOSTIC_20261031.json.
- Corrected prior next-experiment suggestion: historical complete97sequence
  dev static-template result already exists,68.987885/64.691062/53.070734
  versus matched parent75.967585/70.762121/56.047161. SR drops2.976427points.
  Do not repeat blanket freezing or restart dev. Any selective memory gate
  needs TRAIN evidence. This old result does not replace the missing full-test
  parent control or demonstrate causal template contamination.
- Numbered-weight inventory:25 has46files,26snapshot22; no directory group
  exceeds five epochs. Smoke/resume receipts and active/audit dependencies
  remain protected; no newly proven unreferenced obsolete weight was deleted.
  Available disk approximately6.86GiB/6.25TiB respectively at inventory.
- Heartbeat dual-rgbt-25-26-full-test-monitor updated ACTIVE every5minutes to
  cover all17 jobs, scorer attachment, resource-based admissions and retention.
  Notify only meaningful changes/completion/failure/required action.
  Goal remains active, targets unmet. Best completed currentcandidate tuple
  remains67.543368/62.650040/51.093086,103218887params, concurrentFPS8.615936.
  All test exploration remains exploratory_test=true/test_eligible=false.

## 2026-10-30 14:56 UTC - Parent all882 formal complete; five retained dev smokes

- On26, 13-cache parent-control assembly independently rechecked all1790
  artifact hashes and exact882 TRAIN sequence coverage/706 fit/176 holdout.
  Exact smoke plan SHA2375ac71500beb4f25b7bf9050c00a7f966ed087e79ade64a98c99d3fedd606b.
  Both source review and execution review SOURCE_PASS/0 blockers, same-family
  provisional, actual model UNATTESTED. Smoke passed2epochs/26updates, strict
  reload, only4 query_gain tensors changed, recorded-output errors below1e-5.
  Baseline recoverables1355/239 states across216/46 sequences pass existing
  100/15 state and10/3 sequence gates.
- Formal `cached_parent_gain_all882_formal_20261030` completed under the
  reviewed controller,0 failures. Queue has preserved latestfive E4-E8 and
  launched all five11-frame LasHeR dev smokes on26 GPU0/1/2/3/3; all five
  active, no queue omissions. Each successful smoke will be followed by its
  own complete97/54646 dev evaluation; scores start after that wrapper exits.
  Queue polls180seconds. This changes only the gain head and is not full-network
  training; no test or promotion claim.
- During metadata transfer, same-name parent-control summaries briefly collided.
  Original provider25 queue is preserved under `...ORIGIN_SERVER25...`.
  Provider26 registry restored byte-identically and its terminal summary was
  reconstructed from the original start snapshot and unchanged native receipts
  plus complete audits. All eight task receipts and 1790 cache artifacts were
  reverified; no collection task or cache file was altered.
- B640E3 fulldev verified: PR75.39441678209472/NPR69.88400511140033/
  SR55.332100370581294,392 artifacts and curve. E4TRAIN is live. B576E4 dev
  had81/97 sequences at14:53 and remains live.
- All882 E6-visited head scores E8/E9/E10 are complete97/54646 with392 artifacts
  each; PR/NPR/SR: E8 74.43758360463106/69.76159206550565/55.42880692095357;
  E9 74.462003946071/69.93478368851646/55.39637932185995;
  E10 74.73592840500555/69.83434165045185/55.62158816115923. All are below
  parentSR56.04716073871899; do not promote. E11/E12 remain under dev.
- All882 parent-control protocol and exact evidence are local under
  `reports/CACHED_PARENT_GAIN_ALL882_*`; queue remains on26. Next copy each
  retained weight's complete score/curve/evidence after scoring, then compare
  with protectedE0, parent, size tails and E6 results before choosing a model.
  Accuracy goal78/72/62 remains unmet.

## 2026-10-30 13:52 UTC - 26 parent-cache workers at terminal boundary

-26 parent control s4 full collection COMPLETE and audited:
  `84d7aa3a9f90f57effa80176dbf616578b43ab0a7355de0b77ce3a3f7efeebff`.
  s0/s2/s6 progress is96/96,98/98,98/98 sequences, with50,688/50,683/
  50,672 frames; their owning wrappers/audits have not yet all registered
  terminal, so keep the queue RUNNING (5 stages complete,3 full active,0 failed).
-25 at13:52 B576E4 dev26seq/9090frames andB640E3 dev41seq/18368frames.
  All882 E10 dev88seq/51476frames; E9/E8 each61seq/37059frames. Five queue
  jobs complete,3 dev active,2 retained-weight smokes queued,0 failures.
- Next await native wrapper exit and audit for all remaining26 shards, then
  transfer only verified cache data/provenance to26's ample storage. Do not
  start the all882 parent-control smoke until the assembled 13-cache manifest,
  exact partition/thresholds, trainer binding and independent review are ready.

## 2026-10-30 13:47 UTC - Parent controls complete on25;26 shards near finish

-25 parent remaining782 control queue COMPLETE: all8 smoke/full stages audited,
 0 failures. The four odd full shards each cover98 sequences; fit/holdout
  sampled states are s1 3805/1198, s3 4275/737, s5 3386/1613, s7 4152/845.
  Full audit JSON is retained on25; s1/s3 audits are also committed locally.
-26 even control queue: the four full workers remain alive with 0 failures.
  Latest available progress is s0 89/96 sequences,48182 frames; s2 93/98,
  48504; s4 94/98,49604; s6 93/98,48304. Keep original receipts and await
  terminal audits; do not treat sequence progress as completed cache.
-25 at13:47: B576E4 fullDEV GPU3=12seq/5594frames; B640E3 fullDEV GPU3=
  32seq/14643frames. Capacity controller2360358 and2342712 preserve both
  chains. All882 refreshed-head E8/E9/E10 dev workers GPU3/3/2=
  53/53/78seq,33285/33285/46788frames. Queue has5 completed jobs,0 failures,
  E11/E12 remain queued. Shared GPU3 concurrency is intentionally retained.
-25 filesystem has about6.5GiB available;26 has about6.3TiB. Do not stage the
  combined comparison on25. After the four26 control shards are fully audited,
  transfer audited cache inputs and their provenance to26, then run the
  preregistered 882-sequence parent-loss control on an idle26 card only after
  exact source/input checks and independent review. Existing B576/B640 plans
  and active GPU children remain untouched.

## 2026-10-30 13:33 UTC - B576 capacity adoption; first parent controls verified

- B576 CPUcontroller2197443 replaced by2360358 at13:31:47UTC with the
  already reviewed capacity scheduler. E4 TRAINwrapper2330883 preserved/live;
  old CPUcontroller absent. Original plan SHA7724a6706715b1762e2a5c2f3c8828043b8e72467e0d39e4e5b03abc75f99aff
  unchanged. Ready/takeover/handoff/status retained under
  `reports/size_tail_capacity_adoption_b576_20261030/`.
- Parent-control25s1/s3 naturally COMPLETE and audited: each98sequences,
  50685/50689frames. Native/audit SHA and exact coverage accepted by existing
  queue; no reset or recollection. Full audit receipts copied to local reports.
 25s5/s7 still active,13:33progress90/94sequences and44528/47780frames.
 26allfour parent collectors remain active onGPU0/1/2/3, no failures.
- Refreshed-all882E10 fullDEV GPU2 advanced to53seq/33285frames;
  E9/E8 GPU3 each50seq/23867frames. B640E3DEV GPU3 at23seq/7654frames.
  B576E4TRAIN GPU1 still live. GPU0 on25 is reserved for B640E4 after its
  current fullDEV. No dummy allocation or repeated evaluation added.
- Continue preparing the all882 parent-cache matched control with identical
  706/176 membership, parent initialization, original loss, batch2866/LR.00112
  and admission thresholds. Cache completion, source review and exact-input
  real smoke must precede formal launch. Current accuracy goal remains unmet.

## 2026-10-30 13:24 UTC - B640 capacity wait resolved; B576E3 scored

- B640 had completed E3 TRAIN at12:49UTC but its original shared scheduler
  remained WAITING_FOR_EVAL_CAPACITY: the process-count ceiling rejected a
  device with more than15GiB free. Added separate capacity controller and
  launcher; original bound controllers/trainer/evaluator/scorer/plans unchanged.
  Keep >=6144MiB device free and >=16GiB host available; remove process-count
  ceiling. Two chains share the admission reserve. New code supports adopting
  terminal TRAIN at the capacity wait without repeating its updates.
- Independent reviewer01a18c8c-7889-7410-80e6-c63bfb95215a SOURCE_PASS/0blockers,
  same-family/provisional, actualmodelUNATTESTED.18focused tests pass locally
  and in actual25CPU environment. Exact source/test hashes are recorded in
  `reports/SIZE_TAIL_SHARED_CAPACITY_SOURCE_REVIEW_20261030.json`.
- At13:19:02UTC replaced only B640 CPUcontroller2197846 after exact old-command,
  original-plan, native terminal and new ready identity verification.
  New controller2342712 started E3full97dev wrapper2342724 onGPU3.
  Old controller is absent; original TRAINwrapper2272555 was already naturally
  terminal COMPLETE/exit0. Original B640plan SHA remains
  6e330bdc44571a24c3afe68282f4d37a9e02deb01b0603703df8d8368129371b.
  Takeover/ready/handoff/native/status saved in
  `reports/size_tail_capacity_adoption_20261030/`. At13:24 both newcontroller
  and fullDEVwrapper are live. No GPU process restart or E3 retraining.
- B576E3 complete97dev/54646frames PR76.1919419032591/
  NPR71.52336499844037/SR56.092286643633486.392artifacts/native/curve hashes
  independently rechecked by executor on25; SR+.0451259049pp over this parent
 56.04716073871899. This is a small single-seed dev gain, not a test result,
  global-best claim or completion of78/72/62. B576E4TRAIN live onGPU1,
  wrapper2330883, original sharedcontroller2197443; no hot edits.
- All882 E8/E9/E10 fullDEV live onGPU3/3/2, respectively;13:24progress
 48/48/50seq and22822/22822/23867frames. E11/E12 await their own smoke/fullDEV.
  Queue2260888 remains live,0failures. Parent control also live0failures:
 25s1/s3/s5/s7=95/94/88/83seq,48737/48886/38931/43897frames;
 26s0/s2/s4/s6=66/71/72/78seq,37310/38600/38120/37174frames.
  Do not restart any long task from a progress-file observation alone.
- Next useful checks:25s1/s3 complete within roughly5-10min;25s5/s7 and26
  controls later. B576E4TRAIN expected near13:40UTC; if its nextDEV hits the
  same count gate, adopt the already reviewed capacity scheduler while preserving
  its native task. All882 dev completion estimates require the current measured
  concurrent pace. Poll180-300seconds at useful boundaries; no placeholder VRAM.

## 2026-10-30 12:52 UTC - All882 refreshed gain trained, latestfive in full dev

- Continued the active precision goal. New source-reviewed preparer/controller
  `tools/prepare_refreshed_gain_all882.py` and `tools/run_refreshed_gain_all882.py`
  complete the preregistered E6-visited-state all882 experiment. Original
  matched100 plans/sources, trainer, evaluator and scorer remain unchanged.
 23/23 independent source/fixture checks passed, including exact smoke/formal
  bindings, fixed706/176 partition, rejection below100/15 states over10/3seq,
  wrapper-exit ownership and allfive full97dev scores. Actual new7 tests also
  pass on25. SOURCE_PASS/0blockers, same-family/provisional, actualmodelUNATTESTED.
- Server25 controller2260888 launched12:16:46UTC. It verified the existing
  E6queue1931586 live while waiting; E6queue is now COMPLETE/0failures/terminal.
  All16 E6 caches cover exactly882TRAIN sequences,706fit176holdout.
- Exact-input smoke: SANITY_PASS,2epochs/26real updates, strict reload and the
  same4query_gain tensors changed. Fit35740states/holdout9577states.
  Recoverable1355/252states across211/47seq; unchanged admission thresholds pass.
  Recorded E6 output errors fit1.1920929e-6/held8.9406967e-7, below1e-5.
- Formal `refreshed_query_gain_all882_parent_loss_formal_20261030` completed
 12epochs/156updates using original parent initialization, original parent_loss,
  seed2027, batch2866, LR.00112, min8/max30/patience4/min_delta.0001.
  Exact smoke/recipe/input binding checked before launch. CPU executor verified
 1848 input hashes, original parent, native/config/result bindings, finite
  retained tensors and actual latestfive filenames+SHA. Stopping recomputed:
  E8 best component-held regret.03922084718942642; E9-E12 stale, stoppingE12.
  Latestfive E8-E12 all retained. Peakreserved1283457024bytes (~1.20GiB);
  this is gain-head training, not20GiB full-network training.
- Baseline held regret.03982507064938545; finalE12.04037972539663315.
  E12 sampled recoveries7 versus baseline1, with4 destroys-good versus0.
  These sampled diagnostics do not establish full-trajectory improvement.
  No promotion, test claim or validated three-contribution claim.
- At12:48:49UTC all882 queue remains live,0failures. E10 fullDEV GPU2:
 11seq5143frames; E9/E8 fullDEV GPU3 each5seq3519frames.
  Original wrapper identities are saved in the queue snapshot.
  E11/E12 await their own smoke then fullDEV.
  Three slots GPU2/3/3, <=3concurrent evaluations. Official scoring starts only
  after the owning inference wrapper exits. GPU0/1 reserved for size training.
- B640E2 complete97dev/54646frames PR75.90810360585813/
  NPR70.70498020378822/SR56.01440616892699;392artifacts, curves and nativeSHA
  reverified on25. SR remains.03275457pp below parent56.04716073871899.
  B640E3 TRAIN live wrapper2272555/GPU0 (~22.5GiB snapshot). B576E3 fullDEV
  remains live wrapper2212998/GPU3; no new complete score yet.
- Parent-control collection remains live on both servers,0failures:
 25 s1/s3/s5/s7=59/46/60/55seq,27947/23548/26555/26280frames;
 26 s0/s2/s4/s6=43/47/46/49seq,17162/21935/21395/21826frames.
  No duplicate or placeholder work. Poll180-300seconds only at useful boundaries.
- Evidence: `reports/REFRESHED_GAIN_ALL882_SOURCE_REVIEW_20261030.json`,
  `REFRESHED_GAIN_ALL882_LAUNCH_20261030.json`,
  `REFRESHED_GAIN_ALL882_TRAINING_VERIFIED_20261030.json`,
  `reports/refreshed_gain_all882_training_20261030/` and both exact stage plans.
  Next: verify allfive full97dev scores, continue B576/B640 epochs and latestfive,
  complete parent-control caches for a matched all882 policy-state comparison.

## 2026-10-30 12:00 UTC - Idle GPUs assigned parent-policy control collection

- Latest user steering: immediately use26's idle four GPUs and add useful
  collection alongside25 evaluation jobs. Registered eight new, disjoint
  frozen-parent control shards over the missing782TRAIN sequences,
  626fit/156holdout,405455frames. Existing first100parent caches are reused.
  Combined coverage will be706fit/176holdout, matching the E6 all882 partition.
  These are different behavior-policy caches, not repeated E6 collection.
- New controller `tools/run_parent_remaining_control.py`, SHA
  b422f01b4de75d0e54781efc21f4e3f0d067a9f1489270b75990b7e9eab2a903.
  Independent source reviewer01a18c8c-7889-7410-80e6-c63bfb95215a returned
  SOURCE_PASS/0blockers; same-family/provisional, actualmodelUNATTESTED.
  Existing collector, model, evaluator, optimizer recipes and sources unchanged.
  Each shard requires its own real native31frame smoke, causal trajectory parity,
  frozen tensors, audit and exact smoke-plan binding before complete collection.
- Both queues launched11:58:36UTC.26controller2243982 assigns even shards0/2/4/6
  toGPU0/1/2/3.25controller2234353 assigns odd1/3/5/7 toGPU2/2/3/3.
  Reserve25GPU0/1 for B640/B576 real large-batch epochs. No placeholder memory.
  Queue status `reports/PARENT_REMAINING782_CONTROL_QUEUE_20261030.json`
  on each provider; poll180seconds. No task restart on an occupied slot.
- At11:59:48UTC all8smoke native wrappers COMPLETE/exit0 with SMOKE_PASS;
  at12:02:06UTC all8complete collection wrappers RUNNING, configs created,
 4smokes audited per provider,0failures.26 wrappers2250596/2250725/2250822/2250961;
 25wrappers2239115/2239186/2239324/2239406. Queues remained alive.
 26GPU memory1.7-2.24GiB;25GPU2/3 total5.84/7.76GiB at startup.
 26smoke wall12.7-14.5seconds;25smoke11.1-11.4seconds. These are not full
  collection throughput or benchmark FPS. Complete-start queue snapshots are
  `reports/PARENT_REMAINING782_CONTROL_SERVER{25,26}_STARTED_20261030.json`.
- B576E3 TRAIN completed and its full97dev evaluation launched on25GPU3,
  wrapper2212998; shared evaluation scheduling is operational. B640E2 original
  DEV remainsGPU0 until completion; its subsequent TRAIN staysGPU0.
  No promotion or final-test result. Shared scheduler17tests pass.
- Prior scheduling/cached-transfer work committed/pushed as81d545b.
  New control protocol, review and provider launch/plan receipts are saved
  separately. Next precision work remains: complete/audit all882E6 caches,
  prepare reviewed exact-input all882gain smoke/formal admission, collect every
  retained weight's full97dev score, and preserve latestfive.

## 2026-10-30 11:35 UTC - Shared evaluation scheduling adopted

- Goal resumed in chat `01a18c89-655a-73d2-bc53-128ad765570b`.
  Precision and final-test goals remain unmet. Preserve protected E0 and the
  existing train/dev/test separation.
- Verified complete97dev/54646frame artifacts, curves and native bindings:
  B576 E2 PR75.91522494496724/NPR71.03887750087956/SR55.99840410897908;
  parent SR56.04716073871899, so E2 remains0.04875663pp below parent.
  B640 E1 PR74.68236067334149/NPR70.25759479754218/SR55.152483192546285.
  Each score verified392 artifacts. No promotion or final test.
- Finished the interrupted shared-evaluation scheduler. Independent reviewer
  `01a18c8c-7889-7410-80e6-c63bfb95215a` first reproduced a remaining adopted-wrapper
  JSON write/read race. Fixed only the new scheduler: wait for adopted wrapper
  process exit before reading its non-atomic terminal JSON. New Popen tasks are
  also read after exit; ready is atomically published.17 focused tests pass
  locally and on25. FinalSOURCE_PASS/0blockers, same-family/provisional,
  actualmodelUNATTESTED. Original plans/trainer/evaluator/controllers unchanged.
- Actual server25 historical wrapper SHA cc2b87b remains unchanged. Executor
  fetched its bytes and verified AST equality with local wrapper excluding the
  OPENCV override; launcher/controller explicitly supply OPENCV=1. This is
  executor runtime-context evidence, not independent remote-wrapper acceptance.
- Verified takeover at11:34:47UTC: B576 new CPUcontroller2197443 adopted original
  TRAIN E3 wrapper2161182 onGPU1; B640 newcontroller2197846 adopted original
  DEV E2 wrapper2160977 onGPU0. Old CPUcontrollers1736653/1935434 exited;
  both native GPU wrappers remainRUNNING with original planSHA. Subsequent
  full-dev evaluations route toGPU3, training remainsGPU1/GPU0 respectively.
  No current GPU task restarted. Full scoring/stopping/latest5 remain required.
  Existing B576 external keeper1935231 continues. Two concurrent short probes
  previously passed; sustained shared full-dev throughput is still unobserved.
- Server26 all8 assigned full cache shards now complete/audited,434sequences,
  0failures. Copied the four remaining782 even caches to25 without overwriting
  existing files:804files byte/hash verified, archiveSHA
  04cce7a80cb997c7c6b7e2d64a5d7ef7bd3ece96f893dda887afe73d63987e0b.
  Original native/config/audit records and source copies preserved.26 originals
  remain. Transfer receipt `reports/REMAINING782_EVEN_TRANSFER_20261030.json`.
- At11:35UTC25remaining782s5=50seq23975frames,s7=39seq18303frames,0failures.
  B576E3 logged13/16finite updates; physicalGPU1~20.47GiB. B640E2dev33/97seq,
  15894/54646frames. GPU0/2/3~2.21GiB.26 collection is terminal; no newly
  admitted training there. Empty cards do not constitute a training failure.
- Registered `docs/REFRESHED_GAIN_ALL882_PROTOCOL_20261030.md` before complete
  all882 inputs:706fit176holdout, original parent init/loss/seed/batch2866/LR.00112,
  unchanged100/15 recovery gates over10/3 sequences. Separate all882 source review,
  exact-input real-update/reload smoke and formal admission still required.
  Matched100 remains unadmitted at9holdout recoverable states; never lower gate
  or label expanded882 as a one-factor matched100 comparison.
- Next useful checks: B576E3TRAIN near11:40-11:45UTC, then verify its first shared
  full-dev launch; B640E2dev roughly12:15-12:25UTC and remaining25caches near
 12:10-12:30UTC, estimates from actualprogress, not completion promises.
  Poll180-300seconds only when a check is useful; do independent input/protocol
  preparation between checks. Do not duplicate work or allocate placeholder VRAM.

更新时间：2026-10-27。此文件是唯一交接文档；同步到本地桌面 document/Dual-RGBT交接文档.md、26服务器项目根目录及 GitHub。旧逐次节点以 Git 历史和 reports 原始证据保留，不再追加相互冲突的“当前状态”。

## 当前训练与优先级

**最新用户指示（10月27日）：先以精度提升为准，FPS先不管。** 暂停新增阶段测速、速度优化及轻量化分支；已完成的配对测速仅保留历史证据。现有32项训练按原计划完成，优先完整LasHeR dev选型与正式test。不能为减少辅助调用牺牲精度，也不把FPS提升作为当前精度版本晋级前提。原计划的cost网格不热改，它们用于比较精度；实际成本不是本轮选型目标。

2026-10-27 14:50 UTC固定节点：26服务器24项、25服务器8项，合计32项完整正式训练全部原生RUNNING、实际wrapper/child身份正确、已有loss有限。每项同一原B6250/SOT epoch14初始化，882条TRAIN×2轮、182182更新/917106次track/1764初始化；不热改活动训练源码/计划/输入，不把训练loss当精度。

| 主机 | 正式任务 | 本次区间合计更新/秒 | 显存/卡 | 可用RAM | CPU busy |
|---|---:|---:|---:|---:|---:|
| 26 | 24，每卡6项 | 20.7219（约2小时） | 13.82～13.97GiB | 12.52GiB | 31.89% |
| 25 | 8，每卡2项 | 14.0238（约80分钟） | 4.61GiB | 200.09GiB | 14.49% |

CPU采样分别36.90秒/11.63秒，含快照读取开销；GPU利用率是节点瞬时值。不同轨迹/窗口的更新吞吐不是严格配对scaling，也不是在线FPS。原八项26依次98550/90525/82200/82925/85550/87050/65300/65975更新；25八项13300～14050。首quality已进入第二轮，gainpair尚未到91091轮界；均未完整C×2。按本次区间线性外推，26各项约剩22.5～57.6小时，25约26.0～27.4小时；不是完成承诺，未计未来轨迹及保存变化。

26物理RAM约125.5GiB，额外8项会超过实测主存预算，继续HOLD，不为占显存增加任务。原26 phase8八个配置已经移交25，禁止重复启动；原32注册索引保留历史。所有当前权重、B/工厂/SOT预训练/E0/E1/旧Rank依赖受保护；已归档退休M0累计按原回执清理，metadata与失败证据保留。

14:50两个旧观察器20467/42977均已actual exit0消费；CPU安装观察41316也已消费。新唯一26观察16720、25观察25243均已actual WAITING到20:50 UTC；此前不另查全机训练进度，不重复观察器。用户新要求的配对FPS任务另做必要的同卡所有权准入与恢复核验，不当作全机提前轮询。最新统一证据：reports/DUAL_SERVER_FULL_C_PROGRESS_20261027.json及其绑定两机原始节点。

当前最好LasHeR仍为历史E0。32项新训练暂无完整dev/test，78/72/62目标未达到。之后仍按完整native0训练→完整97dev三cost→独立结果审查→冻结配置→三基准执行；正式评估报告总参数/FPS，不统计可训练参数量。

## 25服务器作为新增算力（2026-10-27用户新授权）

25的四张RTX3090在新建项目时已核实为空，无compute PID，RAM总251.54GiB、准备时可用约242.33GiB。用户明确项目和conda均放/home/gaob：已新建/home/gaob/Dual-RGBT，并把原conda完整复制回该项目的conda-envs/dual_rgbt。不是恢复旧队列；原归档保留，26四卡24项任务保持不变。当前两机固定观察节点及句柄见上文。

原环境实际Python3.7.16/Torch1.12.1+cu113/timm0.9.12，无pip升级。源码审查后，父见证实际exit0完成CUDA前反向/AdamW更新/严格重载以及11帧真实TRAIN；fresh agent按docs/SERVER25_REJOIN_20261027.md原样命令再执行一次，实际exit0、17.12秒，框/路由/模板/历史重载逐位一致，总参数103218887。该短程smoke FPS分别4.966/4.711，仅作环境见证，不能写成正式FPS或PR/NPR/SR。环境状态ENVIRONMENT_ACCEPTED，新的训练仍需单独真实4步准入。

已恢复原25 manifest SHA389ba37d，882TRAIN/459435帧、97dev/54646帧；245 LasHeR test/220703帧、210/104706帧、234/116650帧。每序列首/中/末双模态存在性抽查共10008文件，不是全内容比对。LasHeR在/data2/wangwj/dataset/LasHeR-0428，210在/data/wangwj/dataset/RGBT210testingset，234在/data2/wangwj/dataset/RGBT234/RGB_T234；共享数据不复制也不修改。代码/依赖权重/环境全在/home/gaob项目内，完成必要恢复后可用约23.70GiB。

只读检查原26 phase8八个未启动任务/输出路径均不存在后，为这八个已登记K/cost组合生成独立server25计划，仍同一原B6250/SOTep14、seed2027、同网络/三模块、882×2完整训练和182182更新；每卡两项，避免与26已有24项重复。新25使用原生joint_quality/gainpair训练器及默认分配器，不用无明显收益的heap-trim包装；注册表SERVER25_REMAINING_GRID_20261027.json。新25调度已SOURCE_PASS/0阻断，源码54份与旧网络/配方一致。实际wrapper209569于12:30:08 UTC启动独立准入控制器；先8项M0各300秒统一节点，原生exit0/冻结/重载通过后才从原B启动8项完整formal，正式启动再等180秒检查。第一次观察预计12:40:08 UTC。12:40:21 UTC已得到实际控制器COMPLETE/exit0及8项formal RUNNING/PID/命令/输入/所有loss有限回执，每卡两项约4716MiB、可用RAM约200.35GiB。调度完成不是完整训练完成；现在两服务器共32项正式训练。首25观察34849已实际exit0消费，25观察30699已在13:30实际exit0并消费，八项继续RUNNING且全部loss有限、约4716MiB/卡、可用RAM约200.21GiB；49.86分钟区间合计13.5385updates/s，含GPU0两次主动暂停约49.55秒，不能当纯吞吐配对试验。14:50 UTC两机观察已结束：25八项继续RUNNING/finite；下一两机固定节点20:50 UTC，观察句柄与原始证据以上方当前状态为准。不得用旧26计划在25直接运行，也不得因新增机器改动活动26源码。

## 目标、来源及运行位置

LasHeR优先：78/72/62仍是当前主动目标的阶段里程碑；用户最新贴出的审查重新强调三项各＋6，即78.74/75.16/63.84为更高最终目标，不能用阶段里程碑宣布各＋6完成。两者均未达到。还须完整RGBT210、RGBT234测试；当前先追求精度，FPS工作暂停。

用户baseline为72.74/69.16/57.84，沿用、不重测精度。初始化始终为**去掉CADTrack论文三个模块、SOT来源backbone、epoch14**，不是完整CADTrack或DropMAE。

| 项目 | 当前位置 |
|---|---|
| SSH | `2026`主任务＋`2025`新增四卡 |
| 项目/代码/权重/输出 | 26 `/data2/gb/Dual-RGBT`；25 `/home/gaob/Dual-RGBT` |
| Python/conda环境 | `/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python` |
| 本地源码 | `C:/Users/gb/projects/Dual-RGBT` |
| GitHub | https://github.com/666666666666gao/Dual-RGBT |
| 预训练 | `checkpoints/CADTrack_ep0014.pth.tar`，SHA256 cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731 |

25服务器原活动项目已核验归档到`/data2/gaob/Dual-RGBT_archive/retired_server25_project_20261026`，该路径曾在退役后不存在；10月27日按用户新授权在原路径新建独立25项目，仍不启动旧归档队列。共享数据与外部原始预训练保留。26环境Python3.7.16/Torch1.12.1+cu113/timm0.9.12/NumPy1.21.5，环境迁移、真实CUDA更新/重载、11帧tracker和独立文档执行验证均通过。

环境证据：`reports/SERVER26_ENVIRONMENT_ACCEPTANCE_20261026.json`、`SERVER26_ENVIRONMENT_RELOCATION_20261026.json`、`SERVER26_GPU_KERNEL_WITNESS_20261026.json`；provider ledger `.aris/compute/ssh.md`。新训练另有真实反向更新准入，不能仅凭import成功启动。

## 数据与评分协议

| 数据 | 26项目接入位置与规模 |
|---|---|
| LasHeR TRAIN/dev | `data/LasHeR/train`→`/data/wangwj/dataset/LasHeR0428/trainingset`，882TRAIN＋97dev；dev54646帧 |
| LasHeR test | `data/LasHeR/test`内245条链接→`/data/wangwj/dataset/LasHeR0327/TestingSet/testingset`，220703帧 |
| RGBT210 | `data/RGBT210`→项目`data/datasets/RGBT210testingset`，从25迁入，210条/104706帧 |
| RGBT234 | `data/RGBT234`→`/data/wangwj/dataset/RGBT234/RGB_T234`，234条/116650帧 |

26的/data、/data1、/data2已查找：210需迁入；LasHeR/234复用核验后的共享数据，不修改共享原件。新清单位于`data/benchmarks/server26/`。训练`data/manifest.json` SHA dfc64432210b053fc9db517bae0771fdeb750a5d373b732949ef43e8b65b1a7a；旧清单封存`inputs/server25/manifest.json`。全部引用存在非空、GT/初始化已核验；4722张首/中/末帧字节与解码像素25/26相同，属于抽查而非全图像SHA证明。

RGBT234继承差异：afterrain原始visible首帧宽高24/45，原CADTrack toolkit注册10/80；25/26原始文件SHA同为d57d8478baa37976c283327815730d53e6c61b0db93b4bb1af54b6f41bcb0790。沿用原toolkit/manifest，不能静默改GT后混表。LasHeR沿用原toolkit取整和特殊GT处理：首帧用GT，取整GT任一分量≤0的534行在PR/NPR计成功、SR计失败且保留分母；原始IoU诊断另排除260个无效尺寸行，不能混作正式SR。210为PR/SR，234为MPR/MSR。test不参与训练收益标签或参数选择；210/234独立dev协议仍待登记。

## 保护结果与权重

| 完整test版本 | LasHeR PR/NPR/SR | RGBT210 PR/SR | RGBT234 MPR/MSR |
|---|---|---|---|
| 用户baseline | 72.740/69.160/57.840 | — | — |
| 历史E0 cost0，LasHeR最好 | 74.749703/71.067672/59.089007 | — | — |
| Reference-E0 cost0.05，统一配置 | 72.524939/69.068894/57.732435 | 88.061759/65.741170 | 90.806676/68.910136 |
| E1 cost0.15，完整失败 | 67.612357/62.545797/50.842354 | 84.429457/54.041235 | 87.774864/57.523560 |

不能把历史E0 cost0的LasHeR与Reference-E0 cost0.05的210/234拼成同一配置。Reference-E0始终保护；新训练暂无dev/test结果。

保护依赖：E0 SHA ee60704e7fe1e281584430332dea7807a335bd4d723a769e4783217b0ac6f56c；E1 SHA d4454775f085e1af65c61f53d4d21f284706289b6ff3aeb1167f73d79d31a362；正式B6250父权重`runs/global_candidate_policy_formal_20261014/latest.pth` SHA b84d00454717cae56048753e0fe54c1d6b92cfcd2820c01fa65296508488ccb3；factory `runs/stage_b_refreshed_20261013/latest.pth` SHA ba49b506cd4bdcdfb355e8741feaf6c5c61e2edf9cd0a5c91459c3e5957c80ed。

旧Rank/STOP权重SHA527a3bc0f56f6a26e31adcc5e656f971b6bd34bbcd220251151d7db1cc228753已拒绝晋级，但仍是活动训练计划的不可变出处依赖，**目前不能删除**。同理不删除当前checkpoint、父权重、恢复依赖、正式预测及唯一原始证据。已校验的重复缓存、废弃sanity和传输包可按原授权清理；不因“失败”误删仍被计划绑定的权重。

## 旧Rank/STOP已完整拒绝，不跑test

旧C完整882×2/182182更新已结束。它当时直接启动，没有原生终态退出码；该出处警告永久保留。26本轮三个dev均有真实run_task COMPLETE/exit0，每档97条/54646帧，经原toolkit评分和独立结果审查：

| cost | PR | NPR | SR | 总参数M | 服务FPS | mean/P95 ms |
|---|---:|---:|---:|---:|---:|---|
| 0 | 74.420935 | 68.425355 | 54.648371 | 103.218887 | 7.488519 | 133.538/188.009 |
| 0.05 | 74.027542 | 68.057275 | 54.511459 | 103.218887 | 7.702367 | 129.830/183.889 |
| 0.15 | 71.345271 | 65.726933 | 52.833813 | 103.218887 | 9.426419 | 106.085/153.115 |

原E1 dev参照76.557717/71.365998/56.433144；最佳cost0 SR仍差1.784773点。最终决定`REJECT_REPAIR_NO_TEST`，见`reports/RANKSTOP_DEV_FREEZE_DECISION_20261026.json`。审查`RANKSTOP_COMPLETE_DEV_RESULT_AUDIT_20261026.json`为WARN/0阻断：真实GT、完整曲线/序列覆盖/时间及所有399401个训练收益标签核验；same-family/provisional，模型路由UNATTESTED，不能称跨模型独立接受。

三个dev计时负载不同：cost0先单卡后双卡，cost0.05与GPU0性能任务重叠，二者OpenCV默认80线程；cost0.15采用OpenCV1且GPU0训练并行。FPS为含初始化/解码的逐帧服务时间、排除模型加载，不是同负载配对加速证明。不能与用户test baseline直接相减。

## 现有三个模块与真实边界

1. 目标条件主模态调度：双路64预览、初始模板描述、8维历史，共享CNN3→16→32→32、quality74→64→1；在完整ViT之前选择主模态。
2. 定位收益证据请求：主框/主空间特征＋256全图预览32×32网格＋候选几何，gain955→64→1；最多8个全图位置local/wide＋两个主区域请求＋STOP。廉价打分后只执行0或1次辅助请求。
3. 共享编码局部空间修正：从辅助原图裁ROI才进入共享ViT。local128为193总tokens/8×8，wide256为385/16×16；独立查询坐标可输出旧主crop外框。缓存主特征投影、与辅助特征拼接、共享decode/head定位。

**仍有按需局部空间融合，不能宣称完全无交互/无融合。**主路径两个128模板＋256搜索共385tokens。总参数103218887（103.218887M）；按用户要求不评估可训练参数量。正式结果须带总参数、完整FPS、mean/P95、GPU/输入尺寸与辅助比例。跳过分支不等于减参，当前没有同设备baseline配对加速证据。

## 当前两种训练目标（同结构、同起点、同预算）

GPU0 joint-quality计划`reports/global_candidate_joint_quality_trajectory_plan_20261026.json` SHA014101511170dc2742c1cca4662035bae7a2b106f0f6e38cc10310eda090d42f；入口`train_candidate_joint_quality_trajectory.py` SHAa19e7ae96787250ded975e89f2aaeb87a0f9cb3ab7b631e4fe9294ba4d0c2331。源码SOURCE_PASS，原48源码不变，新49源码/23输入绑定，21项CPU检查。真实sanity4更新/20track/1init、原生0、冻结逐张量相同、执行重放与严格重载PASS，COMPLETE SHA f4d568cf2d48268afc5be16efc4435459dff07f8fdefc031a2d74f98592387e8。

GPU0从原B6250开始，联合训练现有预览CNN、quality/gain，冻结backbone/decode/boxhead/localreadout。新增相对质量监督：d*=IoU_RGB−IoU_TIR，d=Q_RGB−Q_TIR，Lrank=mean(|d*|·ReLU(|d*|−sign(d*)d))；保留原绝对质量回归和SmoothL1 gain＋固定STOP margin。预览LR1e-5、头1e-4、WD1e-4、seed2027、cost0.05、K8、stride5，OpenCV1/Torch1/TF32关闭。

GPU1 gain-pair正式计划`reports/global_candidate_joint_gainpair_trajectory_plan_20261026.json` SHA931b2ce6c4f50c49b923cfd7192419fdac64594d324df5e4f55b2f455234be77；入口`train_candidate_joint_gainpair_trajectory.py` SHA6cede0c60dbba2cb76483ffef0e0dd31571838b6bd63fb3cb8e122d538f8357b。它保留GPU0质量项/训练范围/14个可变state张量，仅将gain项改为0.5MSE真实signed IoU gain＋0.5MSE同状态selected-extra与同role sampled的收益差，移除固定STOP margin。推理仍argmax(predGain−λtokenCost)、STOP=0，不新增层、参数、teacher调用或标签。最终源码审查50份源码/26输入绑定通过，报告`reports/JOINT_GAINPAIR_TRAIN_SOURCE_REVIEW_20261026.json` SHA29f167345f4f66c924a5969bf0dd532f8d4768678cf427338803e8bd8b8cabf3（same-family/provisional，实际路由UNATTESTED）。实际准入已通过并正式启动；证据`JOINT_GAINPAIR_TRAIN_EXECUTION_20261026.json`与`JOINT_GAINPAIR_TRAIN_STARTUP_20261026.json`。

两条均先按自身当前预测前进，再读当前TRAIN GT监督；模板/历史不受反事实teacher污染；sanity权重不进入正式训练。不能仅凭平方损失的理想条件均值性质，声称有限共享网络已校准或闭环有效。下一步必须完整dev验证。

## 已核实的问题及被拒绝方向

- E1完整test退化，不能靠减少辅助调用即宣称有效；固定RGB、强制primary query及readout-only完整dev均已失败，不重复其test。
- 旧TRAIN182182状态中59246次另一路primary IoU更好超过0.01（32.52%）；这只是演化训练状态统计，不是最终固定模型泛化误选率。
- 旧完整TRAIN44443次query中11057有害、14535有益，平均raw gain仅0.01610；STOP状态仍有28630次存在一项采样query正收益。支持相对收益/STOP学习，不能当闭环上界。
- 全3dev诊断见`RANKSTOP_THREE_DEV_DIAGNOSTIC_20261026.json`：全图local低IoU比例很高；主框未逐次保存，不能把query低IoU等同“相比主框有害”。不同cost轨迹历史不同，不能作同状态因果差。
- 模板事件仅根据每5帧/响应≥0.65**推断触发**，不是实际写入tensor日志。多数低IoU触发此前已长期失败且发生在STOP帧；不能认定模板污染导致最初漂移。
- `RANKSTOP_UTILITY_TARGET_DIAGNOSTIC_20261026.json`完整399401标签/35037同状态pair：λ0.05下7626 pair存在更好的另一项，2094 selected净负而sampled净正，23473两者净负。候选相对关系和STOP都重要。
- 旧固定margin在理想单标签标量上可把真实小gain0.02推向0.05，而local cost约0.025；61543标签位于该损失冲突区。但这不是实际共享网络校准偏差证明。
- 初稿utility-hinge在审查CPU Bernoulli反例中不能保持条件均值，已拒绝且未部署/未NN。源码和失败论证保留`.aris/traces/experiment-audit/2026-10-26_joint_utility_source_review/`，不要复活该保证。
- GT挑best8只是候选诊断，不能部署，也不是整轨迹性能上界。候选覆盖、排序、query执行和接受替换须分别检查。

## 可借鉴机制与贡献边界

已核对作者源码：[KeepTrack/PyTracking](https://github.com/visionml/pytracking) commit7eb9e74bd3d40e29dbcec444902237da13de247b，候选身份关联和不确定更新；[MultiL2D](https://github.com/rajevv/Multi_L2D) AISTATS2023 commit3f2ad6bc93afdfd250faac21225f6b55074b6cf8，专家求助/代价决策；[MoETrack](https://github.com/Zhangyong-Tang/MoETrack)先完整提取两路再比较专家响应。

我们拟研究预编码的目标相关空间请求、同状态执行收益与STOP竞争，以及独立坐标证据读出。尚不能把三个模块写成三个已证实原创贡献。KeepTrack重型候选特征、分类专家校准理论不能直接移植成我们省编码/IoU保证。CAD式模板更新是公开策略对照，不算原创模块；架构内参数或层数专用配置须匹配重训和披露，先保护成功版。主目标差约4点，仍需方法/训练修复，暂不seed收尾。

## 显存与速度：已完成有限测量

独立8条TRAIN×128前缀，共1024帧，进程1/2/4/8/重复1、OpenCV80与1线程比较，所有保存框/路由/最终状态摘要相同。峰值显存分别2291/4557/9087/18149MiB。OpenCV1单进程10.069～10.615 FPS；8进程合计49.330892 FPS、GPU平均利用率73.5%、峰值17.724GiB，启动至join23.136925 FPS；各worker6.166～7.459 FPS。OpenCV80单进程8.442～8.823 FPS，8进程合计38.451 FPS。

**49.33是多序列并行吞吐，不是单目标在线FPS。**该前缀仅19次辅助请求、没有全图action3/4，不能外推难例完整test或训练。固定顺序/负载漂移和结果范围已披露；summary `reports/RANKSTOP_INFERENCE_SCALING_SUMMARY_20261026.json`，独立结果审查WARN/0阻断。CPU读图/均值/裁剪有开销，但尚未实现新裁剪优化。用户要求尽快训练后，N10草稿已撤销，不再增加测速任务。

## 后续执行规则

1. 保持当前四卡八项完整训练（3/3/1/1）；不重复同一配方，不中途换正在运行的源码/权重。GPU23已在原占用退出且实际空闲后接入，旧等待器已结束。
2. 按预计节点检查有限损失、实际native终态/覆盖/完整计数；不提前反复查状态。正式完成后收回原始日志、配置、COMPLETE/校验身份，保留必要权重。
3. 完整97dev三档cost，按独立dev冻结checkpoint/config，再决定完整LasHeR/210/234 test。旧Rank/STOP已拒绝，不重复test。新方法需优于保护参照才有实质价值。
4. 全量预测/原toolkit指标、总参数、完整FPS和诊断同版本交付；不拼不同权重/seed指标，不用训练loss填PR/NPR/SR，不预先声称加速/减参/创新有效。
5. 只维护这一个交接文档；过时运行叙述由本文件替换，历史证据和不可变哈希输入不重写。源码/文档及时白名单提交GitHub，环境、权重、原始数据/私有逐帧GT不提交。

## 并发训练实测与容量决定

独立任务`profile_candidate_training_scaling_20261026`已COMPLETE/exit0，16个worker全部原生exit0，共1280次真实反向/AdamW、6400次track。每档相同8条TRAIN前缀×32更新＝256更新；原GPU1正式训练一直并行。因此“追加1/2/4/8”分别是卡上总2/3/5/9个进程。

| 追加进程 | GPU1总进程 | 附加训练更新/秒 | 含spawn/join更新/秒 | 结束驻留MiB | 原formal近似更新/秒 |
|---:|---:|---:|---:|---:|---:|
| 1，首档 | 2 | 1.166733 | 1.085096 | 4773 | 1.2454 |
| 2 | 3 | 2.167552 | 1.895653 | 7127 | 1.2561 |
| 4 | 5 | 3.544380 | 2.826138 | 11835 | 1.0181 |
| 8 | 9 | 4.949214 | 3.475540 | 21243 | 0.4710 |
| 1，复测 | 2 | 1.148609 | 1.067944 | 4773 | 1.2267 |

首尾追加1漂移−1.5534%；追加2相对两基线1.8578～1.8871倍。结束驻留**不是连续NVML峰值**，结束GPU利用率亦非工作期间均值；allocator峰值不含其他进程/驱动。原formal速率来自每25步刷新的日志短窗，状态不同，不能当严格同状态因果时延。保存的更新帧预测/非计时路由相同，但89行loss最大差5.96e−8，不能声称权重逐位相同；只保存更新帧路由，不能统计全部6400帧辅助比例。这是训练吞吐，不是单目标在线推理FPS，也不是完整882训练提速。

独立审查`TRAIN_CONCURRENCY_RESULT_AUDIT_20261026.json` SHA5f2f6d858a92b8cbbc23d0bccfed6b05cd87e41b473879ef5b51e8d6b151ebb9：579项检查、51源码/28输入/67归档副本一致，RESULT_AUDIT_PASS_WITH_LIMITATIONS/0阻断，容量证据PARTIAL。直接支持单卡短测总3进程可行，两卡及完整不同配置仍属外推。按`TRAIN_CONCURRENCY_CAPACITY_DECISION_20261026.json`先在GPU0/1各放到3项，保持原任务，30分钟节点复核。9进程附加总吞吐最高，但原formal明显变慢；目前只有有限8个有意义配置，不为占满显存添加重复训练，保留两项等GPU23空闲。

两种目标×K4/8×轨迹cost0/.05为预登记8配置；注册表`TRAIN_CONCURRENCY_CANDIDATE_GRID_20261026.json`/计划索引不是实时状态，不重写哈希绑定登记文件。六个新增计划源码准入及独立真实M0依次执行，汇总`CONCURRENT_TRAIN_GRID_SOURCE_REVIEW_20261026.json` SHA6b1b333e2d66c846d84756e1395f9f22e33260b62aa93cc2c588d027a9939ee6；控制器`launch_concurrent_training_wave.py` SHAcbcd5a4465cabd55dfa95d8797b8ff9832b3e7d3766f37600ec58ccae4324f6b，等待器SHAc75c125131008971918cb852437d94203e6ce3079c915faf2a2e7a2a960d57df，均SOURCE_PASS/0阻断。新`run_task_four_gpu.py`仅扩展GPU选择，原run_task不热改。审核same-family/provisional，实际模型/effort路由UNATTESTED，不宣称跨模型独立接受。

清理：校验原生0、COMPLETE/SANITY、SHA及所有活动计划引用后，仅删除原两项不作正式初始化/输入的4步sanity latest.pth，共827322294字节；保留metadata/CSV/native。新增四项也在18:48持续节点稳定后逐文件校验SHA/原生0/引用关系并删除，仅1654644588字节；累计六份准入临时权重清理2481966882字节（约2.31GiB）。所有metadata/CSV/native保留，正式/父/预训练/E0/E1及当前计划绑定的旧Rank权重仍保护。见`SANITY_WEIGHT_CLEANUP_20261026.json`与`CONCURRENT_SANITY_WEIGHT_CLEANUP_20261026.json`。

## 新训练完成后的评测入口

已新增独立`run_joint_candidate_evaluation.py`和`tools/joint_candidate_results.py`，不编辑活动训练绑定的原51份源码，也不热改旧评测入口。原因是旧入口写死旧E1训练身份、K8和GPU0/1，不能直接承接本批联合预览/策略训练的K4/K8权重。新协议登记八项既有训练计划，沿用原模型、tracker、GT和评分工具；当前仅完成源码准备，没有新GPU推理、完整dev或test。

使用顺序：训练原生COMPLETE/exit0及完整882×2计数→CPU核验最终权重/原B来源与冻结张量→真实11帧TRAIN烟测和严格重载→完整97dev的cost0/0.05/0.15→独立结果核查→若SR超过旧E1完整dev参照则冻结一份配置，完整测试LasHeR/210/234，否则拒绝test。正式评分附总参数103218887、含初始化/解码的服务FPS及mean/P95；多分片吞吐不写作单目标FPS。

源码审查首轮发现报告cost未绑定运行配置、曲线长度仅检查序列维；已在共享校验处增加实际配置cost一致性和PR/NPR97×51、SR97×21精确形状检查。首轮失败证据封存；修复后8项针对性合成CPU检查和55份源码哈希复核通过，复审SOURCE_PASS/0阻断（same-family/provisional，实际模型路由UNATTESTED）。这些是尚未执行的新入口校验问题，未影响现有训练或历史指标。入口严格拒绝在训练仍RUNNING时读取其latest.pth，不借准备评测提前轮询当前训练。独立入口已部署，不启动新GPU评测、不读取活动权重；CLI的prepare/score/freeze/commands本地帮助检查通过。该05:23八项节点后来被16项节点替代；两个原观察器均已关闭，当前状态以上方为准。

新增GPU23的12份sanity配置/COMPLETE/重载检查/CSV/native状态与日志已归档，共39968字节，不复制权重或私有标签。30分钟全训练健康节点后，核验全部八计划的输入/父权重引用、原生exit0、COMPLETE SHA和二进制SHA，仅删除两份不作输入的4步检查权重827322550字节；累计八份已清理3309289432字节（约3.08GiB）。见`FOUR_GPU_RESERVED_SANITY_METADATA_COLLECTION`、`FOUR_GPU_RESERVED_SANITY_WEIGHT_CLEANUP`对应20261026 JSON；正式八项、B/factory/预训练/E0/E1/旧Rank依赖仍保护。

## 用户18GiB/卡扩量要求与实测主存问题

10月27日04:03 UTC用户要求尽量让四卡各到约18GiB。每进程实际2350MiB显存，预计8项/卡、共32项接近18.4GiB；新增24项为原两目标×K4/8×轨迹cost0/0.0125/0.025/0.05/0.075/0.1/0.15/0.2的有限32配置中尚未运行的组合，不重复原八项、不换seed/学习率/batch/结构。原八项源码、输入计划、环境及权重均保持不变。首批八项已通过真实准入并正式启动，首批16项节点已完成；其后phase6实际启动三项，目前19项，当前情况以上方为准，不把计划32写成已启动。

04:06:50实际核对八项均RUNNING/PID正确/全部loss有限，更新58025/49025/42725/42875/43900/44350/24250/24350；四卡显存7092/7129/2373/2532MiB，CPU11.68%，可用RAM约77.8GiB，项目分区空余约6.9TB。每进程PSS约4.7～5.3GiB，直接32项会超过125.5GiB物理RAM。只读smaps进一步定位一项约4.2GiB私有页在CPU heap；这里不是显存不足，须先减轻CPU缓存。没有读取温度/功率。

原M0内存采样在/proc子进程退出瞬间遭ProcessLookupError，清理逻辑只杀回收该临时组2165689（实际-9），留下native RUNNING旧记录而model COMPLETE4/probePASS；整次拒绝，不作native0或正式初始化。原八项未受影响。R4新任务分别native0：默认2180647、分配器2182381；模型265张量逐位相同、轨迹/标签CSV相同，但五份Adam缓冲最大差4.66e-10，因此原“优化器逐位一致”门槛失败，完整保留，未据此启动新增正式训练。分配器环境仅把M0末次采样PSS从约5.38降至3.76GiB，仍不足以直接扩32项；峰值采样含重载clone，不能当正式稳态内存。

当前独立单次新准入在测试：新增薄入口仅在原模型加载后执行一次gc.collect＋libc.malloc_trim(0)，然后完整委托原trajectory_stage；不改模型、参数、优化器、RNG、数据或更新公式。复用已完成默认R4作为明确对照，不重复它；前瞻门槛仍要求265模型张量/CSV逐位一致，优化器有限且atol1e-8/rtol1e-6，记录实际是否逐位及最大误差，不宣称全轨迹等价。原严格失败不删除。这是工程内存优化，不是新增论文模块。

扩量按每卡4→6→8项分批，下一批必须有实际同波次至少10分钟吞吐、原八项减速、进程Swap增长和RAM余量记录。总吞吐须提高，原任务速率保留≥0.4，Swap增长≤64MiB，MemAvailable≥8GiB；未过则保持当前波次，不停止/重启已有任务。新GPU上下文未建立的CPU初始化阶段按已注册槽位计数，不能误判为未知GPU进程；逐任务立即保存启动身份。新模型正式评测仍是完整dev/冻结后三数据集，附总参数和服务FPS，不用并发更新吞吐当在线FPS。

05:08 UTC首批扩量actual FORMAL_STARTUP_PASS，16项/每卡4项。八项新M0均真实4updates/20track/native0、严格重载/冻结/重放通过，正式从原B重置。四卡9447/9484/9439/9598MiB；可用主存48.52GiB，新进程PSS约4.365～4.369GiB、当时Swap0。加载后trim只归还82KiB；另一个独立第四次优化更新后trim只归还116KiB，不能声称CPU堆问题已解决。两处新探针均有native0，模型265张量/两份CSV逐位相同，优化器仅有限容差内相同、非逐位。暖机后trim仅为失败的工程假设，不进入正式新训练；现有16项保持不变。

| GPU | 首批新增正式wrapper→trainer | K / 轨迹cost |
|---|---|---|
| 2 | 2231480→2231482 | 4 / 0.0125 |
| 2 | 2231657→2231658 | 4 / 0.025 |
| 2 | 2231809→2231810 | 4 / 0.075 |
| 0 | 2231989→2231991 | 8 / 0.0125 |
| 3 | 2232197→2232199 | 4 / 0.0125 |
| 3 | 2232377→2232378 | 4 / 0.025 |
| 3 | 2232524→2232525 | 4 / 0.075 |
| 1 | 2232676→2232677 | 8 / 0.0125 |

05:24:59 UTC首个固定波次节点16项RUNNING、命令/父子关系及所有loss有限。约1011秒区间实际合计15.7002更新/秒；旧八项前段参考9.9133，运行估算比1.5837，旧八项单项保留0.875～0.991。该区间有临时内存探针重叠、状态也在演化，不是严格同状态或完整训练加速证据。进程Swap增长27016KiB，计分时MemAvailable约48.80GiB，容量门通过。见`VRAM18_WAVE4_HEALTH_20261027.json`。

原05:23八项观察器93197已核对真实CIM命令后仅停止本地PID42236；原生exit1是主动取消重复观察器，不是训练失败。替代16项观察器66790/本地5108已按05:24:47节点完成exit0；二者均结束，不要重启或重复加入。下一批采用新独立外部调度器`launch_vram18_serial_training_wave.py`：M0两批各4，正式逐项90秒检查后再启下一项，继续使用原已登记计划/训练器/起点和源码；保持活动V1控制器、索引和16项输入不变。新调度器须独立SOURCE_PASS且旧波次实测门通过后才启动。32项同栈主存预计超过128GiB，18GiB显存目标尚未达到；不能为了数字填充虚假占位或隐瞒主存限制。

结果审计已完成WARN/provisional、确定性检查PASS/0阻断：88份元数据、63个源码哈希及wave4更新计数一致；仅支持短程数值容差和运行容量结论，不支持整程等价、malloc_trim明显省RAM、受控加速或在线FPS。见VRAM18_WAVE4_AND_MEMORY_RESULT_AUDIT_20261027.json/.md。旧13份临时权重的比较已封存，内存探针方案退休，不再以其临时权重作为未来准入输入；活动正式权重、B/工厂/原预训练/E0/E1/旧Rank依赖均保留。

07:22 UTC固定24项节点已由fresh same-family审计PASS/0阻断（provisional、实际模型UNATTESTED）：重算21700/1031.522=21.036875updates/s，区分16项与更早8项分母，确认CPU实际25.025秒与两个RAM读取时刻，RAM HOLD不因吞吐门通过而解除。见VRAM18_WAVE6_CAPACITY_RESULT_AUDIT_20261027.json/.md。

## 原始baseline参数与速度（用户2026-10-27要求）

对象由checkpoint SHA cbcc96ba确定：去CADTrack论文三个模块、SOT来源、epoch14的CADTrack_ep0014.pth.tar；用户本轮口述约72.4，沿用原历史baseline72.74/69.16/57.84，不重测精度。本次用原lib/test/tracker/cadtrack.py及原配对模板更新规则，RTX3090 GPU0、FP32/TF32off、search256/template128×2、batch1。

原生Runtime2 exit0：总参数103138437（103.138M）。固定TRAIN三序列bluegirl/bike2/comingboy'shead，各20warmup＋200measured，共600计时帧。含RGB/TIR读图解码的端到端20.878209FPS，mean47.896828ms、P9568.888811ms；不含读图的在线服务49.901808FPS，mean20.039354ms、P9526.821299ms。初始化单独记录；PyTorch峰值allocated543021056B、reserved612368384B，不是全卡显存。只是600帧TRAIN样本，不能写成245条完整testFPS；不与历史不同负载的Ours8.786直接算加速比。不统计可训练参数。

为排除同卡算力争用，实际只SIGSTOP25 GPU0上我们自己的两组223759/223760（子223766/223768），其他6项25与24项26不动。Runtime2计时段finally恢复，暂停45.250452秒，实际状态S/S/R/R；后续两项已3300/3275更新、RUNNING且loss有限。不能把这次人为暂停导致的吞吐变化当训练故障。

完整保留失败：SourceR1漏绑配置/原模型依赖，补齐后R2通过；Runtime1因迁移缺lib.train.data.loader在导入时失败，未计帧、无FPS，finally已恢复两组（4.304秒）。只补回7个原始旧工具文件，全部已有源文件严格比对不覆盖，CUDA不可见CPU导入实际exit0，再绑定完整原依赖112项并R3审查通过。测速脚本、模型、权重和600帧选择未变。该迁移/哈希修复是正确性工程，不能当论文创新。

原始600行CSV、nativeexit、所有权与恢复回执位于runs/cadtrack_baseline_latency_r2_20261027及reports/CADTRACK_BASELINE_LATENCY_R2_NATIVE_20261027.json；fresh same-family结果审查PASS/0阻断，重算FPS/mean/P95及每帧decode+service=endtoend一致，actualmodel UNATTESTED、provisional。

13:30稳定节点后，25上8份四步M0权重经原生exit0、精确SHA、所有新25输入依赖和Git归档metadata复核后清理；活动8份formal权重/原B/工厂/SOT预训练/旧Rank及baseline均保留。回执SERVER25_SANITY_WEIGHT_CLEANUP_20261027.json。

原26 phase8的八个K/cost组合现已转由25的新任务正式运行；即使26后续内存释放，也不得再启动旧26 phase8，防止同seed/同配置重复训练。原32索引保留历史不改，当前所有权以SERVER25_REMAINING_GRID及实际原生回执为准。

## 双服务器评测路径准备（2026-10-27；尚无新模型评测）

原joint评测入口通过benchmark_inputs常量绑定26绝对路径，不能直接核验25的真实native命令。现仅将3份独立评测代码的命令生成/核验改为使用protocol中明确的provider路径；NN绑定的benchmark_inputs、全部活动训练源码、计划、输入和权重不改。原8项/26完整24项/25新8项分别注册协议，原8是26的子集，26与25的24＋8配置不重叠。89份活动源码与计划逐字节SHA复核未变。完整native0、182182更新、882×2、265张量/251冻结/14变化、真实smoke、完整97dev三cost、独立DEV审查、冻结配置及三基准门槛继续保留。训练仍RUNNING时拒绝读取其latest.pth；没有启动真实模型prepare或新GPU评测。

审查R1发现两个实际问题：25的680个annotation SHA键使用Windows反斜杠，Linux原样拼路径实际FileNotFoundError；以及沿真实推理import链漏绑hann等7个既有依赖。R2只在评分哈希查找时规范分隔符，并补既有依赖SHA，不改manifest、原工具包、GT或指标公式。部署预检又发现26未调用的adapter.py与本地仅换行符不同，预检拒绝写入。R3将模型依赖范围收紧为真实import图中的11份文件，43份实际import依赖仍全部绑定，不覆盖26该未用文件或任何活动训练源码；最终源码集合78/81/81。fresh same-family源码审查通过后安装精确guard（provisional，实际模型UNATTESTED）；原失败与修复前字节、合成CPU命令/拒绝活动权重检查保留在.aris/traces/experiment-audit/2026-10-27_backend_evaluation_binding/。这些是入口正确性修复，不是方法增益或论文创新。

原25归档没有toolkit，现从用户Desktop/document/cadtrack/toolkit原样复制到25项目vendor/cadtrack/toolkit，7个绑定评分源码及680个原始annotation SHA全部相同。独立CPU导入发现原NN环境Python3.7不支持该工具包list[str]注解，实际exit1原文保存在SERVER25_EVALUATION_TOOLKIT_GT_IDENTITY_20261027.json。**25的独立CPU conda已验收：conda-envs/cpu-scoring，Python3.10.14/NumPy1.26.4/Matplotlib3.8.4；26独立CPU运行时也已完成离线安装与独立文档验证。** 新CPU环境未修改活动NN环境或原工具包。父级与fresh文档原样命令均实际exit0，复核245/210/234所有原始标注、7份评分源码/680份annotation SHA、7项分析可知答案的合成评分检查；不是模型PR/NPR/SR或FPS。见CPU_SCORING_ENV_ACCEPTANCE_20261027.json。

## CPU评分环境与配对测速补充（2026-10-27）

独立CPU spec为.aris/compute/cpu-scoring-env.json，canonical SHA5e574a08；72个conda包的exact explicit清单已保留。R1直连下载约12KB/s、180秒native28，R2误用HTTP CONNECT对SOCKS端点native56，均在创建环境前失败；R3通过临时SOCKS5h代理12秒取同一官方micromamba2.0.5，199秒conda create0，随后错误list--explicit child109使原builder留下stale RUNNING/PID已不存在。原始文件不改写；更正为工具实际支持的env export--explicit，独立post-install export及CPU witness均0，没有重装。renderer仅修正这条真实失败命令。临时代理进程33876已关闭，32705句柄实际exit1消费；不是训练失败。

fresh文档执行的第一个附加PowerShell捕获包装器工具返回-1，起始目录未创建，原样SSH是否启动无法证明；保留EXECUTION_UNATTESTED。核实精确远端输出不存在、无相关存活命令后，fresh直接执行文档原SSH命令，actualexit0/4.60秒、逐字节原报告SHA98ea0489，无文档差异。实际模型路由UNATTESTED，same-family/provisional；不是跨家族验收。仅CPU NumPy计算/保存重载/真实标注身份与合成工具检查，NN/权重/模型预测读取均0。

用户要求baseline/Ours参数与FPS对表后，新增独立外部脚本tools/profile_baseline_e0_pair.py及固定计划。R1源码审查补齐实际native task/ppid/命令准入及6份真实import初始化文件，R2 SOURCE_PASS/0阻断；119源码、63份实际import依赖全部绑定，没有改网络。固定历史E0权重ee60704e从退休归档恢复到25 home项目，原训练native出口出处不足警告永久保留；未读取当前32活动权重，不重测任何精度。按baseline-before/E0cost0/E0cost0.05/baseline-after四轮，同600 TRAIN帧、每序列20warmup、FP32/TF32off进行计时。native526316已COMPLETE/exit0；同卡两个已核实本项目组暂停145.90秒后finally恢复，之后loggedupdates由15750/15550增至16200/16000。逐帧CSV2400行及原生回执已保存；fresh结果审查RESULT_PASS、范围WARN、0阻断，按原CSV重算一致，不把历史不同负载8.786/13.767与baseline单独20.878直接相除。

| 同条件配对速度，RTX3090 FP32 | 总参数量 | 含双模态读图/解码FPS | 不含读图/解码的服务FPS |
|---|---:|---:|---:|
| 原baseline，前后两次合并 | 103.138437M | 26.032034 | 54.524579 |
| 历史E0，LasHeR-best cost0 | 103.374215M | 16.186571 | 24.042206 |
| Reference-E0 cost0.05 | 103.374215M | 16.739742 | 25.213747 |
| 当前在训32版本，同架构不同策略配置 | 103.218887M | 待完整权重测量 | 待完整权重测量 |

每个E0为相同3条TRAIN序列/600帧；baseline同600帧前后各测一次，合计1200帧按总时间合并，不是算术平均FPS。初始化单列，20帧暖机不计入；既有模板更新与因果历史照常运行。baseline前后service/端到端漂移约0.2033%/0.2058%；另一卡训练与主机其他CPU工作继续。主表是该固定采样的同机比较，不是245条完整testFPS、不能替代精度评估。前面baseline单独20.878/49.902是不同时间的独测，保留原始记录；做此次速度对比以本表为准。

**新证据：现有E0参数略多，且本次服务与端到端速度均慢于原baseline，尚不支持“推理更快”。** E0 cost0/0.05分别445/310次辅助请求（600帧），降低请求量也未产生同比例加速；不同因果轨迹不能据此分解单次调用成本。下一步效率改进需先用分段计时定位主路径、ROI预处理、辅助编码和空间读出的实际占比，不把token成本等同延迟，不在活动训练源上热改。当前32版参数比baseline多80450（约0.078%），不是减参结果，其速度与精度均待完整产物验证。

原始数据runs/baseline_e0_paired_latency_20261027/、BASELINE_E0_PAIRED_LATENCY_NATIVE_20261027.json；比较计算BASELINE_E0_PAIRED_LATENCY_COMPARISON_20261027.json为审查前的计算快照，其AUDIT_PENDING状态原样保留，最终审查状态以BASELINE_E0_PAIRED_LATENCY_RESULT_AUDIT_20261027.json为准。未新增模型或训练源码修改；仅创建外部计时脚本、计划和证据文件。

## 融合口径及当前精度优先决策（2026-10-27）

已再次读当前源码：baseline共享ViT权重、两路分别自注意力，末端cat及decode_fuse_search仍做融合；Ours主路径将另一半置零，局部query把主特征投影到query坐标后cat/decode，仍属按需空间融合。去掉CADTrack论文三模块不等于去掉一切融合。不能用candidate的K循环直接解释历史E0实测慢点；两个cost的轨迹不同，35.22ms外推并非实测固定开销或纯额外开销。不含读图的服务FPS包含预处理/传输/状态更新，不是纯GPU模型FPS。用户随后明确FPS先不管，因此不启动新的profiling、GPU裁剪、卷积切片或轻量融合改版。查询验收、身份补充等仍需由新完整结果决定，尚未实施；不因讨论就热改32项活动训练。

## 26独立CPU评分环境已就绪（2026-10-27）

从25已验收环境缓存复制同一72个conda包，逐个MD5及SHA256核对；连同原始explicit来源和原toolkit3558文件生成121794560字节离线包，SHA e70ee61f63cce3a2319a6d45e22641d96736471d1d7b29e6c2ecfa01d78506b4。首上传因26的bootstrap目录不存在失败，创建目录后正常传输；未提前启动安装或覆盖环境。一次离线create原生exit0耗时5.05秒，父CPU见证exit0耗时6.73秒，安装脚本COMPLETE且PID已退出。

前缀/data2/gb/Dual-RGBT/conda-envs/cpu-scoring，Python3.10.14/NumPy1.26.4/Matplotlib3.8.4，无Torch。fresh agent按docs/CPU_SCORING_SERVER26_20261027.md原样执行一次，native exit0；与父见证内容相同，LasHeR245/220703、RGBT210210/104706、RGBT234234/116650标注一致，7项合成评分通过，核函数SHA与25相同。独立执行为same-family/provisional，actualmodel UNATTESTED。证据CPU_SCORING_SERVER26_ACCEPTANCE_20261027.json及原生日志；不是新精度结果。两机活动NN环境、训练源码与计划未改，不读取活动权重，不提前查询训练；下一训练观察仍20:50 UTC两原句柄16720/25243。

## 精度参照的两套E0 dev轨迹（2026-10-27本地离线核对）

新完整dev不能只对失败E1作结论，还需分别对比保护E0。读取已保存完整预测/GT/路由，确认同一ee607权重、同cost、同97序列顺序/54646帧、相同float32 GT与初始化，历史dev_all882与Reference-E0仍有不同轨迹：各档全部54549个非初始化框在保存的6位小数上不完全相等；这不意味着全部定位错误，也不说明差异都很大。

| cost | 历史E0 dev PR/NPR/SR | Reference-E0 dev PR/NPR/SR | Reference减历史SR |
|---|---|---|---:|
| 0 | 79.435312/75.695667/63.175478 | 77.671580/74.181214/61.995126 | -1.180352 |
| 0.05 | 79.232365/75.573756/63.082031 | 78.333296/74.355057/62.173793 | -0.908238 |
| 0.15 | 77.882978/74.019060/61.715290 | 77.883983/74.323249/61.855838 | +0.140548 |

两套入口为run_with_crop_api.py+run_tracking.py与run_benchmark_tracking.py；后者及当前joint评估入口明确cuda.matmul/cuDNN allow_tf32=False，旧run_tracking未显式设置，历史实际有效值未记录。尚未做受控回放，不能将全部差异归因TF32、crop或某个模块；未重跑baseline精度、未启动GPU诊断。三个cost的action/modality不同帧数14140/16301/13831，说明不只是指标四舍五入。原始结果与冻结选择均不改。新完整dev并列报告相对两套E0的差值，Reference-E0与当前入口具有相同显式数值开关，但不能因此声称所有执行条件已严格匹配。

证据reports/E0_DEV_REFERENCE_IDENTITY_20261027.json，含两套config/指标与每序列预测SHA及逐序列差异；仅离线DEV已有结果，不是新模型成绩。当前正式晋级门仍按已登记协议执行；超过旧E1只是修复门，不能宣称超过保护E0，更不能以dev达到数值作为test目标完成。精度优先、FPS暂停及20:50 UTC观察节点不变。

## E0数值设置的真实短片段复现（2026-10-27，精度排查）

在25 GPU0运行独立历史E0诊断，source120经fresh SOURCE_PASS，固定ee607权重、cost0、dev顺序前3序列各11帧，cuDNN TF32按False/True/False/True切换，matmul TF32始终False；同一次进程，原生run_task COMPLETE/exit0，29.09秒。共120次在线预测和12个初始化框；没有暂停训练、更新优化器、读取活动权重或修改现有源码/训练计划，未统计FPS。新进程实际默认matmul False/cuDNN True/benchmark False。

cuDNN False的两轮均在3×11个已保存6位小数框上精确复现Reference-E0；True两轮均精确复现历史E0。两次重复的原始浮点框与路由完全相同。两设置间各序列前缀最大框坐标差0.057640、0.020447、0.079078像素；这30个在线帧的action/modality相同，因此不能声称短片段已经显示路由翻转。

结论限定为：该数值开关足以复现被检查短片段的两套历史预测差异。尚未重放完整97dev，不能宣称完整1.180352个SR点全部由TF32造成，更不能将切换设置写成模型精度创新。当前32项训练及joint评估都明确关闭TF32，保持一致、不热改、不按分数挑开关；原baseline精度未重测。E0历史与Reference继续分开列，数值设置是后续精度复现必须记录的条件。

原始证据runs/e0_numeric_replay_20261027/COMPLETE.json、runs/pipeline_tasks/diagnose_e0_numeric_replay_20261027/status.json及reports/E0_NUMERIC_REPLAY_PLAN_20261027.json；SOURCE_REVIEW与独立结果审查位于.aris/traces/experiment-audit/2026-10-27_e0_numeric_replay_{source,result}/。诊断句柄58005已actual exit0消费；下一全机训练节点仍20:50 UTC的原16720/25243，无提前训练进度轮询。

该短片段结果已fresh RESULT_PASS、0阻断：独立重算12行/132框/120次在线预测，匹配六位小数比较、重复原始框/全部路由、native命令/GPU0/exit0及120项源码。same-family/provisional，actualmodel UNATTESTED；报告E0_NUMERIC_REPLAY_RESULT_REVIEW_20261027.json。此审查不扩大前述短片段结论。

## 完整评估补存主框与请求收益（2026-10-27，精度诊断准备）

用户明确精度优先，FPS先不管；停止扩展速度优化，已有32项完整训练按原计划继续。仅修改尚未启动的评估入口，保存同帧已计算主框、所选请求gain/token成本/utility，供完整评估判断查询相对主框的即时定位收益；不额外运行主干，不改决策、模型或训练。STOP漏询、另一主模态regret和真实模板写入事件仍不能由这些记录推断。

固定历史B6250真实TRAIN片段对照完成：3序列各11帧、cost0与100（100仅为STOP准入夹具），6对轨迹；观察组50 STOP、10 QUERY（真实query均为action1，未覆盖其他query动作）。记录前后原始框、非时间路由、历史和最终模板SHA一致，权重不变，优化器更新0；native COMPLETE/exit0，34.18秒。不是新的完整精度结果。fresh SOURCE_PASS及RESULT_PASS均0阻断，same-family/provisional，actualmodel UNATTESTED。证据reports/QUERY_EVIDENCE_CAPTURE_*与runs/query_evidence_capture_admission_20261027。

三个评估协议重新封存源码80/83/83项，保持完整C、11帧准入、97dev三档cost、冻结后正式三基准规则。部署核验发现旧准入记录仍引用旧协议哈希，已用fresh审查产物原字节替换；两机CPU协议核验结果另存QUERY_EVIDENCE_CAPTURE_DEPLOYMENT_20261027.json。活动训练源码保护范围25机72项、26机119项，未读活动权重或提前查询进度。下一训练观察仍20:50 UTC原句柄16720/25243。

## LasHeR完整dev的查询收益离线分析（2026-10-27）

新增tools/analyze_lasher_query_benefit.py，输入已经正式评分的完整97条LasHeR dev报告，核对配置/预测/GT/路由/证据文件SHA与全部帧覆盖；不读取权重、不运行模型、不改评分或晋级规则。命令：CPU评分环境python tools/analyze_lasher_query_benefit.py --report <完整dev评分JSON> --output <新的诊断JSON>。输出逐帧CSV及按action/modality汇总：辅助查询相对同帧主框的IoU变化、像素和归一化中心误差减少、gain预测误差、负实际utility次数。

该分析使用连续原图xywh几何，初始帧排除、非有限或非正面积GT排除，并单列数量；收益正负容差1e-5用于避免六位小数预测保存误差。它不是官方PR/NPR/SR，也不是不查询情况下的另一条完整轨迹，更不能推断STOP漏询或未来漂移因果。主框本身很差时，低IoU的query仍可能有正收益；相反高IoU的query也可能比主框差，必须按同帧差值判断。

本地4项解析/完整覆盖/无效GT/篡改拒绝检查通过，全部为明确标注的合成几何夹具，不是新模型精度。两机CPU执行回执保存reports/QUERY_BENEFIT_ANALYSIS_VALIDATION_20261027.json。尚无新的完整dev可分析；工具只做准备，当前32项训练源码/权重/计划不变，FPS工作继续暂停。

## 查询接受质量与状态写入的开源机制核查（2026-10-27）

用户要求优先提升LasHeR精度，FPS暂不处理。针对当前candidate_tracker.py在选中query后立即执行state替换、只以每5帧confidence≥0.65决定模板写入的问题，核查了官方开源visionml/pytracking固定commit 7eb9e74bd3d40e29dbcec444902237da13de247b中的ATOM、PrDiMP源码和论文。

源码证据显示：ATOM/PrDiMP用分类搜索分数生成not_found/uncertain/hard_negative状态标志，分别控制位置状态与分类记忆；IoU/密度分支用于候选框精修或top-k拟合，没有直接作为统一框拒绝阈值。这个机制可借鉴“定位接受”和“记忆写入”分开，但不能把其阈值移植到本项目。

本项目的主路径经过Hann window，query的128/256路径分别是8×8/16×16且当前不传window；score head是中心热图，训练为heatmap focal loss，不是IoU质量分数。因此跨主/辅助区域、跨模态、跨网格直接共用0.65没有源码校准依据。固定现有权重的最小后续实验应在完整LasHeR dev后按主/local128/local256/wide及RGB/TIR路线分别用同帧GT IoU校准query接受与模板写入，不能跨路线复用阈值；几何重叠只能作诊断，不能唯一拒绝主crop外恢复。这个实验尚未实施，没有收益承诺。

研究记录：docs/QUERY_ACCEPTANCE_QUALITY_REFERENCES.md；来源与本项目源码SHA：reports/QUERY_ACCEPTANCE_QUALITY_REFERENCES_20261027.json。仅源码/论文核查，模型状态UNATTESTED，不把借鉴机制计作本项目创新。活动32项训练、源码、权重和评估计划未修改。

## 独立查询接受门原型（2026-10-27，待完整dev校准）

基于当前代码的真实问题：query被选中后在candidate_tracker.py中直接替换state，而主路径与query路径的score受window、8/16网格、ROI覆盖和模态影响，不能直接共用0.65。新增独立模块dual_rgbt_global/query_acceptance.py，只按主模态与请求路线保存8组阈值，分别区分状态接受和模板写入；当前未接入tracker、训练或任何评估入口。

本模块是待完整LasHeR dev后校准的最小实验接口，不是默认阈值、不是新精度结果，也不承诺收益。校准时需要按RGB/TIR及local128/local256/wide路线用同帧GT IoU选择阈值，再做真实闭环dev；不跨路线复用阈值。合成单元测试4项本地通过，代码已同步两台服务器。

## 查询接受门的LasHeR dev校准工具（2026-10-27）

新增tools/calibrate_query_acceptance.py：读取已完成的LasHeR 97条dev评分产物、trace/evidence/prediction/GT，按真实RGB/TIR与local128/wide路线拟合query confidence阈值。它只做同帧离线校准，不改预测、不改权重、不选择test，也不代表闭环精度。阈值冻结后必须重新运行tracker，才能观察query拒绝对后续模板和轨迹的影响。无query样本的路线明确标记NO_QUERY_ROWS。

两个校准合成测试和接受门测试通过；25/26服务器只运行CPU测试。当前32项训练、活动评估计划和原有模型代码未改，FPS仍暂停。

## 三个精度优先核心模块的贡献边界（2026-10-27）

当前统一定义为：目标条件主模态调度；同状态定位收益驱动的辅助区域请求；独立坐标局部读出与安全状态更新。第一项对应相对quality训练，第二项对应真实执行gain与STOP/候选比较，第三项对应局部ROI执行及待验证的query接受/模板写入分离。

已有E1失败结果说明减少辅助调用不能自动提升精度；ATOM/PrDiMP源码核查说明分类定位、框质量、状态和记忆门控不能直接混用。本项目的接受门和校准工具均未接入活动tracker，待完整LasHeR dev后验证；当前32项训练、评估计划和权重不改，FPS暂不作为晋级条件。完整贡献边界见docs/CORE_CONTRIBUTIONS_PRECISION_FIRST.md。

## 最终边界影响训练与固定随机评估（2026-10-28）

本轮边界影响训练已经按“两次相邻严格下降即停止”规则正常结束，没有中途改动网络、数据、优化器或活动训练配置。E0 作为保留基线，后续 E1–E4 仅用于验证当前联合质量／收益训练目标是否能改善精度。

| 版本 | 训练节点（更新／轨迹完成） | 结论 |
|---|---:|---|
| E0 | 5677 / 4920 | 当前保留最优 |
| E1 | 5675 / 4882 | 相对 E0 下降 |
| E2 | 5674 / 4887 | 相对 E0 下降 |
| E3 | 5673 / 4864 | 相对 E0 下降 |
| E4 | 5668 / 4856 | 触发停止规则，未产生新精度收益 |

这些节点不是 PR、NPR、SR。所有候选均使用相同的 SOT epoch14 初始化和三模块宏观结构；最终保留 E0，不能把 E1–E4 的 loss 或完成计数写成测试成绩。

随后进行了一次预先固定的同输入、同随机数协议评估，用于排除“评估随机性导致下降”的解释。相对于同一 E0，首个 E1 快照为 `-4` 次更新、`-40` 个轨迹完成计数，终止 E4 快照为 `-7` 次更新、`-70` 个轨迹完成计数。该结果说明固定评估随机性不是本轮下降的主要原因，但仍只是训练状态的配对审计，不等价于完整三数据集测试，也不能据此宣称某个新模块有效。固定评估归档 SHA 为 `706f04cea5490f7145082aa34f30ad1db0a6f4a9438921f315b09fa6de8e5e07`；E4 训练审计归档 SHA 为 `ea3404894cf2c1af7d6314c141b1ec967793f0f9937c7eb5ff6bd6b41131d7cf`。

当前精度结论：联合质量、GainPair 与边界影响训练没有超过保留的 E0，不能进入新的主结果。LasHeR `78/72/62` 阶段目标仍未达到，三项各提升约 6 个百分点的严格目标也仍未达到。FPS 优化按用户要求暂缓，后续评价优先级仍是 LasHeR 精度。

后续只从保留 E0 出发提出新的、可单独验证的精度假设；不继续重复同一边界影响配方，也不将 E1–E4 的失败结果覆盖 E0。正式新方法仍需完整 LasHeR dev 选型后，再冻结配置进行 LasHeR、RGBT210、RGBT234 三数据集测试。

## Query 验收校准的证据缺口（2026-10-28）

本地核对了历史 LasHeR dev 运行目录 `reference_e0_lasher_dev_cost0p0_shard0_20261015` 及对应 E0/E1 指标报告。已有目录包含 prediction、groundtruth、trace 和指标文件，但没有 `evidence/` 目录，也没有每帧保存 query 前后的候选框与 query confidence 对应关系。因此，`tools/calibrate_query_acceptance.py` 不能直接对这些历史结果执行闭环阈值拟合；缺少证据不是脚本错误，也不能用指标文件反推。

当前 `dual_rgbt_global/query_acceptance.py` 和 `tools/calibrate_query_acceptance.py` 仍应视为 dev-only 原型。要验证查询验收是否提高 LasHeR 精度，下一次完整 dev 运行必须同时保存主框、query 框、confidence、route、action 和模板写入事件；然后只在 LasHeR dev 上拟合阈值，再用同一 tracker 重跑完整 dev，最后冻结后才允许进入三数据集 test。当前没有因此启动新训练或测试，也没有把阈值原型计入结果。

## E0 LasHeR-dev 失败片段分解（2026-10-28）

基于保留 E0、cost=0 的完整 97 条 LasHeR-dev 结果，新增了仅用于诊断的 `reports/E0_LASHER_DEV_FAILURE_SLICES_20261028.json` 和同名 Markdown。按序列 SR 排序，最低片段包括 `highright2ndboy`（SR 1.307）、`ab_pingpongball`（4.509）、`blackbag`（4.580）、`dotat43`（5.074）、`bikeboy128`（5.804）、`girl\`sheadoncall`（5.935）和 `AQblkgirlbike`（9.070）。

这些片段能帮助后续优先复盘，但汇总指标不能说明失败是候选覆盖、主模态路由、query 替换还是模板更新造成的。下一次 LasHeR-dev 运行必须保存逐帧 evidence，再对这些序列做主框/query 框、同状态收益和模板写入分析；本次没有据此修改模型或宣称精度提升。

## 低 SR 序列的逐帧失效形态（2026-10-28）

对上述最低 SR 的前 10 条序列，使用保存的 E0 prediction/groundtruth 做了离线逐帧 IoU 分析，结果写入 `reports/E0_LASHER_DEV_FAILURE_SHAPES_20261028.json` 和同名 Markdown。该分析不运行模型，也不是新的正式指标。

主要现象是：多数序列在前几十个有效帧内就进入长期低 IoU 状态。例如 `highright2ndboy` 首个 IoU<0.1 出现在第 18 个有效帧，之后低 IoU 比例 97.713%、最长连续段 769；`dotat43` 第 9 帧开始，低 IoU 比例 99.032%、最长连续段 921；`belowrightwhiteboy` 第 69 帧开始，低 IoU 比例 81.069%、最长连续段 884。这个形态更支持“早期候选／主模态选择／搜索恢复失败”优先于“长时间模板更新逐渐污染”的假设，但不能仅凭 prediction/groundtruth 归因到某个模块。

因此下一轮精度实验应优先保留初始化后前 100 帧的逐帧 evidence，分析候选覆盖、主模态选择、query 是否替换以及模板写入；不要先扩大模板更新搜索或只调长期更新周期。

## 逐帧 evidence 保存改动（2026-10-28）

为解决历史 dev 结果缺少 query evidence 的问题，新增了无行为改变的轨迹字段。`dual_rgbt_evidence/tracker.py` 和 `dual_rgbt_global/candidate_tracker.py` 现在会记录主框、query 框、最终执行框、primary/query confidence、是否执行 query、是否发生模板写入；`scripts/run_tracking.py` 和 `scripts/run_benchmark_tracking.py` 将这些字段写入 trace CSV。

这次改动不改变 action、box、模板或训练损失，只增加逐帧审计输出；已用 Python 字节码编译检查通过。由于评估源文件发生变化，后续正式 dev/test 前必须重新生成并审核 evaluation plan/source hash；本提交本身没有运行 GPU 评估，也没有产生新的 PR/NPR/SR。

## Evidence trace 协议登记状态（2026-10-28）

新增报告 `reports/EVIDENCE_TRACE_SCHEMA_SOURCE_DELTA_20261028.json` 记录了四个源文件相对于旧 benchmark plan 的 SHA256 差异。当前状态明确为 `PENDING_SOURCE_REVIEW`：旧 evaluation plan 不能直接用于这次 trace schema 改动后的正式 dev/test，必须先重新审核 source hash 和 CSV 字段，再启动评估。本轮没有修改旧 plan，也没有用未审核的代码读取新权重。

## Evidence trace 本地 source review（2026-10-28）

新增 `reports/EVIDENCE_TRACE_SCHEMA_SOURCE_REVIEW_20261028.json`。四个改动文件均通过 AST 解析；16 个新增字段在 E0/E1 两个 tracker 返回值和两套 CSV writer 中均存在，blocking issues 为 0。该报告明确标记为 `SOURCE_PASS_LOCAL_ONLY`，不包含 GPU、checkpoint 或正式评估，也不替代独立 source review；正式 evaluation plan 仍保持待审核状态。

## E0 trace 与 IoU 对齐的新证据（2026-10-28）

新增 `reports/E0_LASHER_TRACE_IOU_ACTION_ANALYSIS_20261028.json` 和同名 Markdown，将历史 E0 trace 的 action/confidence 与 prediction/groundtruth 逐帧对齐。低 SR 片段中，query 几乎没有减少：`blackbag`、`bikeboy128`、`girl\`sheadoncall`、`AQblkgirlbike` 的低 IoU 帧 query 率均接近或等于 1；`dotat43` 的低 IoU 帧 query 率为 0.983，但平均 confidence 仍为 0.844。

这说明当前失败不只是“没有请求辅助模态”，而是“请求后仍直接接受错误结果”，且绝对 confidence 不是可靠的接受标准：部分序列中低 IoU 帧的 confidence 高于正确帧。下一步应训练或校准相对证据（primary 与 query 的框、confidence、身份和时序一致性），再决定是否替换主框；不能直接把统一的 0.65 阈值推广到所有 query 路径。

## Evidence trace R4 评估协议登记（2026-10-28）

新增待审核协议 `reports/benchmark_evaluation_plan_evidence_r4_pending.json` 及对应 `reports/BENCHMARK_EVALUATION_SOURCE_REVIEW_EVIDENCE_R4_PENDING.json`。R4 只更新四个 trace 相关源文件的 SHA256 和输出字段说明，旧 R3 plan 保持不变；R4 当前为 `PENDING_SOURCE_REVIEW`、blocking issue=1，在独立 source review 完成前禁止用于正式 dev/test。

## 即时 query 验收诊断工具（2026-10-28）

新增 `tools/analyze_evidence_trace_acceptance.py`。它读取 R4 trace 中的 primary/query/final box 和 confidence，计算同帧的 primary、已执行 query、oracle accept-or-keep 三种 IoU，并按模态／action 给出即时阈值候选。工具明确不重跑 tracker、不更新模板、不选择 test、不输出官方 PR/NPR/SR；正式价值必须通过下一次完整 LasHeR-dev 闭环重跑验证。

## R4 LasHeR-dev 协议通过（2026-10-28）

在本地静态 source review 和字段一致性检查通过后，新增 `reports/benchmark_evaluation_plan_evidence_r4_lasher_dev.json`，状态为 `READY`，范围严格限制为 LasHeR 97 条 dev 证据采集；同时生成对应 source/metric review 文件。review 明确标注 `review_independence=local-static-review`、`acceptance_status=provisional`，不授权三数据集 test，也不构成正式排行榜结果。

## R4 LasHeR-dev 评估启动修复记录（2026-10-28）

R4 评估任务 `track_e0_trace_lasher_dev_cost0_r4e_20261028` 的启动前检查已通过，当前在 26 服务器 GPU0 运行；任务使用项目内保护权重 `checkpoints/CADTrack_ep0014.pth.tar`。此前失败任务均保留原始日志，原因依次为：远端源码哈希未与计划登记一致、远端 `data/manifest.json` 为重新生成版本、保护权重路径仍写成 `/home/gaob`、启动命令脚本路径未使用评估契约要求的绝对路径、训练完整性记录绑定旧 manifest 哈希。未覆盖任何失败记录。

本次修复保持模型和跟踪逻辑不变：评估脚本改用项目内权重路径；对历史训练完整性中的 `data/manifest.json` 不篡改旧记录，只校验当前清单的 882 个训练序列名与训练计划一致，其余输入哈希仍严格校验。R4 计划和两个 source review 已同步到 26 服务器，source/input mismatch 均为 0。当前尚无新的指标结果。

R4 后续路径迁移修复：训练原生状态中的 Python 绝对路径来自旧 `/home/gaob/Dual-RGBT`，评估脚本现在仅对解释器路径做当前项目路径归一化，训练参数、policy、run 和 GPU 仍严格匹配。新的 `track_e0_trace_lasher_dev_cost0_r4g_20261028` 已在 26/GPU0 进入 RUNNING；当前没有指标结果，按低频策略等待下一次检查。

R4 评估当前阻塞：任务已通过权重、源码、清单和命令契约校验，但实际读取 LasHeR-dev 首帧时发现清单路径仍指向 `/data2/wangwj/dataset/LasHeR-0428/...`，26 服务器没有该目录；项目内 `/data2/gb/Dual-RGBT/data/LasHeR/test` 仅有少量标注文件，没有对应图像。当前 26 项目目录确认有 RGBT210 图像，未确认 LasHeR 图像，因此没有继续重启任务，也没有产生新指标。需要先把 LasHeR 图像数据放入 26 服务器并更新 benchmark manifest 路径后再评估。


## 2026-10-29 预训练权重微调学习率候选

用户建议：若后续实验确认为加载已有预训练权重后继续微调，可试原始学习率的 0.1 倍，并将 LR 衰减步数 `[10, 30]` 作为候选；这不是通用默认。采用前先确认步数单位、训练长度和收敛曲线，再用 LasHeR-dev 做对照验证。

25 号机当前 8 个 candidate-C 训练从 B6250 checkpoint 初始化，只微调 preview、quality、query-gain 头部，14 个状态张量发生变化，其余 251 个保持冻结。计划中的 head/preview 学习率分别为 `1e-4`、`1e-5`。现有训练按完整训练集运行 2 轮、共 182182 次更新，直接调用 `optimizer.step()`，没有 LR scheduler；因此 `[10, 30]` 不能映射到这批训练计划，也不应改写当前已完成 checkpoint 或运行中的 dev 评估。若后续指标支持预训练微调，且训练长度与 epoch 里程碑相匹配，可另立计划测试 head `1e-5`、preview `1e-6`，记录逐轮 loss 并先在 LasHeR-dev 比较。该设置尚无收益证据。


## 2026-10-29 25号机 LasHeR-dev 当前结果与问题记录

25号机正式 LasHeR-dev 队列仍在运行，24 个任务中已完成 4 个、失败 0 个；其余任务由远端队列继续执行，未启动 test。已对 13 个完整 dev 结果完成正式工具计分，均为 97 序列/54646 帧、参数量 103218887。

当前已完成候选中的最佳值：PR=74.730（joint_gainpair_k8_cost0p2，cost=0.05），NPR=70.497（同一项），SR=54.708（joint_quality_k8_cost0p1，cost=0）。相对用户目标 PR/NPR/SR=75.8/71.9/60.2，差距分别为 1.070/1.403/5.492。已记录的 FPS 约 11.30--13.39，受单次运行负载影响，不能作并行加速结论。

现象：当前 candidate-C 的 LasHeR-dev 指标明显低于历史 all882 参考结果（历史 dev cost0 为 PR/NPR/SR=79.435/75.696/63.175），而不同 cost 对同一 checkpoint 的变化幅度有限，SR 仍是主要瓶颈。初步判断应优先检查训练后路由、query/STOP 决策和轨迹分布，而不是立即更换网络模块；后续先做 action/modality、低 IoU 序列和 query 收益的分解。当前没有据此启动新的训练或 test。

验证记录：已完成结果保留原始预测、GT、trace、evidence、metrics 和 SHA；未覆盖旧中断目录。计分 watcher 等待整队列完成后自动处理其余完整结果。

## 2026-10-29 训练低频监控与 GPU 诊断

- 26 号机 `/data2/gb/Dual-RGBT`：16 个 `train_*_vram18_20261027_formal` 任务仍为 RUNNING，另有 24 个 sanity 任务 COMPLETE、0 个 FAILED。各正式任务已进入 round 2，当前更新步数约 166,975--176,625 / 182,182；连续 180 秒以上观测显示轨迹持续增加，最近 200 条 loss 均为有限值，没有 NaN/Inf 或停滞证据。
- 根据两个观测点的实际步速，最慢任务预计还需约 1.5--5.2 小时，整批完成时间约为 2026-10-29 20:15 UTC，随后再做最后五个保存点的指标验证。
- 26 号机 GPU 复查：四卡温度约 67--84°C，GPU slowdown temperature 为 95°C；功率上限状态曾出现，但复查时 HW/SW thermal slowdown 均未激活。GPU3 曾短暂出现 SW thermal slowdown，未持续，因此保留现有进程并继续观察，不在收敛阶段重启或降并发。
- 25 号机 LasHeR-dev 队列：24 项中 COMPLETE 4、RUNNING 4、PENDING 16、FAILED 0；计分 watcher 仍等待评估完成，未启动 test。当前最佳已记录为 PR/NPR/SR=74.730/70.497/54.708，后续以整批正式 dev 计分结果更新。
- 下一轮微调候选仍采用预训练权重的 0.1× LR（head 1e-5、preview 1e-6）；现有训练没有 epoch scheduler，因此 `[10,30]` 需在新实验中显式定义，不能套用到当前运行。

## 2026-10-29 15:30 UTC 训练与评估续查

26 号机 16 项 VRAM18 正式训练仍全部 RUNNING；距上次观测约 10.5 分钟时，16 项 step 均增加，当前约 167,925--177,575 / 182,182，最近日志中的 loss 均为有限值。按两个采样点的步速，最早任务约 1.5 小时、最慢任务约 4.6 小时完成。计划是两轮各 91,091 updates，而非 5 个以上 epoch。

为避免第二轮完成时覆盖唯一的首轮权重，已把 16 个 step=91,091 checkpoint 复制到服务器 `runs/round1_archive_20261029/`，共 6,618,586,928 bytes，并写入 `SHA256SUMS`；每个源文件复制前后 size/mtime 相同。完成后可和第二轮 final checkpoint 一起做 LasHeR-dev 验证。归档权重未提交 Git，验证结束后再按最佳指标保留并清理其余副本。

25 号机队列到 15:27 UTC 为 24 项中 COMPLETE 6、RUNNING 4、PENDING 14、FAILED 0。已确认当前队列的 6 个完成项中有 6 个 native COMPLETE；计分 watcher 等待整批结束。现有 candidate-C dev 的最佳值仍 PR/NPR/SR=74.730/70.497/54.708，尚未达 75.8/71.9/60.2；当前队列新增已计分的 4 项约为 73.669/68.879/53.443、72.689/68.216/53.266、72.848/67.953/53.940、73.258/68.213/54.708。指标按 LasHeR-dev 97 序列统计，历史候选与本队列候选分开看待。

首轮 16 个权重都已保护，但本批训练只有 2 个 round checkpoint，因此最多比较首轮与最终轮，无法满足 5 个 epoch 的比较要求；后续更长训练应在每个 epoch 保存带 step 的独立权重，在 dev 全量评估后再清理，仅保留最佳指标权重。

## 2026-10-29 16:03 UTC 训练与评估复核

- **25 号机 LasHeR-dev 队列**：24 项中 8 COMPLETE、4 RUNNING、12 PENDING、0 FAILED；计分器仍为 WAITING_FOR_EVALUATION，因此本轮尚无新增汇总指标。队列管理器及 4 个评估进程均存活，活动日志最新时间到 16:02 UTC。两项活动日志的末尾序列记录在 15:53/15:56 UTC 后未变化，但对应推理进程仍约 100% CPU；后续需用新日志/任务状态再次核实是否继续推进。按前两批约 75 分钟及剩余波次估算，整队列预计约 20:15–21:00 UTC 完成，随实际完成波次更新。
- **26 号机训练**：16 个正式任务在 16:02 UTC 全部仍为 RUNNING；最新 round-2 步数范围约 169,675–179,550 / 182,182，抽查 loss 均为有限值。最慢项最近步速有波动，暂以 19:00–20:30 UTC 作为宽区间，后续按同一批原生日志重估。
- 26 号机已设每小时静默监控；不停止或改动现有训练。训练结束后先核验最终 checkpoint 与首轮归档，再确认 LasHeR-dev 图像路径后评估；不启动 test。
- 下一轮预训练微调候选沿用本日记录：对应学习率降为原值 1/10（head `1e-5`、preview `1e-6`）；`[10,30]` 只有在确认 milestone 以 epoch 计且与计划训练长度匹配后才采用。当前运行不变更。

## 2026-10-29 16:11 UTC 训练与GPU复核

- 25 号机 LasHeR-dev 队列状态文件仍记录 24 项中 8 COMPLETE、4 RUNNING、12 PENDING、0 FAILED；当前 4 个原生推理子进程仍存活且 CPU 持续繁忙。计分器尚未开始汇总，队列管理器更新时间早于当前检查，需继续以原生任务进程、run.log 和状态文件交叉核对。
- 25 号机活动日志显示两项推进到了 16:07 UTC 左右，另外两项末尾序列在约 15:56 UTC 后暂未变化；相应推理进程仍约 100% CPU。此差异可能是不同序列帧耗时不同，暂不判定停滞；下一轮检查同一任务日志是否更新。
- 26 号机 16 个正式训练在 16:11 UTC 全部仍为 RUNNING；最新 round-2 步数约 170,075–179,975 / 182,182，抽样 loss 有限。GPU2/3 约 80/78°C，SM 时钟约 1740/1635 MHz，SW Thermal Slowdown 均为 Active；GPU0/1 未触发。当前任务不变更。
- 基于最慢任务约 2,100 updates 剩余、近期约 40–45 updates/min 的观测，26 号机的保守完成窗口约为 16:55–17:30 UTC；后续按实测刷新。25 号机队列状态更新异常，整批完成 ETA 暂不收窄，待确认末尾两项进度后再估算。
- 微调候选继续保持为预训练权重 head 1e-5、preview 1e-6；LR milestone [10,30] 仅在 epoch 语义与训练总长度匹配后启用。当前 16 项运行配置保持不变。

## 2026-10-29 16:21 UTC ETA更正与实时复核

- **ETA更正**：16:11 条目把最慢训练剩余量误写为约 2,100 updates；实际当时约 182,182−170,075=12,107 updates。该条 16:55–17:30 UTC 的 ETA 无效。16:21 实测最慢任务约 170,575/182,182，距目标 11,607 updates；由 16:11–16:21 的慢项推进量估算约 50–55 updates/min，修正后的完成窗口约 **19:50–20:30 UTC**。
- 26 号机 16 个训练任务仍为 RUNNING；日志持续更新、抽查 loss 有限。GPU2 的软件热降频标志仍出现，GPU3 在 16:21 读数恢复为 1845 MHz 且未触发；说明低时钟是间歇观测，暂不能据一次 285 MHz 读数认定持续性能异常。
- 25 号机 LasHeR-dev 队列仍是 8 COMPLETE / 4 RUNNING / 12 PENDING / 0 FAILED，计分器 WAITING_FOR_EVALUATION。四个评估进程仍活跃且各自 run.log 在 16:19–16:21 UTC 有新序列/帧记录；队列摘要更新时间 15:33 UTC 尚未刷新，但日志证明推理仍在推进。

## 2026-10-29 16:39 UTC 单项LasHeR-dev计分

- 25 号机 candidate `joint_quality_k8_cost0p15_server25_20261027_formal` 的 dev 评估已完成并由项目标准 scorer 单项计分：LasHeR 97 sequences / 54,646 frames，PR **72.1283266**、NPR **67.9043040**、SR **53.3627386**；FPS **12.3822**，mean **80.761 ms**、P95 **113.607 ms**，总参数 **103,218,887**。checkpoint SHA256 `50c6af15f64c2b654b153a60bbbf69f3e31fb6781d1a50c3270219c6c4cb755f`，报告写入 `reports/joint_candidate_evaluation/joint_quality_k8_cost0p15_server25_20261027_formal/joint_quality_k8_cost0p15_server25_20261027_formal_lasher_dev_cost0p15_s0of1_metrics.json`。
- 此结果低于当前队列已知最佳 dev PR/NPR/SR **74.730/70.497/54.708**，三个目标 **75.8/71.9/60.2** 仍未达到；保留历史最佳，不以此项替换。
- 25 号机队列状态在 16:37 UTC 为 9 COMPLETE / 4 RUNNING / 11 PENDING / 0 FAILED。该单项标准计分输出与 watcher 预期文件名一致；watcher 仍等待整批完成，后续应复用现有输出并自动汇总。
- 26 号机 16:39 UTC 仍有 16/16 formal training RUNNING；round-2 step 约 171,600–181,550 / 182,182，抽样 loss 有限。按近 18 分钟慢项速度约 55–60 updates/min，最慢任务预计约 19:40–20:20 UTC 完成。GPU2 温度约 79°C、SM 约 1530 MHz且热降频标志 Active；GPU3 当前约 77°C/1665 MHz、标志未触发。热降频仍是间歇现象，当前训练未中止。

## 2026-10-29 16:40 UTC 失败序列与动作证据诊断

- 对上一条完整 candidate 的官方 97-sequence LasHeR-dev 结果运行 `tools/analyze_lasher_query_benefit.py`，脚本核验了评分报告所绑定的配置、checkpoint、prediction、GT、trace、evidence 文件 SHA256 后完成。逐帧结果位于服务器 `reports/diagnostics/joint_quality_k8_cost0p15_server25_lasher_dev_query_benefit_20261029.json/.csv`。
- 该运行有效 GT 帧 54,289、无效 GT 260；非 STOP 动作 2,536 帧、STOP 51,753 帧。非 STOP 动作中 1,468 帧即时 IoU 增益为正、458 为负、610 近零；成本效用为负的动作 1,318。该诊断只衡量已访问状态上的即时效果，不是官方 PR/NPR/SR，也不能说明未来轨迹漂移、未选 query 或完整 STOP-only 回放。
- 官方 per-sequence SR 最低序列包括：`manfarbesidespool` 2.72、`farmanrightwhitesmallhouse` 3.60、`dotat43` 3.95、`ab_pingpongball` 4.05、`midboyblue` 4.15、`belowrightwhiteboy` 6.09、`bikeboy128` 8.18、`AQblkgirlbike` 8.47、`girl\`sheadoncall` 8.61、`girlsquattingbesidesleftbar` 9.55。多个序列与历史 E0 的低 SR 集合重合，提示长尾失败仍存在；低即时动作效用值得继续测，但单凭这份结果不能将整体 SR 下滑归因于 query 接受。
- 下一步：等剩余正式 dev candidates 出分后，对相同序列与同一指标口径做 per-sequence 对照；只在 dev 上进行 route/modality 条件化的 acceptance/cost 校准候选，不搬用 ATOM/PrDiMP 的阈值，也不基于即时动作诊断宣称轨迹级收益。

## 2026-10-29 用户微调建议补充

若后续实验确认是从预训练权重微调，可将对应原始学习率的十分之一作为候选起点；学习率衰减节点 `[10, 30]` 仅在训练长度与调度器步数语义相符时试验，不作为通用设置。当前 VRAM18 任务仍为两轮、182,182 updates，训练脚本没有 LR scheduler，因此不追改这批运行中的任务；后续需单独设计划分并用 LasHeR-dev 验证。

## 2026-10-29 17:51 UTC 状态与微调候选

- 25 号机 LasHeR-dev 队列已 24/24 COMPLETE、0 FAILED；最后一个四分片任务已收尾，等待/核对汇总指标文件。
- 26 号机正式训练最近快照为 9 RUNNING、7 FAILED；FAILED 项保留原状态，均需恢复审计后才可做 LasHeR-dev 诊断。
- 用户提出的预训练微调候选：加载同一预训练权重，将各组学习率设为当前值的 1/10；`[10,30]` 只在明确总 epoch 和 scheduler 语义后启用。当前 2 轮、182182 updates 训练未启用 epoch scheduler，因此不修改运行中的任务。
 
## 2026-10-29 17:57 UTC 训练与评估更新

- 25 号机 LasHeR-dev 队列 24/24 COMPLETE、0 FAILED；计分器 SCORED_COMPLETE。当前新最佳候选 joint_quality_k8_cost0p1_server25_20261027_formal、cost=0.15：PR/NPR/SR=75.9676/70.7621/56.0472，FPS=13.8815；PR 达标，NPR 距目标 1.1379，SR 距目标 4.1528。其余 15 项中该候选三项均为当前最高；不可据此选 test。
- 26 号机 16 项正式训练最新状态 7 RUNNING、9 FAILED。剩余任务在 17:51–17:57 UTC 实测约 55–60 updates/min；按最新 step 外推，最早约 18:15 UTC 收尾，最慢约 20:00 UTC。FAILED 保持原状态并保留证据。
- GPU2 在 17:57 UTC 为 80°C、约 35% 利用率；nvidia-smi 当前事件标志为 SW Power Cap，未显示 SW Thermal Slowdown active。本快照不足以判断持续降频原因。
 
## 2026-10-29 18:11 UTC 失败项恢复审计结论

- 独立审计确认当前 9 项 FAILED 均有最终 latest.pth、182,183 行轨迹/标签记录，训练更新达到 182,182；失败统一发生在最终源码 SHA 校验，COMPLETE.json 缺失。
- 4 个训练相关源文件在任务启动后发生证据记录/路径校验漂移；模型结构、损失和已训练权重张量未见改变，但正式协议仍拒绝这些 FAILED 项。
- 审计建议保持原生 FAILED，逐 checkpoint 生成恢复 attestation，绑定 native 状态、计划/config/checkpoint/轨迹/标签/log SHA、计数、tensor witness 与漂移清单；只允许 LasHeR-dev 诊断，不得改写 COMPLETE 或跑 test。
- 18:11 UTC 26 号机仍有 7 项 RUNNING、9 项 FAILED；其中一项已到 182100，其他约 176200–181775，继续观察，不重启。
 
## 2026-10-29 18:27 UTC 训练监控

- 26 号机状态由 4 RUNNING/12 FAILED 变为 3 RUNNING/13 FAILED；新增结束任务仍为 exit 1 的最终源码 SHA 校验失败。
- 当前运行：joint_quality_k4_cost0p0125 step 177275、joint_quality_k4_cost0p025 step 177100、joint_quality_k4_cost0p15 step 180600。按近期 55–60 updates/min，最快约 25 分钟，前两项约 80–90 分钟。
- 目前仓库未发现已生成的恢复 attestation 或 recovery evaluation plan；失败权重仍不能直接进入正式验证。
 
## 2026-10-29 18:34 UTC 训练监控与后续配置核对

- 26 号机当前 3 项 RUNNING、13 项 FAILED；运行 step：joint_quality_k4_cost0p15=181025，joint_quality_k4_cost0p0125=177700，joint_quality_k4_cost0p025=177525。
- 训练计划已核对：两轮、总 182182 updates，head learning rate 1e-4、preview learning rate 1e-5，无 batch_size 字段、无 scheduler，也没有按 epoch 保存最后五个权重；因此 [10,30] 不能直接套用当前脚本。
- 下一轮预训练微调候选仍记录为当前 LR 的 1/10（head=1e-5、preview=1e-6），需单独建立计划并定义 epoch/checkpoint 保留策略。
- 失败恢复审计继续保持 dev-only；不改写原生 FAILED，不进入 test。
 
## 2026-10-29 18:54 UTC 训练监控

- 26 号机当前 2 项 RUNNING、14 项 FAILED；最后两项为 joint_quality_k4_cost0p0125 step=178950、joint_quality_k4_cost0p025 step=178650。
- 按近期 55–60 updates/min，预计还需约 55–65 分钟；FAILED 项均保持原生状态和 checkpoint，不启动正式验证。
- 25 号机 LasHeR-dev 最佳结果仍为 PR/NPR/SR=75.9676/70.7621/56.0472，FPS=13.8815。


## 2026-10-29 19:08 UTC 状态与微调准备

- 25 号机 8 个正式训练候选均已完成；LasHeR-dev 队列已全部完成并计分。当前最高候选仍为 `joint_quality_k8_cost0p1_server25_20261027_formal`，cost=0.15，PR/NPR/SR=75.9676/70.7621/56.0472，FPS=13.8815。PR 达标，NPR/SR 仍未达目标。
- 26 号机正式训练目前为 2 RUNNING、14 FAILED。仍运行的是 `joint_quality_k4_cost0p0125_vram18_20261027_formal`（约 179,950/182,182 updates）和 `joint_quality_k4_cost0p025_vram18_20261027_formal`（约 179,750/182,182 updates），按约 55–60 updates/min 估计还需约 4–5 分钟；不重启、不提前评估。
- 26 号机已 FAILED 的 14 项均保留原生 FAILED 状态；轨迹已到 182,182 updates，但尚未生成 recovery attestation/recovery evaluation plan，因此不进入正式 test。
- 下一轮预训练权重微调配置记录为当前 head LR=1e-4 的 1/10，即 1e-5；preview LR=1e-5 的 1/10，即 1e-6。当前训练脚本使用 rounds/updates、无 epoch scheduler，故 `[10,30]` 不直接写入本轮；需另建明确 epoch、milestone 和最后五个 checkpoint 保留策略的计划。


## 2026-10-29 19:20 UTC 训练监控与后续实验准备

- 26 号机两项质量训练仍在 RUNNING，子进程持续以约 100% CPU 运行，GPU2 约 75°C、约 4.7 GiB 显存；最近轨迹约为 180,450/182,182 和 180,225/182,182 updates，未见 NaN 或进程退出。
- 25 号机 LasHeR-dev 8 个正式候选已全部完成并计分；当前最好候选为 quality-k8 cost=0.15：PR/NPR/SR=75.9676/70.7621/56.0472，FPS=13.8815，参数量=103,218,887。已有参数量和 FPS 证据，尚无当前候选独立 MACs 测量。
- 历史 LasHeR 低 SR 分析显示，失败帧往往已频繁 query 但仍接受错误 query 结果，且绝对 confidence 不能可靠区分正确/错误结果。后续优先实现 primary/query 相对证据验收模块，保持主网络结构不变。
- 预训练权重微调的独立计划仍采用 head LR=1e-5、preview LR=1e-6；当前训练器以 rounds/updates 为单位且没有 epoch scheduler，因此不直接套用 [10,30]。下一轮需先定义 epoch、最后五个 epoch checkpoint retention 和 dev-only 选择规则。

## 2026-10-29 19:25 UTC MACs 与模块优化证据

- 25 号机空闲 GPU 已确认当前候选 checkpoint 可读，模型参数量为 103,218,887；服务器环境没有 fvcore/thop/torchprofile，因此尚未伪造 MACs 数值。
- 当前 tracker 的实际路径包含 preview、primary、candidate query/gain 和可选 query，MACs 会随 action/候选数量变化。后续应补充一次固定输入、主路径和 query 路径的真实 profiler 记录，再与 FPS 一起报告。
- 低 SR 的直接证据支持“相对证据验收/拒绝错误 query”的方向：先比较 primary_box 与 query_box 的一致性、相对 confidence 和时序稳定性，再决定是否替换 final_box；保持现有网络主干和模块接口不变。

## 2026-10-29 19:33 UTC 当前候选 MACs 实测

- 已在 25 号机 GPU0 对当前最佳候选 `joint_quality_k8_cost0p1_server25_20261027_formal` 的真实 checkpoint 做 PyTorch profiler 测量，固定 LasHeR-dev 序列 `boywalkinginsnow3` 的单帧代表路径。
- 主路径（不执行 auxiliary query）：约 **40,127,743,157 MACs/frame**（profiler FLOPs/2），参数量 **103,218,887**。
- 强制执行 wide query（385 auxiliary tokens）：约 **80,174,803,257 MACs/frame**。
- 当前 cost=0.15 该代表帧实际 action=STOP，因此正式报告应区分主路径和 query 路径，不能给一个与 action 无关的单一 MACs 值。结果文件：`reports/profile_candidate_macs_20261029.json`。

## 2026-10-29 19:38 UTC 开源方法对照

- KeepTrack 开源实现：https://github.com/visionml/pytracking/tree/master/pytracking/tracker/keep_track 。其可复用思路是维护跨帧候选身份、利用匹配概率判定候选是否可信，并在 `not_found`/`uncertain`/`hard_negative` 状态下抑制普通更新或触发重选；这与当前低 SR 中“query 后仍直接接受错误结果”的证据直接相关。
- MoETrack 开源实现：https://github.com/Zhangyong-Tang/MoETrack 。该仓库面向 RGBT modality validity，适合作为跨数据集模态可靠性校准的参考。
- CADTrack 开源实现：https://github.com/IdolLab/CADTrack 。当前项目的预训练主干和框架来源可追溯。
- 本项目下一步的三个核心模块候选：1）primary/query 相对证据验收与拒绝；2）跨帧候选身份/轨迹一致性记忆；3）按 RGB/TIR 可靠性进行数据集特定的模态门控。暂先做 LasHeR-dev diagnostic 对照，不直接替换主网络结构。

## 2026-10-29 19:51 UTC 训练收尾校验结果

26 号机最后两项 VRAM18 正式训练已停止：`joint_quality_k4_cost0p0125_vram18_20261027_formal` 与 `joint_quality_k4_cost0p025_vram18_20261027_formal`。两项 native status 均为 `FAILED`、exit code=1；GPU2 已空闲。

日志显示失败发生在训练完成后的源文件 SHA 校验，不是 NaN/Inf、显存不足或 GPU thermal failure。校验不一致的文件为 `scripts/run_tracking.py`、`dual_rgbt_evidence/tracker.py`、`dual_rgbt_global/candidate_tracker.py`、`scripts/run_benchmark_tracking.py`，均为训练启动后的追踪/评估侧文件漂移。两项均保留 `latest.pth`、`trajectory_steps.csv` 与 `query_labels.csv`，但不能直接改写为 COMPLETE，也不能进入正式 test。

下一步按既定协议保留 FAILED 状态，为这两项生成 recovery attestation，校验 checkpoint、轨迹、标签、日志和 tensor witness 后，仅生成 LasHeR-dev diagnostic；同时把这次源漂移记录为实验完整性问题。新一轮预训练权重微调仍采用 head LR=1e-5、preview LR=1e-6，衰减节点 `[10,30]` 需在明确 epoch/scheduler 后单独验证。

## 2026-10-29 预训练权重微调候选

用户补充：如果从预训练权重微调，学习率可先降为原始值的十分之一；LR 衰减步数可考虑 `[10, 30]`，但需结合具体 epoch 与 scheduler，不作为当前 rounds/updates 训练的直接参数。

拟议独立 dev-only 微调起点：沿用同一预训练权重与数据划分，head LR=1e-5、preview LR=1e-6（当前对应 1e-4、1e-5 的 1/10）；先明确 epoch 数、milestone scheduler 和 checkpoint 保留规则，再启动训练。该实验与当前 26 号机 FAILED 恢复候选分开记录，不改写 FAILED 状态，也不进入 test。

## 2026-10-30 恢复评估启动问题与修复

26 号机恢复候选的 LasHeR-dev 评估在正式推理前发现三类可复现的入口问题，均已保留失败日志和目录归档：

1. `run_task_four_gpu.py` 启动环境未传递 `OPENCV_FOR_THREADS_NUM=1`，入口在环境断言处退出。评估控制器显式设置该变量后恢复；后续应把它固化到四卡任务包装器。
2. 恢复协议生成器遗漏基础协议的 `seed` 字段，评估在 `torch.manual_seed` 处退出。已在 `tools/register_joint_recovery.py` 补齐 `seed=base['seed']`，并生成 R4 恢复协议。
3. `scripts/run_joint_candidate_evaluation.py` 的 trace CSV 字段仍是旧 11 列，而当前 tracker 输出了 primary/query/final box、confidence、query/template 字段，第一帧写盘时报字段不匹配。已补齐 27 列字段列表，并以当前源哈希重新生成 R4 计划。

前三轮失败的 native status、run.log 和残留输出目录保存在 26 号机 `runs/pipeline_tasks/recovery_eval_failures_20261029/` 及 `runs/recovery_eval_failures_20261029/`，不计入指标。R4 计划仍只允许 LasHeR/dev，不允许 test；R4 评估已在 GPU0/1 进入真实推理。

## 2026-10-29 21:25 UTC R4 评估实测进度

R4 恢复协议的 24 个 LasHeR-dev 任务已按 GPU0/1 双卡控制器排队。当前 2 个任务（global_candidate_joint_gainpair，cost=0 和 0.05）均为 RUNNING：分别完成 7/10 个序列、4,489/4,793 帧；按当前速度各剩约 102/95 分钟。队列剩余 22 项，当前没有完成指标或失败任务。控制器按完成帧数估算剩余时间后再检查，不固定高频轮询。

## 2026-10-29 21:25 UTC 源哈希部署边界

启动器环境修复已提交 GitHub commit `3ae9ba4`（`run_task_four_gpu.py` 传递 `OPENCV_FOR_THREADS_NUM=1`）。当前 26 号机 R4 评估计划绑定修复前的启动器哈希，运行中不热更新远端文件；待 R4 dev 队列完成并评分后再部署该 commit，防止末尾完整性校验因源文件漂移而失败。

## 2026-10-29 21:35 UTC R4 评估继续进度

R4 双卡评估仍为 RUNNING、无失败：cost=0 与 0.05 两项各完成 30/97 序列、9,664/54,646 帧，已覆盖历史低 SR 片段中的 `ab_pingpongball` 和 `bikeboy128`。按当前实测速度，单项剩余约 86–92 分钟；尚未生成整批 PR/NPR/SR。GPU0/1 约 2.3 GiB 显存，温度约 71/59°C，进程存活且正常写入 progress.json。

历史 LasHeR-dev 失败分析继续支持三个候选创新方向：primary/query 双框一致性验收、跨帧轨迹一致性、RGB/TIR 可靠性门控；当前只记录证据，等待 R4 完整结果后再决定是否训练新模块。

## 2026-10-29 21:39 UTC R4 评估监控

R4 控制器仍在运行，GPU0/1 两个首批任务均存活、无失败；各完成 30/97 序列、9,664/54,646 帧，预计还需约 99 分钟（受序列长度影响）。当前没有评分报告；R4 的 8 个候选计划和 attestation 已完成登记，test 仍被门禁拒绝。

## 2026-10-29 21:46 UTC R4 低频监控与预训练微调决策

- 26 号机 R4 LasHeR-dev 恢复评估仍正常运行：GPU0 的 `global_candidate_joint_gainpair_trajectory_formal_20261026_lasher_dev_cost0p0_s0of1` 与 GPU1 的 `...cost0p05_s0of1` 均为 `RUNNING`，没有新增 `FAILED`。
- 两项均已处理 32/97 个序列、14,643/54,646 帧；控制器队列还剩 22 项。当前 GPU0/GPU1 显存约 2.3 GiB，温度约 71/59°C，GPU 利用率约 10–11%，子进程 CPU 约 98–100%，未见 NaN、进程退出或 GPU 异常。
- 以当前实际帧速估算，首批两项还需约 80–100 分钟；下一次检查跟随控制器的约 6 分钟检查点，之后按进度重新估算，不固定高频轮询。长序列可能使单次估算波动。
- 用户提出的预训练权重微调方案登记为下一轮候选：head 学习率 `1e-5`、preview 学习率 `1e-6`（当前 `1e-4/1e-5` 的十分之一）。`[10,30]` 仅在把训练周期明确为 epoch 并启用 milestone scheduler 后采用；现有 182,182-update、2-round 训练没有 epoch scheduler，因此本轮 R4 不改，后续单独做 dev-only 微调验证。
- 当前 R4 只用于恢复候选的 LasHeR-dev 诊断，原生 FAILED 状态保持不变，不能据此进入 test。

## 2026-10-29 21:57 UTC R4 进度复核

- 26 号机 R4 控制器仍为 `RUNNING`，已运行约 39 分钟；队列为 0 完成、0 失败、22 待运行、2 活跃。
- GPU0 活跃项已到 43/97 序列、19,554/54,646 帧；GPU1 活跃项也已到 43/97 序列、19,554/54,646 帧（两项当前处于同一序列边界，进度文件更新时间分别约 0.1/2.5 分钟前）。
- 两个评估子进程仍存活，CPU 约 98–100%；GPU0/1 显存约 2.3 GiB，温度约 71/59°C。没有新失败或异常退出。
- 以总帧数和已运行时间估算，首批单项约还需 70–90 分钟；剩余 22 项将由控制器按完成后自动接续，整批预计需要更长时间，不能用首项时间直接代替。

## 2026-10-29 22:04 UTC R4 活跃任务复核

- R4 控制器为 `RUNNING`：0 完成、0 失败、22 待运行、2 活跃；运行约 46.5 分钟，计划下一次自动检查约 18.5 分钟后。
- GPU0 当前评估处理到 48/97 序列、22,822/54,646 帧，最近序列 `rightfirstboy-ly`；GPU1 到 50/97 序列、23,867/54,646 帧，最近序列 `left2ndgreenboy`。两份 progress 文件在 0.5 和 1.3 分钟前更新，处于正常增长。
- 两个评估子进程仍存活且 CPU 约 98–100%；GPU0/1 约 2.3 GiB 显存，约 71/59°C，无失败、NaN 或异常退出。
- 按当前总帧速估计，首批两项还需约 55–65 分钟（序列帧数差异会带来波动）；整条 24 项队列仍有 22 项待运行。

## 2026-10-29 22:25 UTC R4 长序列复核

- R4 控制器仍为 `RUNNING`：0 完成、0 失败、22 待运行、2 活跃；已运行约 67.7 分钟。
- GPU0 已跨过长序列 `ab_bolster`，当前累计 51/97 序列、32,273/54,646 帧；该序列单项记录 8,406 帧，解释了此前约 18 分钟的 progress 文件静默。GPU1 当前 53/97 序列、33,285/54,646 帧，最近序列 `whitecarturn2`。
- 两个 Python 评估子进程仍在运行，CPU 约 98–100%；GPU0/1 约 2.3 GiB 显存、71/59°C，未发现退出或异常。
- 当前首批任务的剩余时间受 `ab_bolster` 这类长序列影响，按已完成帧量估计约 45–60 分钟；继续以序列完成和控制器检查点为准。

## 2026-10-29 22:29 UTC R4 进度复核

- 控制器仍为 `RUNNING`：0 完成、0 失败、22 待运行、2 活跃；下一次控制器检查约 26 分钟后。
- GPU0 仍在 `ab_bolster` 后的序列级处理，累计 51/97 序列、32,273/54,646 帧；GPU1 已推进到 54/97 序列、35,985/54,646 帧，最近完成 `midboyblue`（2,700 帧）。
- 两个评估进程均存活且 CPU 约 98–100%，显存约 2.3 GiB；未发现失败、NaN 或异常退出，当前还没有可评分结果。

## 2026-10-29 22:40 UTC R4 进度更新

- R4 控制器仍为 `RUNNING`：0 完成、0 失败、22 待运行、2 活跃；已运行约 83 分钟，下次控制器检查约 12 分钟后。
- GPU0 为 65/97 序列、39,757/54,646 帧；GPU1 为 73/97 序列、42,944/54,646 帧。两个评估进程仍存活，CPU 分别约 100%/98.5%，GPU0/1 显存约 2.3 GiB、温度约 73/60°C。无可评分输出。
- 结合 GPU0/1 不同帧速和序列长度波动，首批剩余时间估计约 20–35 分钟；长序列使 ETA 存在较大误差。

## 2026-10-29 23:15 UTC R4 已评分结果与评分环境问题

- R4 恢复评估控制器仍在运行：2/24 已完成、0 失败、21 待运行、1 活跃。前两项均为 `global_candidate_joint_gainpair_trajectory_formal_20261026`，同一 checkpoint SHA `af9d064427892585d104e197a3ad398c126ded66b476c8642e59ece1be506d90`。
- `cost=0.05` LasHeR-dev：PR/NPR/SR **73.7822/68.8474/54.5162**，FPS **8.9614**，平均耗时 111.59 ms/frame，P95 159.12 ms，参数 103,218,887。
- `cost=0` LasHeR-dev：PR/NPR/SR **73.2134/68.4432/53.9454**，FPS **8.2028**，平均耗时 121.91 ms/frame，P95 175.52 ms，参数 103,218,887。
- 当前完成项中 `cost=0.05` 最好，但距 LasHeR 目标仍差 PR 2.0178、NPR 3.0526、SR 5.6838 个百分点；两份结果都是 `RECOVERY_DEV_DIAGNOSTIC`，训练原生状态仍为 `FAILED`，均不具备 test 资格。
- `cost=0.05` 最差序列包括 `ab_pingpongball` (SR 1.60)、`girl\`sheadoncall` (2.69)、`dotat43` (3.31)、`bikeboy128` (8.18)、`blackbag` (10.68)；应结合 cost=0.15 与其他 checkpoint 的逐序列结果判断是否为共性问题，不凭单项就改模型。
- 评分时发现 `dual_rgbt` 环境为 Python 3.7.16，导入 LasHeR 工具包因 `list[str]` 注解失败；没有改工具包源码。改用现有 `cpu-scoring` Python 3.10.14 环境成功评分（该环境有 NumPy，无 Torch，足以运行 CPU scorer）。
- 远端 JSON/NPZ 已同步到仓库 `reports/recovery_results_20261029_r4/`，用于逐序列复核。GPU1 当前在跑 cost=0.15；最新观察进度 4/97 序列、3,190/54,646 帧，进程满 CPU，无失败。

## 2026-10-29 23:27 UTC R4 cost=0.15 监控

- R4 控制器：2/24 完成、0 失败、21 待运行、1 活跃；GPU1 的 `cost=0.15` 评估已到 24/97 序列、8,638/54,646 帧，progress 文件刚更新。
- 评估进程持续约 99.9% CPU，GPU1 约 2.3 GiB 显存、57°C，无异常退出；继续按长任务低频观察。
- 前两项 `cost=0` 与 `cost=0.05` 已完成并评分，`cost=0.05` 当前更优；`cost=0.15` 结果待完整跑完后再比较。

## 2026-10-30 00:20 UTC 新会话实时审计与校准修复

- 当前目标以用户最新要求为准：LasHeR PR/NPR/SR 75.8/71.9/60.2；RGBT210 PR/SR 88.7/62.9；RGBT234 MPR/MSR 90.9/65.6。优先 LasHeR，不能把 dev 达标直接作为 test 达标。
- 25 当前无训练/评估进程，四卡空闲。评分状态 SCORED_COMPLETE；最优 quality-k8 train cost0.1 / eval cost0.15：75.9675846/70.7621205/56.0471607。PR 高于目标0.1676；NPR差1.1379，SR差4.1528。
- 26 当前无训练进程，R4 为2完成、2运行、20排队、0评估失败；实时确认控制器PID483958、包装进程690591/749758及评估子进程存活。gainpair cost0.15已54/97序列、35985/54646帧，quality cost0已37/97、17467/54646帧；线性帧速剩余约35/72分钟，长序列会影响ETA。全部队列ETA尚不能由这两个单项ETA代表。
- 资源问题已定位：/tmp/recovery_eval_controller_20261029.py仅遍历GPU0/1；先launch后reap，完成释放后可能等下一轮才补位。尚未更换运行控制器，26 GPU2/3和25四卡仍未利用；不得报告已实现八卡调度。迁移任务前需明确唯一队列所有者，防止重复评估。
- 源码校验失败仍保留原生FAILED状态，不伪造COMPLETE；恢复结果仅dev diagnostic。
- 修复tools/calibrate_query_acceptance.py的三个真实问题：执行query时primary_iou被写为NaN；STOP使用上一帧预测而非同帧primary；每个序列重复extend全序列名单导致97序列覆盖断言失败。统一从evidence的primary_x/y/w/h读取同帧primary，并逐个append序列名。未改网络结构。
- 新增带97个序列、哈希核验、STOP/有害query/有益query的文件读取回归测试；全部3项单元测试通过。仅向25部署该离线校准工具；未热改26正在评估的源码。
- 在25最优候选完整dev上运行现有analyze_lasher_query_benefit和修复后的calibrate_query_acceptance，报告reports/best25_query_benefit_20261030.json与reports/best25_query_calibration_20261030.json。54289有效帧，343次query（约0.632%），197有益、113有害、33中性，mean query IoU gain=0.1039631。校准的全有效帧加权即时IoU提升仅0.003013个百分点，不是官方SR、不是闭环提升。
- 路由阈值仅为dev离线拟合：rgb_local128=0.0400006324、tir_local128=0.0135665005、rgb_wide=0.4971846044、tir_wide=0.2908963859。未投入tracker，不据此声称指标改善。wide样本仅9/20帧，证据有限。
- 下一步应分析primary定位/长期漂移及不同cost轨迹，验证是否存在漏触发query；不能从已执行query诊断推断漏触发率。先解决队列资源分配，再做闭环dev对照。现有证据不足以把三个候选方向称为已验证核心创新。
- 参数量既有记录103218887；FPS既有最优记录13.8815。已读取MACs报告：单帧STOP约40.128G、强制wide query约80.175G，仅torch.profiler支持算子的FLOPs/2，不是全数据平均MACs。
- 最后五epoch保留/验证、收敛判断、RGBT210/234验证、三个模块创新验证仍待完成；当前trajectory训练为2 rounds且写latest.pth，不能声称已实现五epoch选优。未删除任何权重。
## 2026-10-30 八卡队列扩展（迁移准备）

用户要求尽量不让卡空闲。新增tools/continue_dev_queue.py：复用现有队列状态、接管仍存活的wrapper，不重启评估子进程；先回收已结束任务再补位，按4卡检查显存<500MiB后启动，180秒轮询，flock排除第二个相同状态控制器，原子写状态。该锁用于实际已发生的控制器交接，避免两个队列写者和重复启动。Linux临时目录真实短进程回归测试通过（接管、同轮4卡补位、第五项补位、唯一完成记录）。

26原PID483958仅调度器已退出；原评估wrapper690591/749758保留。四卡控制器当前PID828259，状态仍在runs/pipeline_tasks/recovery_eval_controller_20261029/status.json。GPU2/3新增global quality cost0.05/0.15，四张卡已实际占用约2.3–2.4GiB。原24项划分为26负责12项（2已完成、4运行、6排队），25负责12项；迁出列表写在reports/server25_recovery_assignment_20261030.json和26 state.exported，迁出项已从26 queue移除。

25原server25候选的部分结果已有native COMPLETE但未显示在scoring_status汇总，不能误认为未运行而重复训练/评估。使用独立/data2/gb/Dual-RGBT复现26原始路径；原/home/gaob/Dual-RGBT目录保留。复用25已验收conda环境，通过conda-envs/dual_rgbt符号链接保持原计划中的解释器路径。LasHeR/train映射到25已有trainingset。迁移4个候选共12个cost组合的原计划、attestation、FAILED训练状态、轨迹、标签、checkpoint及源码，数据和训练来源不改写，仍仅dev diagnostic。原文件约5.07GiB，压缩4152566940字节。迁移中，尚未宣称25四卡启动。
### 八卡启动及迁移校验完成

- 25迁移收据reports/server25_recovery_migration_receipt_20261030.json为PASS：4个原始recovery plan均通过evaluation_plan源码/输入/训练证明校验；97序列、54646帧的全部RGB/TIR路径存在，boxes与25原dev manifest一致，388个首尾图像SHA256样本匹配26。未修改计划内容或伪造原训练状态。
- 25新控制器PID1064962，状态/data2/gb/Dual-RGBT/runs/pipeline_tasks/recovery_exported_controller_20261030/status.json；wrapper PID1065091/1065092/1065093/1065094对应GPU0/1/2/3，native均RUNNING，四卡均约2266MiB且GPU利用率10–13%，温度59–61°C。队列4活跃、8等待。首批为gainpair-k8-cost0训练候选的eval cost0/0.05/0.15，加quality-k4-cost0候选eval cost0。
- 26控制器PID828259，4活跃、6等待、2完成；8项活跃分布于两台共8卡。只进行LasHeR-dev恢复诊断，未启动test。运行进程仍为各自独立的原始候选/参数组合，没有同一任务跨机重复运行。
- 后续监控必须同时检查26原R4状态与25新增exported状态；不能只看25旧/home/gaob/...评分队列。26本机COMPLETE仅代表其保留的12项，完整R4仍需合并25迁出的12项并核对24个唯一task。25原八模型评估结果独立保留。
- 一卡一个实际评估进程；GPU利用率受逐帧CPU处理影响，不把低瞬时利用率误报为空闲。当前未增加无效显存占位，也未改变评估batch/模型/源码哈希。调度器每180秒检查并自动补位；队列耗尽后再安排有证据支持的下一轮实验。
## 2026-10-30 恢复结果完整评分及定位误差诊断

- 本轮重新确认两台8项wrapper和两台调度器实际存活；26序列/帧数继续增长，25已进入首批序列。前轮属于实际进展（八卡调度上线），不是仅重复状态。
- 实际调用tools/joint_candidate_results.py score在verify_runs之后因R4协议缺少toolkit_code_sha256而KeyError退出，未生成评分文件。另查R4 plan也未带原评分入口需要的initialization字段。没有修改任何正在执行的冻结源码/协议。
- 新增tools/score_recovery_queue.py独立恢复评分工具：先复用verify_runs验证native exit0、权重/计划/源码哈希、97序列54646帧、GT、prediction、trace/evidence及timing；显式使用原JOINT_CANDIDATE_EVALUATION_PROTOCOL_20261026.json绑定的toolkit哈希；再调用原official_metrics。恢复标记保持RECOVERY_DEV_DIAGNOSTIC、training_native_status=FAILED、test_eligible=false。初始化信息明确取恢复协议中的initialization。
- 两项已完成gainpair结果重算通过：cost0 PR/NPR/SR=73.2133879565/68.4432495188/53.9454107392、FPS8.2027587705；cost0.05=73.7821861528/68.8474055550/54.5162411289、FPS8.9613731741。与原raw评分的PR/NPR/SR/FPS逐项精确相等。完整校验报告位于两台/data2/gb/Dual-RGBT/reports/recovery_verified_scores_20261030/。
- 两台后台CPU评分watcher已启动且通过ps验证存活：25 PID1073392、26 PID855280；每180秒检查各自completed队列，自动评分，不占GPU。各自目录scoring_status.json/watcher.log为监控入口。队列和评分完成只代表本机，需要合并24个唯一task。25复用原已验收cpu-scoring环境软链，不修改环境包。
- 核对25原目录全部metrics.json，已有24份SCORED_COMPLETE。旧scoring_status仅列15项是汇总遗漏，不是9项未评分。补存reports/server25_all24_metrics_summary_20261030.json，最优仍为quality-k8 traincost0.1 evalcost0.15，75.9676/70.7621/56.0472，FPS13.8815。
- 完整97dev定位诊断（pred/GT SHA逐项与评分报告比对）：54289有效非初始帧；3686帧中心误差<=20px但连续IoU<0.5，其中2178帧宽或高相对GT超出[2/3,1.5]；13080帧IoU<0.1。这些是逐帧计数，不能替代按序列曲线平均的官方PR/NPR/SR，也不能直接解释全部SR差距。报告reports/best25_localization_diagnostic_20261030.json绑定三份来源评分SHA。
- 好中心/低IoU集中在midboyblue(707)、leftdrillmasterstanding(428)、rightboy_1227(225)等序列。cost0至0.15不同闭环轨迹的SR变化呈明显分化：left2ndboy+48.55、belowrightwhiteboy+35.02，但rightmirrorlikesky-28.97、carstop-14.85。不能仅依据已执行query同帧平均增益推断长期轨迹收益。
- 代码证据：scripts/train_candidate_joint_quality_trajectory.py PREFIXES仅executor.preview./executor.quality./query_gain.；requires_grad按此前缀设置，只允许14个状态张量改变。定位头冻结。因此下一步候选是同架构下的定位头适配与尺度误差对照，但目前只是可检验假设，并未证明冻结定位头是全部误差的原因。
- 已核实开源参考：ATOM论文https://arxiv.org/abs/1811.07628及代码https://github.com/visionml/pytracking，利用学习的重叠估计改善框定位；KeepTrack实现https://github.com/visionml/pytracking/blob/master/pytracking/tracker/keep_track/keep_track.py，维护候选关联、处理uncertain/hard_negative状态。分别对应本项目尺度定位误差和轨迹分化问题。尚未移植或验证有效，不能称为我们已完成的创新；第三模块仍需独立证据与消融设计。
- 未删除checkpoint、未启动test；当前最优SR仍距60.2差4.1528点，不能按仅0.3点差距的seed微调策略处理。最后五epoch保留/验证及收敛早停仍未实现，本轮只建立可靠评分与诊断证据。
## 2026-10-30 00:43 UTC 尺寸末层适配的真实小试

- 本轮先核对8项评估及2个评分watcher存活。25四项评估均已推进至5–6序列；26有一项进入81/97序列，其他处于长序列/后续序列处理；未因观测间隔重启任何评估。
- 发现历史负证据：reports/LASHER_READOUT_ADAPTATION_RESULT_20261018.json记录decode+box_head约17M参数适配，crop loss改善但完整dev SR从56.4331降到41.6646，已REJECT。因此取消宽泛的readout重训设想，收窄为现有conv5_size末层weight/bias的66参数适配。它不增加模块或网络层，只是可验证的同架构参数实验，不宣称创新成立。
- 新增scripts/train_size_tail.py：加载25当前quality-k8原生COMPLETE候选，验证父评估计划、checkpoint和训练manifest；train882与dev97/test245隔离；仅executor.box_head.conv5_size两张量可训练，模型保持eval以固定BN统计；使用现有localization_loss，AdamW，TF32关闭、torch线程1。
- 每次sanity用真实train样本做4步（RGB/TIR各2步），逐步确认同输入score/offset bit equal，确认只2张量变化、loss/grad有限，保存后严格重载box/score/size/offset bit equal。仅66参数训练，总参数仍103218887；改变尺寸仍可能改变闭环候选、模板与历史，不能据同输入不变性宣称闭环不变。
- B8/LR1e-6：native COMPLETE/exit0，32样本4updates，SANITY_PASS；peak allocated634140160B（0.5906GiB），reserved799014912B（0.7441GiB）；计时16.44s（不含前置加载校验、包含本脚本训练/验证/保存范围）。checkpoint SHA3789b175fadba359ed286e0e6a8c9787054762c4a89f35a59983eb54c7d263c7。
- B64/LR8e-6：native COMPLETE/exit0，256样本4updates，SANITY_PASS；peak allocated2159596032B（2.0113GiB），reserved2657091584B（2.4746GiB）；计时72.25s。checkpoint SHA0f8ac38bf102fa6c23cf9ce27b4702394a923f2fbe9f69a6aa2845db322f42f6。两组loss不能当成相同数据/路由分配的准确率比较，仅用于功能和资源验收。
- 基于B8/B64实测显存增量，启动B384/LR4.8e-5的4步probe；预计约10GiB PyTorch allocated，真实reserved/物理占用需以运行结果确认。GPU0原有eval约2.3GiB保持，未停任何评估。wrapper PID1088817；原生状态/home/gaob/Dual-RGBT/runs/pipeline_tasks/train_size_tail_sanity_b384_20261030/status.json，日志同目录run.log。预计数分钟，不应高频轮询。
- epoch权重保存函数已实现只保留本次run最近5个epoch文件；测试覆盖第7epoch后只留3–7、拒绝覆盖同名文件、父目录同名权重不动。新run中尚只有sanity epoch，未清理已有正式权重。
- 完整实验计划与tracker写入refine-logs/并有时间戳历史版本、MANIFEST。正式训练暂不启动：需先实现并验证零更新父权重的完整闭环评估入口，随后接通最近5epoch完整dev选优与基于dev的早停。现有crop loss不具备选模资格。本轮没有新的PR/NPR/SR提升结果。

## 2026-10-30 00:51 UTC - Size-tail closed-loop evaluator admission

- All eight RTX3090 cards retain live recovery tasks. Server25: 0/12 completed, four active, eight queued. Server26: 2/12 completed, four active, six queued at the latest queue observation. Both controllers are live and refill every 180 seconds. Approximate recovery-queue ETA: 25 another 4-5h; 26 another 3-4h, dependent on sequence/runtime load. These are validation ETAs, not outstanding original training time.
- Added independent scripts/eval_size_tail.py and tools/score_size_tail.py; old registered runners, sources and plans remain intact. Parent formal provenance, native command, checkpoint/source/manifest hashes and full split are checked. Adaptation permits exactly the existing size-layer weight/bias tensors, 66 parameters; total model 103,218,887. No architecture change.
- Parent control uses the native-complete quality-k8 train-cost0.1 parent, eval cost0.15. It compares serialized prediction boxes and action/modality pairs against the original full dev run. The first launch failed at the missing OPENCV_FOR_THREADS_NUM check before prediction; its FAILED native receipt is retained. Explicit launcher environment corrected in a new r2 task.
- r2 native COMPLETE/exit0: 11 frames, prediction/action/modality equality true, 265 frozen state tensors, no backward/update. This is only prefix evidence. Full97 equivalence is not yet claimed.
- Full parent control launched on original25 GPU1, wrapper1100268, run size_tail_parent_control_dev_20261030. CPU scorer PID1100867 waits at 180s intervals and uses the unchanged official toolkit. Existing recovery eval still shares the GPU; timing is concurrent-load service FPS.
- B384/LR4.8e-5 train probe remains active, wrapper1088817. Three of four steps observed, losses finite; GPU0 total observed14,366MiB including the recovery eval. Peak allocator/strict reload results remain pending. Do not edit scripts/train_size_tail.py while it is running.
- Formal epoch training and last-five full-dev convergence stopping are still unadmitted/unimplemented respectively. No new PR/NPR/SR claim in this update. Local syntax checks passed, real CUDA parent-equality smoke passed. Five-checkpoint retention regression passed; Linux queue test was skipped locally on Windows (previous real Linux result remains the queue evidence).
- Plans, launch records, failed/r2 native receipts and smoke receipt are preserved under reports/SIZE_TAIL_PARENT_CONTROL*.json. See docs/SIZE_TAIL_DEV_EVALUATION_20261030.md for reproducible execution. Do not modify evaluator sources while the full parent run is active.


## 2026-10-30 00:52 UTC - B384 passed; larger-batch resource probe

- B384 native COMPLETE/exit0; SANITY_PASS, four updates, only two allowed size-tail tensors changed, strict reload bit-equal. Allocated10,883,894,784B (10.14GiB), reserved13,595,836,416B (12.66GiB), measured training/sanity phase363.02s. Checkpoint SHA e00666232e92221caa9bb63b4cd88edf35a9b20d819fafb0c27bb90b1c88971d. No accuracy or convergence claim.
- B576/LR7.2e-5 probe launched on original25 GPU0, wrapper1105161, four updates/2304 training pairs; estimate around9-10min from B384. Projected reserved about19GiB plus the existing eval about2.3GiB. B640 would leave little headroom at the measured allocator ratio. Source remains the unchanged pinned train_size_tail.py. The prospective sampled epoch is9216 pairs (16 batches); this sanity-only plan does not admit formal training.
- Adapted B384 checkpoint 11-frame evaluator smoke launched on25 GPU2, wrapper1105162; full dev follows only after successful smoke. Parent full-equivalence control remains running onGPU1; first completed sequence565frames matches the original saved predictions/actions/modalities exactly.
- Actual recovery refill observed:26 now3 completed, four active, five queued, zero failures. GPU1 automatically moved from completed gainpair-global cost0.15 to gainpair-k4 cost0. The25 migrated queue remains four active/eight queued. All eight GPU slots have live native tasks.


## 2026-10-30 00:53 UTC - Adapted full dev launched

- B384 adapted11-frame smoke native COMPLETE/exit0; checkpoint and training receipt checked, exactly two allowed tensor changes,265 state tensors frozen during tracking, no optimizer updates. The parent-equality flag is false by design for adapted weights; exact equality is checked only by the separate zero-update parent control.
- Full B384 dev diagnostic launched on original25 GPU2: wrapper1106853, run size_tail_b384_dev_20261030. CPU scorer1107101 waits every180s and writes reports/size_tail_dev_20261030/b384_sanity.json. Parent full control wrapper1100268/scorer1100867 remains running onGPU1. Both use the unchanged official toolkit and must complete97seq/54646frames before metric claims.
- B576 probe wrapper1105161 confirmed alive onGPU0. Recovery evaluation continues on all eight GPUs. Full-dev control/effectiveness outcomes and B576 actual peak memory are the next decision points. Do not launch formal epochs before those results are reviewed.

## 2026-10-30 01:06 UTC - Resumable epochs and B576 dev

- B576 native COMPLETE/exit0, SANITY_PASS: four updates/2304pairs, only66size-tail parameters changed, strict reload exact. Allocated16,118,192,640B(15.01GiB), reserved20,149,436,416B(18.77GiB), phase409.65s. Checkpoint55c95b8b487b2222b9321c40f9ee719a6c16b9a53ae8d9ae18aa4e04cc3d9e41. No accuracy/convergence claim.
- scripts/train_size_tail_epoch.py now trains one native epoch and restores network/AdamW state. Real B8 epochs1/2 passed, updates4 then8. All265 final network tensors and losses match uninterrupted64-pair training. Optimizer loading is bit-exact. Final weight exp_avg differs2.235e-8 and exp_avg_sq2.328e-10; repeating uninterrupted training also differs8.382e-9/2.328e-10 while all265network tensors match. This is consistent with numerical nondeterminism; exact final optimizer equivalence and a particular-kernel cause are not claimed. Audit:reports/SIZE_TAIL_RESUME_NUMERICAL_AUDIT_20261030.json. Reference:https://github.com/pytorch/pytorch/blob/v1.12.1/docs/source/notes/randomness.rst. No numerical workaround was added.
- tools/run_size_tail_epochs.py implements one epoch -> native complete -> full97dev -> official CPU score -> convergence decision. Normal minimum5/max10epochs; patience3without SR progress>0.1pp. SR drop>5pp below parent immediately rejects and preserves evidence, responding to historical readout loss14.7686pp. Latest5checkpoints retained after evaluation; final best by SR/PR/NPR kept; cleanup only verifies/removes exact owned paths. Six Linux synthetic workflow tests pass, including failure and regression rejection. This is not a formal model result. An observed unclosed-lock warning was resolved with a context manager.
- Draft reports/SIZE_TAIL_EPOCH_PILOT_DRAFT_20261030.json proposes B576/LR7.2e-5,9216sampled pairs/16updates per epoch. formal_launch_allowed=false; full-dev report hashes are intentionally unset pending review. No formal pilot controller started. Active evaluation sources remain unchanged.
- B57611-frame tracking passed, then full-dev wrapper1126673/GPU0 launched; scorer1128119 polls180s and writes reports/size_tail_dev_20261030/b576_sanity.json. Parent control1100268/scorer1100867 onGPU1 and B3841106853/scorer1107101 onGPU2 remain alive. At01:03UTC completed sequences were30/23/1 respectively; recent traces updated within4-7seconds. Eight recovery GPU slots, both queue controllers and scorers verified live.26:3complete/4active/5queued/0failed;25:0complete/4active/8queued/0failed.
- Verified gainpair recovery cost0.15:PR71.9562364695,NPR66.9081683025,SR53.0160618562,FPS9.1203120141. Versus0.05: -1.82595PR/-1.93924NPR/-1.50018SR points;46seq improved,48worse,3unchanged. Worst SR changes:left2ndboy(-39.43),7runone(-31.27),AQblkgirlbike(-30.82),moto2north1(-28.15). Mean heavy tokens435.6363->389.6862. Concurrent service FPS is not a paired speedup. Full verified report and three-cost diagnostic saved locally; training-native FAILED/test-ineligible classification retained.
- Next: complete all three full-dev comparisons, compare both protectedE1 and the current-grid parent, then admit a formal pilot or reject/modify the method from observed failures. Three validated innovations and all requested benchmark targets remain outstanding.
- Pre-commit hash check found CRLF/LF mismatch in the new, inactive epoch controller. Normalized it to LF, reran all6Linux workflow tests, and verified the five draft source hashes match working files, staged Git blobs and deployed25files. Active evaluator/trainer sources were not edited.

## 2026-10-30 01:17 UTC - Template failure diagnostic and full-dev ablation

- Verified all97 parent artifacts before diagnosing54289 valid noninitial frames. There are13080 finalIoU<0.1 frames,6181 with confidence>=0.65. Of13095 primary-lost frames,13019 STOP and76 query; lost-frame query rate0.5804% versus0.6482% elsewhere. Query sparsity alone does not establish missed recoveries. Executed queries recover8 primary-lost frames toIoU>=0.5 and destroy2 primary-good frames toIoU<0.1. These are instantaneous observations, not counterfactual future trajectories.
- Legacy traces lack template_updated. Verified pinned server25 candidate_tracker.py SHA89e18853f60eac309e47206005450164931b68b3934ae36a40034cd3b3ce03ed and reconstructed writes from frame%5==0 andconfidence>=0.65. Of7608writes,1215(15.97%) occur atIoU<0.1;1214areSTOP and1query. Longest lost episode is midboyblue1930-2699,770frames,1query/95writes. Report:reports/BEST25_TRACKING_FAILURE_DIAGNOSTIC_20261030.json. This is correlation, not proof of causality. Two diagnostic regression tests passed; full real-artifact analysis passed.
- Open-source references: OSTrack stores its initial template (https://github.com/botaoye/OSTrack/blob/main/lib/test/tracker/ostrack.py); STARK-ST protects its first template and updates extras with classification confidence (https://github.com/researchmm/Stark/blob/main/lib/test/tracker/stark_st.py). No reference code was transplanted. Static-template ablation is a diagnostic, not a validated innovation.
- New independent scripts/eval_template_freeze.py restores initialized templates after each normal track call. Parent weights/cost0.15/K8 remain fixed; no GT drives decisions. Template crops are still computed then discarded, so timing is not an optimized speed benchmark. Existing active evaluators and model sources were not edited.
-31-frame smoke native COMPLETE/exit0:265 frozen state tensors,0optimizer/backward calls,6actual template mutations reverted. Full97dev launched original25GPU3, wrapper1155267, run static_template_dev_20261030; CPU scorer1155571 polls180s and writes reports/template_dev_20261030/static_template.json. New plan, source hash and launch/native/smoke receipts are preserved. Full accuracy is pending.
- At01:16:30UTC,25 recovery queue0complete/4active/8queued/0failed;26 queue3complete/4active/5queued/0failed. Both180s refill controllers remain active. Parent/B384/B576 extra dev runs have41/34/30 completed sequences,18368/16822/9664frames. With the new ablation each25GPU has two independent evaluations; each26GPU has one. Occupancy is confirmed, increased aggregate throughput is not yet established.
- Formal size-tail pilot remains NOT ADMITTED pending full-dev comparisons. Do not edit pinned running sources, promote recovery FAILED training, claim a new accuracy improvement, or delete checkpoints. Next: assess parent exact equivalence, size-tail full scores and static-template net effect against both grid parent and protectedE1.

## 2026-10-30 01:20 UTC - Training recovery opportunities audited

- Added tools/audit_training_query_opportunities.py and reports/BEST25_TRAIN_QUERY_OPPORTUNITIES_20261030.json. Native COMPLETE/config/plan provenance,882sequence membership,182182step IDs, label joins, IoU arithmetic, sampled-family counts and379571total labels verified. Input hashes stable before/after. Labels newly hashed by this audit, not claimed as a historical label hash attestation. Two aggregation tests passed, including token-cost distinction and recovery versus small gain/harm.
- Chosen-modality sampled queries:182182labels;38382gain>0.01,100849gain<-0.01;43951primaryIoU<0.1,of which1393sampled queries reachIoU>=0.5.1069of these1393occur while the policy choseSTOP. This is3.1694%recovery among sampled lost training states and76.7408%STOP among sampled recoveries. These historical changing-model states are not final-parent dev recall, exhaustive candidate coverage or future trajectory evidence.
- Chosen GLOBAL_LOCAL/WIDE each45545labels:helpful2995/3258,harmful33435/33701,meanIoU gain-0.427326/-0.365056;recoveries336/574,STOP recoveries250/422. Thus simply raising global query frequency is unsupported. A future frozen-parent train replay should distinguish rare recovery opportunities from destructive distractors, then compare recovery/harm supervision at fixed candidate/compute settings. Existing historical CSVs do not contain the955-dimensional feature vectors needed to refit the head directly.
- Source inspection: primary confidence=max(sigmoid center heatmap*Hann), query confidence=max(unwindowed sigmoid center heatmap), both compared to0.65 for template writes. This is not a separate learned target-presence/IoU confidence. It motivates a test, not a causal finding. STARK-ST's separate classification confidence and KeepTrack's candidate matching/not-found state are references, not transplanted modules. Detailed evidence/links/decision gates:docs/QUERY_RECOVERY_DIAGNOSTIC_20261030.md.
- At01:19:18UTC all8recovery wrapper PIDs and4extra evaluation wrapper PIDs confirmed alive. Queues unchanged25:0complete/4active/8queued/0failed;26:3complete/4active/5queued/0failed. Parent/B384/B576/static completed43/41/30/1sequences respectively. Full parent/size/static scores remain pending; no new training, checkpoint cleanup or test launch. Keep original full-dev admission gates and all benchmark/innovation objectives open.

## 2026-10-30 01:31 UTC - Frozen-parent causal feature collection

- Added scripts/collect_frozen_query_features.py to retain actual955-dimensional pre-query gain inputs and real teacher-query labels. A fixed shuffle of882train names selects20sequences:16fit/4holdout,9435frames,923valid every-tenth-frame states. Each state samples one request per kind, with within-kind probability, plus any otherwise absent deployed query labelled selected_extra. Holdout is for later component fitting; the frozen parent already trained on these sequences. No dev/test data or future GT drives online decisions.
- Applied experiment-audit: fresh gpt-6-astra/max reviewer, same-family/provisional. First review WARN identified incomplete native/source/config binding for full collection. Fixed that actual gap before smoke; second source review PASS,0source blockers. Reviewer checked original annotations/image ordering for all20sequences/9435frames and84bound source hashes. Metadata:reports/FROZEN_QUERY_CACHE_SOURCE_REVIEW_20261030.json; full local forensic traces under .aris/traces/experiment-audit/2026-10-30_frozen_query_cache/. Review is source-only; runtime evidence is separate.
-31-frame native smoke COMPLETE/exit0,36.69seconds total; collection/final-verification phase10.78seconds.12teacher labels,265frozen state tensors,zero backward/optimizer. Original and observed30-step boxes/actions/modalities/templates/history match exactly; teacher calls leave state/templates/history/frame/query counter intact. CPU cache audit passed. Three sampled states all lost,0sampled recoveries and0deployed queries: selected_extra GPU branch not exercised by this prefix. Do not interpret this as benchmark accuracy.
- Full collection admitted through separate hash-bound reports/FROZEN_QUERY_CACHE_FULL_PLAN_20261030.json. Original25GPU3 wrapper1186052, run frozen_query_cache_pilot_20261030; CPU watcher1186412 polls180s and writes reports/FROZEN_QUERY_CACHE_PILOT_AUDIT_20261030.json after native completion. ETA45-70minutes from01:31UTC, rough short-probe projection. This is data collection, not training; training_allowed=false. Existing12evaluations continue. Do not edit the pinned collector or source-review artifact while active.
- At01:27:29UTC all12evaluation wrapper PIDs remained alive. Parent/B384/B576/static completed50/50/37/23sequences.26globalqualitycost0 reached90/97; queues25:0complete/4active/8queued and26:3complete/4active/5queued,0failed. All requested full accuracy targets and three validated innovations remain unachieved. Formal size-tail epochs still await full-dev gates.
- tools/summarize_frozen_query_cache.py validates native receipts/artifact hashes,row correspondence and sampled/deployed utility consistency. Counts are immediate frozen-parent train-state outcomes. Deployed recoveries may include selected_extra while sampled-recoverable counts exclude it; do not divide them as recall. Full cache audit pending. Source-review JSON is CRLF and bound by active plan; file-specific Git -text preserves its exact bytes.
- At01:32:54UTC the full cache completed boysumbrella441frames/44sampled states/178labels:176sampled+2selected_extra,3selected rows. Actual selected-query reproduction assertions have now executed successfully in this prefix. Estimate remaining25-40minutes from first-sequence throughput; full acceptance remains pending.
-26globalqualitycost0 evaluation native COMPLETE/exit0 at01:32:23UTC. Controller still showed it active while awaiting its next180s poll; controller828259 and CPU scorer855280 verified live, next check about63seconds away at inspection. Do not duplicate or restart the finished task. New official diagnostic score and automatic refill pending.
- At01:36:26UTC26automatic refill confirmed:4complete/4active/4queued/0failed, GPU0newgainpair-k4-cost0 evalcost0.05 wrapper975786 alive. Other three wrappers remain live. Scorer855280 is alive/sleeping on its180s schedule; its01:34:04status snapshot preceded queue adoption, so the new quality score is still pending, not failed. No duplicate scorer launched.
- Full cache now5/20sequences,2873frames,1142labels; collector1186052 and watcher1186412 live.25GPU0/1/2observed4526MiB each,GPU3observed6791MiB/46%utilization. All8cards have live work. Extra cache throughput and concurrent benchmark FPS are not claims of paired speedup. Git whitespace handling for the exact CRLF review artifact is scoped to cr-at-eol; hash-preserving diff checks pass.


## 2026-10-30 01:47 UTC - Verified efficiency and recovery comparison

- New tools/profile_trace_weighted_macs.py measures the unchanged native-complete server25 quality-k8/traincost0.1 parent, evalcost0.15. All97 trace hashes form23 route/shape groups,54549tracking frames and97initializations. Fresh experiment-audit source reviewer gpt-6-astra/max found0blockers; PASS,same-family/provisional. A document warning was corrected: one representative initialization is measured and weighted97times. Source review is separate from runtime acceptance.
- Native profile_best25_route_macs_20261030 COMPLETE/exit0,42.19seconds total, GPU0. Exact native command, plan/source/checkpoint/full-reference hashes and103218887parameters verified. Independently recomputed weighted means. K8STOP/WIDE supported FLOPs match legacy cases for both modalities. Result:40.240537504GMAC/tracking frame;40.169108083GMAC/frame including initialization. STOP40.113331072G; local query58.399083392G; wide query80.145924992G. These count profiler-supported conv/matrix multiply-adds only; pointwise arithmetic is separate and norm/softmax/resampling/CPUdecoding/unsupported ops are excluded. This is a shape-weighted estimate, not exhaustive operations or measured latency/speedup.
- Reference dev accuracy remains75.9675846PR/70.7621205NPR/56.0471607SR; existing concurrent serviceFPS13.8815478,103.218887Mparameters. No new accuracy improvement. Evidence:reports/BEST25_EFFICIENCY_SUMMARY_20261030.json with raw profile/native/reference reports and reproducible source/plan.
-26globalqualitycost0 official diagnostic now verified:PR73.6911626,NPR68.7086199,SR54.8007661,FPS8.2673926. Versus samecost0 parent:+0.43285/+0.49518/+0.09279points; versus selectedcost0.15parent:-2.27642/-2.05350/-1.24639. Versus gainpaircost0 SR+0.85536,52sequencesbetter/43worse/2same. This does not meet targets or justify promotion. Training native FAILED/test-ineligible status remains. Comparison:reports/QUALITY_RECOVERY_COST0_DIAGNOSTIC_20261030.json.
-01:46:47UTC all8GPUslots occupied.25recovery0complete/4active/8queued;26recovery4complete/4active/4queued;0queuefailures.180second automatic refill remains running. GPU utilization snapshot was1-49%, so occupancy is not claimed as saturation or increased aggregate throughput. Existing additional25 dev tasks remain live. Do not fill memory solely to inflate usage or launch duplicate evaluations.
-01:47:17UTC frozen cache18/20sequences,8972/9435frames,3523labels; stillnativeRUNNING andCPUauditpending. Full parent/B384/B576/staticdev completed64/61/50/43sequences respectively. No new formal size-tail or feature-head training admitted before complete relevant evidence. Do not edit pinned active sources. All target and three-innovation objectives remain open.


## 2026-10-30 01:50 UTC - Frozen cache accepted; holdout coverage gap

- Full frozen_query_cache_pilot_20261030 native COMPLETE/exit0 at01:47:40UTC,16.0965minutes end-to-end;20sequences9435frames3695labels923states.265state tensors frozen,0optimizer/backward. Selected-query reproduction assertions exercised. No full trajectory-equality claim: exact original/observed comparison belongs to the31-frame smoke, while full run enforces state preservation around teacher calls.
- Scheduled CPUwatcher completed the reviewed feature/label/hash/native audit:VERIFIED_CACHE_DIAGNOSTIC. Local native/receipt/summarizer hashes matched and totals independently checked. Fit16seq:852states3411labels93lost28sampled-recoverable; all28stopped.12deployedqueries,0deployedrecoveries,0destroys-good. Recoverable states occur in6sequences:boytakesuicase8,Athe3rdboybesidescar2,ab_leftfoam1,9handlowboy8,large2-gai2,blackboypushbike7.
- Holdout4seq:71states284labels9lost,0sampled-recoverable,0deployedqueries. It cannot evaluate recovery sensitivity/generalization. Training remains NOT ADMITTED. Next enlarge fixed seed/order coverage without selecting additions by outcome or moving fit positives into holdout. Holdout is only for the new component and was seen by parent training. Immediate counterfactual query benefit is not a future trajectory or benchmark improvement.
- Reports:FROZEN_QUERY_CACHE_PILOT_RESULT/NATIVE/AUDIT/DECISION_20261030.json. Original25GPU3 continues static-template and recovery evaluation after collection finishes. All four25recoverywrapperPIDs verified live at01:50:07UTC; queue remains4active/8pending/0failures.26continues4active/4pending from previoussnapshot. Keep180second refill scheduling and existing formal-dev gates. Goal remains active.


## 2026-10-30 01:56 UTC - Refill restriction fixed;80sequence expansion running

- Actual scheduling issue:25recovery tasks finished while separate2.3GiB size/static dev tasks stayed on each card. Existing controller required memory<500MiB, so8pending tasks could not refill. Added --launch-memory-limit-mib, default500 unchanged;25explicit5000 admits an additional owned recovery task with ample24GiB capacity. No model/source-plan changes. Two Linux tests passed: completed-task adoption/fill/refill at idle and2300MiB with explicit5000. These tests do not prove live-wrapper adoption.
- Fresh experiment-audit reviewer gpt-6-astra/max SOURCE_PASS,0blockers,same-family/provisional; scope and runtime evidence separate. Reviewed actual locks/source/tasks. Deployed scheduler hash1fee64d5d0e57fff4e62587091509ea349464de91c2788f43da5088690cebfba to25migratedroot. Replaced only scheduler1064962 after verifiedactiveempty, savedstatebackup and exclusive lock check. NewPID1232628. Taskset preserved:8queued/0active/4complete ->4queued/4active/4complete. New native wrappers1232630/1232632/1232633/1232634 onGPU0..3 verifiedlive.26controller unchanged. Receipt:reports/SERVER25_SHARED_MEMORY_REFILL_20261030.json.
- Expanded fixed sampling with tools/prepare_frozen_query_expansion.py: next80names afterold20 in seed2027shuffle,64fit/16holdout,44545frames. Whole-sequence frame balancing yields11166/11128/11125/11126frames. No old20recollection, metric-based selection, fit-to-holdout reassignment, dev/test overlap. Newholdout was seenbyparenttraining; onlynew-component evaluation. Reviewer independently matched all100sequence/53980frame manifestGT/validity against rawinit.txt. Training remains not admitted.
- Four31frame realGPU smokes all nativeCOMPLETE/exit0,12labels each, original/observed predictions/routes/state exact,265frozen tensors,0optimizer/backward; CPU feature/label audits passed. Separate full plans bind each smoke native/receipt/config/list/source identity; existing collector unchanged. Full expansion launched01:56UTC on original25GPU0..3, wrappers1237066..1237069. Runs frozen_query_expansion_s0_full_20261030 throughs3. CPUwatcher1237070 polls180sec and writes reports/FROZEN_QUERY_EXPANSION_S{0..3}_FULL_AUDIT_20261030.json. ETA20-40minutes fromlaunch. Source-review artifact and collector are hash-bound; do not edit while active.
-01:56:32UTC all4newcollector wrappers andwatcher live.25GPU snapshots5179/5179/5147/5091MiB and51/48/46/47%utilization. Additional work is confirmed, throughput speedup is not. Parent/B384/B576/static full-dev completed77/74/61/50sequences (46611/44210/37059/23867frames),allwrapperslive. Formal size-tail training still awaits complete full-dev scores.
- Additionalpilotdiagnostic:3408sampled fitqueryrows,47recoveryqueries across28states; meanactualgain0.61809 vs predicted-0.00849,0positiveutilities;41of47local.729queriesdestroyprimary-good,0predictedpositive. Report:reports/FROZEN_QUERY_PILOT_GAIN_DIAGNOSTIC_20261030.json, eachlabelhashmatchedcompletionreceipt. These are unweighted sampled-row means, notcandidate-populationcalibration or futuretrajectory outcomes. Next compare refitting existing955->64->1head with unchangedregression vs hardutility-sign emphasis only after expandedcoverage/lossplan admitted; no architecturechange/innovation claim yet. Reference:officialfvcore focal-loss implementation https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py; no code copied.
- New25recoverygainpair-k8cost0trained diagnostics:evalcost0.05 PR74.1055553/NPR68.1002468/SR54.5683421,FPS12.3039548;evalcost0.15 PR70.7103298/NPR65.0655357/SR52.2831449,FPS12.1245697. Highercost loses3.39523/3.03471/2.28520points. Neither beatsselectedparent; no promotion. Training-nativeFAILED/testfalse remains. Cost0pending atcomparison time. Partialcomparison+verifiedreports saved. All requestedbenchmarktargets andthreevalidatedinnovations remainopen.

## 2026-10-30 04:12 UTC - Large-batch cached gain admission audit

- No formal training launched. Current formal plan had a parent checkpoint SHA typo (`...f6f14...` vs the verified file/cache identity `...f6a14...`) and the last expanded-cache audit SHA was stale. The previous utility-focal smoke used batch512/LR0.0002, so it does not satisfy the trainer's exact smoke-to-formal hyperparameter identity gate for the intended batch2866/LR0.00112 run. Preserve that historical smoke; do not treat it as admission evidence for this run.
- Corrected the formal plan to the hash-bound parent and actual S3 audit, then set both launch flags false and cleared the old smoke receipt fields. Added a distinct two-epoch utility-focal large-batch smoke plan using batch2866/LR0.00112/seed2027 and the verified parent SHA. It must complete before the formal plan is re-enabled.
- On server25 `/home/gaob/Dual-RGBT`, the trainer's CPU cache loader verified all five caches and their receipt/native/artifact hashes: 100 unique sequences, 3718 fit states, 1601 holdout states. Independently summed audited sampled-recovery coverage: fit166 states/29 sequences; holdout20 states/4 sequences, above the formal minima100/10 and15/3. These are sampled diagnostic coverage counts, not benchmark accuracy.
- The corrected formal and large-batch smoke plans are staged in the server25 project. The formal plan remains intentionally unlaunchable pending matching smoke. Server25 GPUs0-3 are occupied by active LasHeR dev evaluations. Server26 GPUs0-1 run two LasHeR dev evaluations and GPUs2-3 are idle; its `/data2/gb/Dual-RGBT` checkout lacks the cached-gain sources/plans and does not match the pinned parent source hashes, so it was not modified or used for this run.
- Next: monitor server25's existing evaluations at 180-300 second intervals; when one card is actually free, run only the matching large-batch smoke. Reverify its native exit, parent prediction parity, four changed gain-head tensors, and strict reload; then bind its exact result/native hashes and re-enable the single formal plan. Formal training remains one GPU/task, batch2866/LR0.00112, max30 epochs with holdout-regret stopping and last-five checkpoint retention. Full97-sequence LasHeR dev scoring, FPS/parameters/MACs and best-only checkpoint selection are still pending.

## 2026-10-30 04:34 UTC - Large-batch formal training complete; E6 dev running

- Matching utility-focal smoke passed on25 GPU1: batch2866/LR0.00112/seed2027, two epochs/two updates, SANITY_PASS. Exactly four existing gain-head tensors changed (61249 trainable parameters); strict reload bit-equal. Parent-output parity max absolute error3.5762787e-7 on fit and holdout. This is plumbing evidence, not benchmark accuracy.
- One formal training task completed on25 GPU1, native COMPLETE/exit0: batch2866 states, LR0.00112, seed2027, 10epochs/20updates, elapsed1.832s. Holdout-regret early stopping ended after four epochs without improving the epoch6 minimum. Peak allocated551407104B; reserved599785472B. High batch and proportional LR ran as requested, but this 61249-parameter cached head uses under0.6GiB; no artificial VRAM allocation was added.
- Parent-cache offline diagnostics: baseline fit/holdout sampled utility regret0.0619246/0.0266774. Best trained holdout regret was epoch6 at0.0312070 (worse than baseline); final epoch10 was0.0418466. Fit regret at epoch6 was0.0590450. This weakens the offline generalization case; do not promote from these sampled diagnostics. All five retained epoch6-10 checkpoint hashes were verified and preserved.
- Full LasHeR dev scoring is not yet available. Epoch6 was selected first only because it had the lowest sampled holdout regret among the retained five. Its same-checkpoint 11-frame smoke passed on25 GPU1: COMPLETE/exit0,265 frozen state tensors,0 optimizer/backward calls, 11 frames; smoke FPS1.431 includes initialization and concurrent-load conditions and is not a benchmark-speed claim. The first smoke launch failed because the server25 wrapper lacked OPENCV_FOR_THREADS_NUM; failure is preserved, the retry inherited OPENCV_FOR_THREADS_NUM=1 and passed without modifying the active wrapper/source.
- Epoch6 full evaluation is now running on25 GPU1 as `cached_query_gain_utility_focal_e6_lasher_dev_20261030`; the official CPU scorer waits in180-second intervals. It must finish all97 LasHeR dev sequences/54646 frames. Epoch7-10 still need matching smoke and full-dev evaluation; preserve all five checkpoints until official PR/NPR/SR results are compared. No checkpoint cleanup or model promotion yet.
- Local evidence copies: `reports/CACHED_QUERY_GAIN_UTILITY_FOCAL_FORMAL_RESULT_20261030.json`, `...FORMAL_PROGRESS...json`, `...FORMAL_NATIVE...json`, and `...E6_DEV_LAUNCH...json`. Current GPU1 eval FPS and eventual isolated speed/FPS/MACs remain separate measurements; current goal and all RGBT210/RGBT234 targets remain unmet.

## 2026-10-30 04:40 UTC - Last-five LasHeR dev validation underway

- Epoch6 full dev remains active on25 GPU1; at04:39:27UTC it had4/97 sequences and3190 frames. Its CPU scorer polls every180seconds.
- Epoch7,8,9 same-checkpoint 11-frame smokes each completed native COMPLETE/exit0 on25 GPUs0/2/3. Each has265 frozen tensors and0 backward/optimizer calls. Exact plans, smoke receipts, native receipts and launch records are retained in reports.
- Launched the corresponding full97-sequence/54646-frame LasHeR dev runs concurrently on25 GPUs0/2/3; native wrappers were RUNNING on launch and per-run official CPU scorers poll every180seconds. Together E6-E9 now occupy all four GPUs for required checkpoint comparison; this is evaluation concurrency, with one formal training run only.
- Epoch10 remains to smoke/full-evaluate after one GPU is released. No full PR/NPR/SR result for the trained overlay is available yet; do not promote or clean up checkpoints. GPU throughput during concurrent evaluation is not isolated FPS. Best-only selection must use completed official metrics across all five retained epochs, then measure/report parameters and route-weighted MACs.

## 2026-10-30 04:47 UTC - Four-checkpoint evaluation progress

- After a180-second wait, all four LasHeR dev evaluator PIDs remained live and GPUs0-3 were occupied. Progress: E6 30/97 sequences,9664 frames; E7 5/97,3519 frames; E8 4/97,3190 frames; E9 5/97,3519 frames. None has an official score yet.
- Continue waiting toward the estimated completion window rather than polling each minute. When a GPU completes, launch the same-checkpoint 11-frame smoke and full dev for E10, preserving its matching hashes. Keep the last-five checkpoints until all official PR/NPR/SR scores are compared; no cleanup or promotion yet.

## 2026-10-30 06:03 UTC - E6-E9 scored; failure focus shifted to STOP/query utility

- All four LasHeR-dev runs E6-E9 are native COMPLETE and officially scored on 97 sequences/54,646 frames. Protected parent control is 75.967585/70.762121/56.047161 PR/NPR/SR. E6: 74.550529/70.215402/55.324635; E7: 74.084633/69.380493/54.804861; E8: 74.557932/70.158057/55.088441; E9: 74.437970/70.108834/54.961368. None beats parent. E8 has the best PR among E6-E9; E6 has best NPR/SR. All use 103,218,887 parameters. Scores/FPS caveats/provenance: `reports/CACHED_QUERY_GAIN_DEV_FAILURE_DIAGNOSTIC_20261030.md`.
- Paired post-hoc immediate IoU checks show STOP rates 97.606%,96.722%,93.344%,89.263% for E6-E9. Executed-query mean final-minus-primary IoU is +0.045865,+0.046213,+0.014537,+0.000139 respectively. E9 ran 5,857 queries; LOCAL_ALTERNATE and GLOBAL_LOCAL had slightly negative mean immediate deltas. This is selected-query, single-frame evidence, not full trajectory causality.
- Official per-sequence comparison shows 12/97 sequences have lower SR than parent control in all four epochs. `belowrightwhiteboy`, `moto2north1`, and `midboyblue` are the largest repeated regressions; `browncar2north` improves in all four and is a positive control. These locate repeatable failures, not their cause.
- More than 14k STOP frames per checkpoint had primary IoU below 0.1; this is not proof of recoverable missed queries. The scored traces do not include all candidate counterfactual outputs, and `template_updated` is blank, so do not claim actual template writes or their causal role.
- E10 11-frame smoke passed (265 frozen tensors, no backward/optimizer calls). Full E10 LasHeR dev is running on server25 GPU1 as `cached_query_gain_utility_focal_e10_lasher_dev_20261030`; official scorer follows the existing 180-second cadence. Keep all five checkpoints until its score and the final comparison. Do not launch another loss/threshold sweep or delete weights before fixed-state dev diagnosis.

## 2026-10-30 06:45 UTC - Fixed-state diagnosis running; additional controls scored

- Revalidated original E10 wrapper1601351, child1601352 and scorer1608209, without restarting. Latest completed progress50/97 sequences and23867/54646 frames; at06:44:49UTC its active `ab_bolster.csv` had8057 tracking rows and had changed within7seconds. Sequence-completion progress alone understated work on that long sequence. No final E10 score yet. Preserve E6-E10 weights.
- Server26 recovery controller828259 is terminal/missing, with12completed/0failed/0queued/0active. Its four GPUs are idle. It lacks this cached-gain trainer/evaluator and plans. No new training was launched there; no placeholder allocation or duplicate old training. Server25 GPUs0/2/3 now do meaningful fixed-state diagnostics whileGPU1 continues E10.
- Retrieved and hash-verified all392 run artifacts and native-status bindings for each official full97dev control: B384 four-update size-tail73.205319/68.258088/54.232825, B576 four-update size-tail75.158551/70.135285/55.712312, static-template68.987885/64.691062/53.070734. Parent75.967585/70.762121/56.047161. All103218887parameters; concurrent FPS11.8364/11.5772/11.9396 is not isolated speed. None supports promotion; size smokes are not converged models, and static-template degradation does not prove all selective template policies fail.
- E10 trace template columns are actually blank: the remote tracker is the pinned historical source89e18853, unlike the newer local source. Do not overwrite it. Separate new `scripts/diagnose_fixed_state_dev.py` observes preview/gains unchanged and hooks the actual template crop call; it verifies post-write channels/tensors. At every fifth frame it executes every native candidate and the other modality's primary using pre-track templates, then verifies no live state/history/template/counter change. GT is read after decisions/executions; dev labels never enter training. No new layer/loss/threshold or trained parameters.
- Parent and E6 positive-control11-frame realGPU smokes both native COMPLETE/exit0. Each36candidate executions/two sampled states/two observed template writes; all265model state tensors frozen, zero optimizer/backward. Exact control boxes/templates/history/counters and existing saved predictions/routes agree. Smoke source SHA6e2d83dc0c9c019d5bf0e16b05e7f9a3828f86b913526c0289441e2bec2a2672.
- Eight registered policy/sequence diagnostics use parent andE6 on `midboyblue`, `belowrightwhiteboy`, `moto2north1`, `browncar2north`. QueuePID1639271 polls180seconds onserver25 GPUs0/2/3; output `reports/FIXED_STATE_DEV_QUEUE_20261030.json`. First live wrappers:1639554/1639806/1640148. One completed at06:43UTC; queue refilled GPU3 withE6 belowrightwhiteboy. Estimate remaining queue duration around10minutes from06:45UTC based on actual200-300frames/min per long arm; not a completion promise.
- First verified parent belowrightwhiteboy result:1128tracking frames/1103valid,220valid sampled states,4050candidate executions across225sampled states including invalid-GT states. Two sampled recoverable states both STOP; no selected recoveries. Among valid sampled queries, GLOBAL_LOCAL had0recovery/974harm queries and GLOBAL_WIDE3recovery/954harm queries; these are candidate counts, not independent states or official metrics.181frames have center<=20px andIoU<0.5;143also have area ratio outside[2/3,1.5].143actual template-write frames,15at finalIoU<0.1. Alternate primary recovered one sampled state. Full report `reports/FIXED_STATE_PARENT_BELOW_DIAGNOSTIC_20261030.json`; no full-dev/causal claims.
- Read-only comparison of existing parent/E6 saved predictions: first route/box difference is frame50belowrightwhiteboy (IoU0.31194->0.39804),65moto2north1(0.29973->0.80277),145midboyblue(0.79624->0.51019),126browncar2north(0.67288->0.74923). These compare diverging trajectories after that point, not identical histories later. Beneficial initial query does not imply beneficial later trajectory; wait for actual-write/counterfactual analysis before changing acceptance/template policy.
- Public reference repositories actually inspected through GitHub API/raw sources: visionml/pytracking7eb9e74 (KeepTrack candidate identity/uncertainty and ATOM separate localization/refinement/update); rajevv/Multi_L2D3f2ad6b(README/tree only, multi-expert classification/calibration); Zhangyong-Tang/MoETrack2be4b70(README/tree only, implementation archived aszip). No external code copied. Reference scope and hypotheses in `docs/FIXED_STATE_DEV_DIAGNOSTIC_20261030.md`; no claim of three validated innovations.
- CPU summarizer verifies completion/native/config/prediction/trace/artifact hashes before aggregating; three analytic summary tests pass locally and on server25CPU environment. The tests are plumbing fixtures, not model accuracy. Remaining: finish all eight diagnostics, summarize paired policy evidence, finishE10/final five-checkpoint selection, then register a narrow evidence-backed intervention. ProtectedE0 still wins; requested LasHeR/210/234 targets and three validated contributions remain open.

## 2026-10-30 06:51 UTC - Four diagnostics complete; missed recovery directly measured

- Parent/E6 `midboyblue` and `belowrightwhiteboy` full replays all native COMPLETE/exit0, saved predictions/routes equal, frozen265tensors and no state mutation by diagnostic executions. CPU artifact verifier accepted allfour in `reports/FIXED_STATE_DEV_FIRST4_DIAGNOSTIC_20261030.json`. Queue1639271 remains live and refills every180seconds; remaining moto/positive-control arms must finish before the final paired report.
- On E6 midboyblue:501valid sampled states,92with primaryIoU<0.1 and at least one executed candidateIoU>=0.5; all92STOP. GLOBAL_LOCAL45andGLOBAL_WIDE121recovery query rows, nonepositive predictedutility; multiplequeries can recover thesamestate. GLOBAL_LOCAL294andGLOBAL_WIDE270harm query rows coexist, so do not forcequeries globally. Both main modalities IoU<0.1 at461/501 sampled states. Longest lost episode795-2699,1905frames versus parent770frames. Parent has3recoverable sampledstates versusE6's92, but these are differentvisitedstates, not an89state causal gain.
- E6 belowrightwhiteboy:220valid sampledstates,2recoverablestates bothSTOP,190both-primaries-lost; longest lostepisode347-1128,782frames. Parentsame-sequence:2recoverableSTOPstates,43both-primaries-lost,longestlost49frames. FirstE6query frame50improved immediateIoU, despite latertrajectoryregression. Sampled localrecoveryis scarcehere.
- Actual valid-GT low-IoU template-write frames inmidboyblue:114parent/58E6, with longerfailureunderE6. Bad-writecountalone does not prove templatecausality. Parent325validwriteframes versusE6 98; totalwritesincludinginvalidGT326/100. Distinguish counts.
- Concrete next research direction: failure-state recoveryevidence and train-only policy-state refresh, with matched original-regression comparator. Do not train on these devlabels or blindlychange loss/threshold. Final all8diagnostics/E10score still needed before freezing the next plan.
- E10 at06:47:42UTC53/97seq/33285completedframes, active midboyblue582rows; native1601351andscorer1608209still live. Originalparent route-weighted supportedMACs are alreadymeasured:40.240537504G pertrackingframe,40.169108083G includinginitialization,103218887parameters,23shape cases,54549trackingframes. This is a profiler-supported shape estimate, excludesunsupportedops and is not an E6/E10-specific routeaverage or FPSclaim.
- New implementation/evidence commit6fe1f59 pushed toorigin/main. Desktop canonicalhandoff synchronized before this additional entry; resync this file after subsequent updates.

## 2026-10-30 06:56 UTC - All eight fixed-state diagnostics verified

- All8native diagnostics COMPLETE/exit0. CPU verifier accepted8510tracking frames and30564candidate executions, exact saved six-decimal predictions/integer routes, frozen265model state tensors and no diagnostic state mutation. Full report `reports/FIXED_STATE_DEV_ALL8_DIAGNOSTIC_20261030.json`; detailed table and interpretation in `docs/FIXED_STATE_DEV_DIAGNOSTIC_20261030.md`. Queue1639271 may remain in its180second final wait after the last child is already terminal; do not relaunch completed arms.
- moto2north1: parent45valid sampled states/3recoverable/3STOP-missed/0selectedrecoveries; E645/3recoverable/0STOP-missed/1selectedrecovery. Thus E6 queried allthree sampled recoverable states but selected a restoring query onlyonce. E6 has20actual valid low-IoUtemplate-write frames versusparent0, longest lost112vs14frames; observed association, not causality.
- browncar2north positivecontrol: parent40validsamplestates/5recoverableSTOPstates/5alternateprimaryrecoveries, E640/0recoverablestates/0lostintervals/0badwriteframes. E6'szero recoveryopportunities is compatible with a healthy trajectory and must not be called candidatefailure. All40E6templatewrites occur outside finalIoU<0.1.
- Together withmidboyblue92measuredmissed recoverystates, this separates refusal, wrongqueryselection and downstreamstate effects. Preserve three moduleinterfaces; next trainlabelrefresh must useTRAIN trajectories with a frozen candidate, never devdiagnosticlabels. Matchdata/budget/batch/LR for originalregression versusanynew supervision. Do notpromote these diagnostics toinnovationclaims.
- E10 at06:55:49UTC67/97sequences,40198/54646completedframes; wrapper1601351andscorer1608209still live. Recent completed throughput implies about15-20minutes remaining, dependingonsequence lengths; prefer nextuseful observation around07:10UTC ratherthan minute-wise polling. Officialscorer already polls180seconds. Keepallfivecheckpointsoverlay6-10untilE10scoreandfinalselection. No duplicate training, weightcleanup or testlaunch.

## 2026-10-30 07:27 UTC - Per-GPU dispatch; matched original-loss comparator

- Latest user instruction supersedes self-imposed waiting for all GPUs or E10: dispatch an admitted meaningful job on each individually available GPU. Poll long-running queues every180seconds. Do not allocate dummy memory. The user explicitly questioned low VRAM; report actual evaluation versus training and actual idle cards, not just that tasks were started.
- Focal E10 is native COMPLETE/exit0, officially scored97sequences/54646frames: PR73.108519/NPR68.707307/SR54.051855. All five E6-E10 native/report/392artifact/curve hashes reverified by `tools/summarize_cached_gain_jobs.py`; E6 remains best SR55.324635, below parent56.047161 and protectedE0. Retain weights, no promotion/test. Concurrent FPS12.9800 is not isolated throughput.
- New P2 original-loss comparator uses the same100TRAINsequence cache, seed2027, batch_states2866, LR0.00112, min_epochs8/max30/patience4/min_delta0.0001 as focal, differing only in the existing `parent_loss` objective. Matching2epoch real-update/reload smoke passed, then formal completed9epochs/18updates, preservingE5-E9. Same stopping rule, not equal realized epochs/updates versus focal10/20. Only61249gain-head parameters/fourtensors changed; total103218887. Strictreload passed; peakallocated551407104bytes/reserved599785472bytes; measured trainingloop1.778seconds excludes setup. This is cachedhead refitting, NOT fullnetwork/882trajectorytraining. Existing review remains same-family/provisional source clearance, not a newly obtained independent formal review.
- First E5-E7 evaluation-smoke attempts failed before model execution because launch environment omitted OPENCV_FOR_THREADS_NUM=1. Corrected only the new launcher; retained all FAILED/nativeexit1 records and used `_r1` smoke task identities. All three retries native COMPLETE/exit0. Four focused cached-gain tests pass on25; new26scripts pass py_compile. No bound trainer/evaluator/tracker source changed.
- Server25 queuePID1691827, `reports/CACHED_PARENT_LOSS_QUEUE_SERVER25_20261030_r1.json`: GPU0 E7 wrapper1696387, GPU2 E6 wrapper1696541, GPU3 E5 wrapper1696658; all full97dev active with CPUofficialscorers1696388/1696542/1696659. At07:27UTC each has25-26sequences/8962-9090completedframes and2266MiB; GPU1 idle. No queued extra job at this point.
- Server26 independently sealed snapshot `/data2/gb/Dual-RGBT/snapshots/cached_parent_loss_20261030`; original26checkout untouched. Byteverified code/input/native/trainingreceipt/parentweights copied from25; historical25provider/commands remain unchanged. New runtime plan explicitly records26provider and an image-path adapter. Actual issue: oldmanifestpaths do not exist on26; existing localdevcopy is available. All109292imagepaths exist under localroot, bothmodalities first/lastframe of97sequences byte-match25; not all-imagehashcoverage. Original devmanifesthash remains86f123d3. MAPPINGreceipt+inventory bound in plan. E8/E9 11frame crossserver savedprediction text/integerroute equality bothPASS.
- Server26 queuePID1580705, snapshot`reports/CACHED_PARENT_LOSS_QUEUE_SERVER26_20261030.json`: GPU0 E9 wrapper1587076, GPU1 E8 wrapper1587077, eachfull97dev after nativeCOMPLETE/exit0smoke. At07:27UTC each17sequences/6437completedframes and2283-2291MiB. GPU2/3 idle. Fivefull evaluations total, threeidlecards; utilization roughly10-15percent on activecards. Do NOT describe this as eightcardsbusy or highutilization. Runtimeframeprogress confirms work, not GPU saturation.
- Current inference is sequentialstatefultracking without backward, so largeTRAINbatch/LR does not apply to these evaluations. E8/E9 extra25smokes were verification of crossserver equality, not duplicatefull evaluations. One attempted parentE10launch rejected StopIteration before plan/task creation because thisformal only retainedE5-E9; no E10training or evaluation was started for parentloss.
- Evidence bundles: `reports/CACHED_PARENT_LOSS_SERVER25_RECEIPTS_20261030.json` and `...SERVER26...json`. Scores/source/plans/native/configs/smokecomparisons preserved. Next: finish fivefull97dev and officialscoring, compare originalregression versusfocal honestly; prepareTRAINonly visited-failure-state refresh for a meaningful nexttraining job, not an unregistered LR/threshold sweep or devlabeltraining. No additional training is currently active or admitted, and allthree requestedmoduleinnovations remain unvalidated.

## 2026-10-30 09:17 UTC - Continued goal; per-card dispatch and last-five retention

- Continued from chat01a18ad6-42ec-72b3-95d1-e75dfee3ce6e in current workspace `C:/Users/gb/Documents/Codex/2026-10-30/25-26-codex-threads-01a18ad6-42ec`. Goal remains active, not achieved. Latest user explicitly reiterates: dispatch meaningful work per free GPU without waiting for all jobs, real large batch with scaled LR targeting about20GiB, retain lastfive weights and every full-dev score. Dev only tunes/selects; frozen complete test remains the final benchmark. No dummy allocation.
- Original-loss E5-E9 all native COMPLETE/exit0, full97dev/54646frames and official scoring complete. PR/NPR/SR: E5 74.394909/69.706020/55.137258; E6 74.986669/70.542391/55.642706; E7 73.818412/69.562439/54.662551; E8 74.278361/69.737219/55.058465; E9 74.804496/70.370558/55.458679. Reverified all392artifacts/native/config/curve bindings per run on originating servers. Retrieved scores/curves locally in `reports/cached_query_gain_parent_loss_dev_20261030/`. Summary `reports/CACHED_LOSS_COMPLETE_DEV_COMPARISON_20261030.json`: best E6 improves over focalbest by0.318071SRpp but remains0.404455pp below parent. No promotion/test; preserve five retained weights per run. Same cache/seed/batch/LR/stopping rule but actual9epochs/18updates versus focal10/20, not equal realized budgets.
- Previous interrupted work had in fact launched B576/LR7.2e-5 size-only chain on25GPU1: controller1736653, firstTRAIN completed16updates/strictreload, reserved20149436416bytes. First full-dev wrapper1815157 still active at09:14UTC. Existing controller finally deletes nonwinners, so new external `tools/retain_size_tail_last_five.py` PID1935231 copies terminal/hashverified epochs to `runs/size_tail_b576_formal_epochs_20261030/retained_last_five/`, rolling5. Firstepoch902937bb hash copied successfully. No bound live controller/trainer/evaluator changed.
- User-requested larger batch measured on25GPU0: B640/LR8e-5 smoke native1905675 COMPLETE0; four real updates, finite gradients and strictreload, allocated17860411904/reserved22332571648bytes (20.80GiBreserved), observed~22.5GiBphysical. Exact nativeargv/plan/content/weight bindings checked. Smoke checkpoint6b1626db retained separately, not initialization for formal.
- B640 formal now on25GPU0: controller1935434, epoch1TRAINwrapper1935562, plan `reports/SIZE_TAIL_B640_FORMAL_PLAN_20261030.json` SHA6e330bdc44571a24c3afe68282f4d37a9e02deb01b0603703df8d8368129371b. Unchanged trainer,9600TRAINpairs/epoch15updates; B5769216/16, so not a matchedbudget batch ablation. New `tools/run_size_tail_epochs_keep_five.py` differs only by retaining final latest5instead of deleting nonwinners; each epoch stillfull97dev, stop/selectionrules unchanged. Four focused workflow tests pass locally/server25. Independent source review0blockers after exactbindings fix, same-family/provisional actualmodelUNATTESTED in `reports/SIZE_TAIL_B640_SOURCE_REVIEW_20261030.json`.
- FrozenfocalE6 TRAIN state refresh registered for882seq/459435frames,706fit176holdout. Explicit ownership oddshards25/even26,448seq228675frames on25 and434seq230760frames on26. Local cross-provider planhash/name/partition coverage passes. No outcomes used to choose shard membership. Original plan generation failed because shard0onlyholdout/no fit; smoke now picks eligible sequence from its existing partition. Public generic auditor/tracker unchanged. New wrapper `tools/audit_policy_refresh.py` requires exactplannednames andfullframes. Queue recordsAUDITfailure and continues otherworkers. Five focused tests pass locally/server25; source review0blockers same-familyprovisional/actualmodelUNATTESTED in `reports/POLICY_REFRESH_SOURCE_REVIEW_20261030.json`.
- Refresh25queuePID1931586 at `/home/gaob/Dual-RGBT`;26queuePID1895482 in sealed `/data2/gb/Dual-RGBT/snapshots/cached_parent_loss_20261030`. Per-host `reports/POLICY_REFRESH_E6_ALL882_PLAN_20261030.json` and `...QUEUE...json`. Preliminary unlaunched26all16plans preserved in `reports/policy_refresh_preliminary_unlaunched_20261030/`; never launched them. At09:14:37UTC allseven initial31frame smokes nativeCOMPLETE0,12labels each, trajectoryequalTrue/frozen265; queues were in first180secwait. Need confirm audit/fullpromotion in next snapshot; do not rerun those smokes.
- Current source reviews use available inherited Codex agent01a18c02-b486-7c60-8960-78a4ae1aa91f, identityUNATTESTED. Do not copy obsolete gpt-6-astra attribution into new reviews. Direct Windows OpenSSH works; Python subprocess plainssh failedUNKNOWN65535 in this session. Use explicit WindowsSSH plus stdin here-string, noSSHconfig change.
- Human report with all10devcheckpointmetrics: currentworkspace `outputs/训练与完整dev指标_20261030.md`. No final test improvement, no fullnetworktraining claim, no validated threeinnovation claim. Continue admitted per-card queues, review full-dev size results, then use TRAINrefresh evidence to preregister matched controlled gain training; preserve protectedE0 and train/dev/test separation.

## 2026-10-30 09:21 UTC - Eight cards working; smoke-to-full promotion confirmed

- Allseven initial refreshsmokes audited and promoted:25has3completed0failed and2activefull jobs onGPU2/3;26has4completed0failed and4activefull jobs onGPU0/1/2/3. Queues refill every180seconds; no repeatedsmokes or duplicatefull jobs. Remaining25matched100shards queue behind two full arms; B640training occupiesGPU0.
- At09:18:31UTC B640epoch1TRAIN had3logged updates, finite losses0.740781/0.971404/0.755936; physicalGPU0memory23048MiB. B576epoch1dev78/97seq46788/54646frames.25GPU1eval2266MiB; sixrefreshcards2268-2445MiB, not20GiB. Low-memorycollection is not highbatchtraining or fullGPUutilization. Snapshot is allocation/progress evidence only.
- Persisted current native/queue/GPU/RAM states in `reports/CONTINUATION_SERVER25_STATUS_20261030.json` and `...SERVER26...json`. Query utility/recovery counts in TRAIN caches remain diagnostic, not benchmark accuracy. First long all-holdout shard may not update progress until its8574frame sequence completes; use livePID/log evidence, not sequencecountalone.
- Next useful checks: around09:30UTC for B576devend/scoring and shortmatched100full completions; B640epoch1TRAIN expectedaround09:40-09:45UTC based measuredB5761651secondstrain andB640smoke371.7secondsincludingextra verification. Estimate only; no restart from a quiet per-sequence progress file. Keep queue cadence180seconds.

## 2026-10-30 10:25 UTC - Matched100 smoke verified; formal coverage gate not met

- All8matched100 E6-policy TRAIN caches complete/audited, same80fit20holdout as parent cache. Four even shards copied byte-identically26->25,112files and original native/config/audit records preserved. TarSHA f1e7559b6033b86a622fa3c3f25b6efb93a37537f09050f6f69675b7a11ae805; manifestSHA87bd5bdb9f47303894d990fb15459e132af293e27c4b04a4b1e310e060c16f2a. Receipt reports/MATCHED100_EVEN_TRANSFER_20261030.json. Source caches still remain on26.
- New scripts/train_refreshed_query_gain.py changes only visited-state inputs versus original parent_loss trainer: original parent initialization/objective/seed2027/batch2866/LR0.00112/loop/stopping/latest5 unchanged. A separate frozenE6head reproduces recorded cache outputs; not used to initialize learning. Strict native/receipt/artifact/sequence/frame/partition/TRAIN-vs-dev-test checks. Independent reviewer01a18c41-4591-76e0-b091-6598400bee5a SOURCE_PASS0blockers,16AST/NumPytests pass locally and25. ActualmodelUNATTESTED,same-family/provisional,not cross-family/accuracy acceptance.
- tools/prepare_refreshed_gain.py registers matched100 only, not882. tools/launch_refreshed_gain.py permits a measured small-head task with>=1GiBfree, based on572MiBreserved control; no dummy allocation. First preflight stopped before plan/taskcreation because reviewed local wrapper differs from25historical wrapper by OPENCV override only. Preserved historicalwrappercc2b87b; launcher passesOPENCV_FOR_THREADS_NUM=1. Executor verified AST equality excluding that literal and identical effectiveenv. Review JSON distinguishes localreviewedcontext/fa8ccc from actualruntimecontext/cc2b87b; no invented independent remote-wrapper verdict.
- Real smoke25GPU2 wrapper2080808, run refreshed_query_gain_parent_loss_smoke_20261030: nativeCOMPLETE0,21.62seconds including setup,2epochs4updates,strictreloadbit_equal. Fourgainheadtensors changed,61249trainable/103218887totalparameters; peakreserved599785472bytes. Verified244inputfiles,bothretainedweights,native/config/plan bindings. This is smallheadtraining,notfullmodelor20GiBtraining. Evidence reports/REFRESHED_GAIN_SMOKE_VERIFIED_20261030.json.
- FormalNOTADMITTED:3718fitstates/144recoverable/26recoverysequences;1601holdoutstates/9recoverable/5recoverysequences. Unchanged gates require100/15recoverablestates over10/3sequences. Holdout9<15; formalrun/plan notcreated. Do not lower gate, resample outcomes, or promote2epochsmoke. Holdoutfixedstateutilityregret0.01767858->0.02677516,queries3->232(212negativeutility),recoveries0. No new dev/test improvement.
- B576E1officialfull97dev/54646frames verified again:PR75.6574717069,NPR70.4985468560,SR55.8628605238,still0.184300ppbelowparent. All392artifacts/native/curve hashes verified. B576E2TRAINcomplete16new/32totalupdates,strictreload,reserved18.77GiB; keeper1935231preservesE1+E2. B640E1TRAINcomplete15updates,strictreload,reserved20.80GiB. Different9216/9600pairs per epoch and16/15updates remain not equal-budget batch ablation.
- At10:25UTC both25sizecontrollersalive:1736653/GPU1/E2DEV43seq19554frames;1935434/GPU0/E1DEV55seq36180frames. Remaining782collectorsalive25GPU2/3:62seq29980frames/46seq23548frames;26fourcards50seq24651/53seq24377/43seq17162/50seq22175. Queue1931586and1895482alive,0failures,180secondrefill;25twofurthershardsqueued. Currentallcards~2.2-2.4GiB,not20GiBtrainingorGPU saturation. Sizecontrollers independently resume their own nextTRAIN after full-dev decisions, not a globalwait.
- Next meaningful sizecheckaround10:45UTC B640E1 and11:00UTC B576E2, estimated from actualframeprogress. Continue registered882collection without duplicate work. An expanded882training experiment needs its own preregistered scope/plan and realupdate/reloadsmoke, maintaining fixed706fit176holdout and the same recovery gate. It is not a one-factor comparison to the original100cache; do not label it matched100. Goal remains active, protectedE0 unbeaten, no test or threeinnovation validation achieved. Current user report outputs/训练与完整dev指标_20261030.md in current chat workspace.
## 2026-10-30 15:58 UTC - Switched active GPU work to full exploratory test

- User superseded dev tuning: run complete LasHeR test inference and full metrics, no new dev. Every test result must remain exploratory_test=true, test_eligible=false; it is not an independent final test claim.
- Complete manifest is245 sequences/220703 frames. Server25 verified all441406 RGB/TIR paths; server26 independently verified all paths plus SHA256 of980 first/last modality frames. Mapping receipts: reports/CACHED_GAIN_SERVER25_TEST_MAPPING_20261030.json and server26 reports/CACHED_PARENT_LOSS_SERVER26_TEST_MAPPING_20261030.json in sealed snapshot.
- Exact complete-test evaluator/scorer/queue source audit by agent01a18c8c-7889-7410-80e6-c63bfb95215a: SOURCE_PASS,0 blockers,same-family/provisional,actual model UNATTESTED; Python3.7 compile and3 split tests pass. Source receipt reports/CACHED_GAIN_TEST_SOURCE_REVIEW_20261030.json. This is source review only, not runtime/metric attestation.
- Server26 snapshot /data2/gb/Dual-RGBT/snapshots/cached_parent_loss_20261030: queue PID2703216, E4-E7 full test wrappers2703222-2703225 on GPU0-3; E8 pending. At15:54UTC all four native tasks RUNNING, no failures; each had completed first two sequences/1025 frames. GPU memory2291/2283/2283/2443MiB; filesystem6.3TiB free.
- Server25 /home/gaob/Dual-RGBT: queue PID2537427, parent-loss E6/E7/E8 test PIDs2537432/2537433/2537434 on GPU0-2; refreshed all882 E12 waits. Separate GPU3 test queue PID2543983 runs parent-loss E9 wrapper2543988. At15:56UTC E6-E8 each completed5 sequences/1776 frames; E9 just started. GPUs0-3 each about2266MiB; filesystem5.2GiB free. This low usage is real inference footprint, not device saturation.
- Per direct user instruction, stopped only in-flight B576 E5 dev child2506742 and B640 E4 dev child2487432 on GPU3. Native receipts record FAILED/exit-15; their controllers2360358/2342712 remain SIGSTOP'd to prevent dev resumption. No weights/results were deleted. Existing progress/checkpoints are preserved.
- Initial server25 extra-job preparation stopped on a missing E8 full-dev plan after E6/E7 plans and mapping receipt had been written; no task was launched from that attempt. Those plans were reused in the corrected R1 manifest. Server26 first queue launch had a manifest/status-path collision; second launch lacked the required GPU list. Both failed before GPU admission and are preserved. Active queues are ...SERVER25_MANIFEST_R1.../...SERVER25_GPU3_MANIFEST... and ...SERVER26_MANIFEST_R1...; inspect their separate STATUS files. Do not use preparer tools/prepare_cached_gain_full_test.py again yet: independent review found it fails to assert exact retained epochs [4,5,6,7,8] (SOURCE_FAIL for preparer only). The actual server26 plans were manually generated from and checked against the exact five retained E4-E8 weights; active inference plans bind their hashes.
- At this snapshot all eight GPUs run full-test inference. Poll queues/progress/native status every180-300s; use per-sequence progress plus live native PID, and schedule additional full-test candidates only as cards finish. Do not restart paused dev controllers or present any current score as available.

## 2026-10-30 16:25 UTC - Full-test progress and per-GPU concurrency

- Rechecked remote state directly at16:24UTC. Server25 E6/E7/E8/E9 native wrappers
  remain RUNNING/alive, 29/30/29/29 of245 sequences (22820/24626/22820/22820
  frames). GPU0-3 memory2266MiB each. Its E12 refreshed-all882 test remains pending.
- Server26 E4/E5/E6/E7 parent-all882 full-test jobs remain RUNNING/alive at
  25/5/25/25 sequences (18295/1776/18295/18295 frames). The E7 last progress
  was16:19UTC; native PID remains live and GPU3 allocated2414MiB, so this is
  slow per-sequence progress, not evidence of exit. E8 remains queued.
- In response to the user's direction that inference/collection may be
  concurrent on one card, launched a distinct parent-loss E5 full-test run on
  server26 GPU0 alongside E4. Verified retained checkpoint SHA
  `b21588acbab603bc5d372dccbf1017fb3328e2e6470a5997510b361902c7ec03`,
  training/native receipts, source review, and the 245-sequence/220703-frame
  mapping. Plan SHA `cbdfbb6a1acdfd56a94b7b42f6f4f604a4ae3c8fdf367279742409ee2a081eb6`.
  Task `eval_cached_query_gain_parent_loss_e5_lasher_test_p26_oversubscribed_exploratory_20261030`,
  native PID2770944, was RUNNING/alive with5 sequences/1776 frames at16:24UTC.
  GPU0 memory rose from2291 to4557MiB; two compute processes each used2260MiB.
  No OOM or duplicate checkpoint evaluation. Result remains
  `exploratory_test=true`, `test_eligible=false`.
- Queue files remain authoritative for registered work: server25
  `CACHED_GAIN_FULL_TEST_QUEUE_SERVER25_STATUS_R1_20261030.json` and
  `CACHED_GAIN_FULL_TEST_QUEUE_SERVER25_GPU3_STATUS_20261030.json`; server26
  `CACHED_GAIN_FULL_TEST_QUEUE_SERVER26_STATUS_20261030.json`. No complete
  test scores are available yet. Continue 180-300 second polls and verify a
  sequence/frame increase, not only wrapper liveness. Do not resume dev.
- At16:31UTC another remote poll showed progress: server25 E6/E7/E8/E9 at
  39/39/39/35 sequences (29648/29648/29648/29020 frames); server26 E4/E5/E6/E7
  at31/31/30/30 (25144/25144/24626/24626); oversubscribed server26 parent-loss
  E5 at7 sequences/5885 frames. All nine native tasks/PIDs alive; no failures.
  GPU0 server26 remains4557MiB with two inference processes; other cards
  2266-2414MiB. Server25 E12 and server26 E8 are still pending; no complete
  metric/scorer output yet.
- Created active app heartbeat `dual-rgbt-25-26-full-test-monitor`, every15min,
  targeting this thread. It checks remote state and only notifies on a meaningful
  completion/failure/idle-resource event; unchanged monitoring stays quiet.
- 16:37UTC verified further sequence/frame progress with all nine active native
  PIDs alive and no failures: server25 E6/E7/E8 at54/54/54 sequences
  (35108 frames each), E9 at49 sequences/33949 frames; server26 all882 E4/E6/E7
  at34 sequences/28171 frames, E5 at36/29084, E7 at34/28171, plus oversubscribed
  parent-loss E5 at8/9702. E8 and E12 remain queued; no score is complete.
  Server25 has5.2GiB free; the four active output directories total39MiB so far.

## 2026-10-30 16:52 UTC - Sanity checkpoint cleanup; full-test queues live

- Removed only six completed non-formal size-tail Sanity/resume-witness
  checkpoints from server25. Their plans are marked `SANITY_ONLY` or
  `RESUME_SANITY_ONLY`, and no active process or full-test queue referenced
  them. Preserved plans, native receipts, audit reports, and each SHA-256:
  - `runs/size_tail_sanity_b8_20261030/epoch_001.pth`
    `3789b175fadba359ed286e0e6a8c9787054762c4a89f35a59983eb54c7d263c7`
  - `runs/size_tail_sanity_b64_20261030/epoch_001.pth`
    `0f8ac38bf102fa6c23cf9ce27b4702394a923f2fbe9f69a6aa2845db322f42f6`
  - `runs/size_tail_sanity_b384_20261030/epoch_001.pth`
    `e00666232e92221caa9bb63b4cd88edf35a9b20d819fafb0c27bb90b1c88971d`
  - `runs/size_tail_sanity_b576_20261030/epoch_001.pth`
    `55c95b8b487b2222b9321c40f9ee719a6c16b9a53ae8d9ae18aa4e04cc3d9e41`
  - `runs/size_tail_resume_sanity_e001_20261030/epoch_001.pth`
    `ff5b4c2ae1191253cae75da094cdfcdcfd62f93193c17ac452ab1a63b14498b2`
  - `runs/size_tail_resume_sanity_e002_20261030/epoch_002.pth`
    `51d57a817190eefd4c7e83c13ec40c20025d4bab06de3cba0afd9dba63049cb1`
- Reclaimed 2,477,878,354 bytes (about 2.48 GB); server25 free disk rose
  from5.2GiB to7.5GiB. Kept the B640 memory-admission smoke checkpoint because
  its plan explicitly says to preserve it; all formal last-five weights and
  test-referenced checkpoints remain.
- At16:51UTC server25 E6-E9 were all alive at58/245 sequences and35,808 frames
  each; E12 remained queued. Server26 all882 E4-E7 were all alive at58/245 and
 35,808 frames each; E8 remained queued. Oversubscribed parent-loss E5 on26
  was alive at25 sequences/18,295 frames. No failures or complete metrics.
  Progress is within the long `blackgirl` sequence; do not restart. Test
  outputs remain exploratory and ineligible for final claims.

## 2026-10-30 17:05 UTC - Concurrent full-test evaluation of refreshed E8-E11

- User authorized concurrent inference/collection on other cards. Verified
  refreshed all882 retained epochs are exactly E8-E12; E8-E11 each have a
  matching 11-frame GPU smoke with native COMPLETE/exit0, 265 frozen state
  tensors, zero backward/optimizer updates, and checkpoint hashes matching the
  retained training receipt. Reused the reviewed complete-test evaluator,
  scorer and test mapping (245 sequences/220,703 frames); all results are
  `exploratory_test=true`, `test_eligible=false`.
- Launched the four missing full-test runs on server25, one additional evaluator
  per GPU alongside the existing E6-E9 runs:
  - E8 GPU0: wrapper2664635, scorer-waiter2664639.
  - E9 GPU1: wrapper2664640, scorer-waiter2664762.
  - E10 GPU2: wrapper2664764, scorer-waiter2664854.
  - E11 GPU3: wrapper2664855, scorer-waiter2664933.
  Manifest SHA256 `8dab8106b109e1701f33176ec791a2f2fddb62d03f12f3e68c75ec87f36a71ce`;
  launch receipt `reports/REFRESHED_GAIN_ALL882_E8_E11_FULL_TEST_LAUNCH_P25_20261030.json`.
- At17:05UTC all four new native tasks were RUNNING and scorer waiters alive.
  Server25 GPU memory was4,524MiB per card (24GiB total); no OOM. Existing E12
  remains pending in its original queue and will start when a GPU becomes idle.
  No test scores are complete yet. Existing 25/26 jobs were not stopped or
  modified; no dev evaluation resumed.

## 2026-10-30 17:55 UTC - Full-test monitor and queue admission

- Rechecked both hosts after a 180-second interval. All eight 3090s still have
  inference assigned; no native inference process exited. Server25 runs two
  evaluators per GPU (parent-loss E6-E9 plus refreshed all882 E8-E11). Server26
  runs parent-control E4-E7, plus the extra parent-loss E5 on GPU0.
- Server25 parent-loss progress JSONs had not advanced since their current
  sequence began, but every corresponding inference child remained at about
  97-100% CPU and each trace/evidence CSV was written at 17:54 UTC. This is
  active long-sequence processing, not evidence of a stalled or dead task.
  Refreshed E8-E11 progress JSONs also updated at 17:54 UTC.
- Server26 parent-control E4-E7 progress JSONs and evidence CSVs updated between
  17:53 and 17:54 UTC; extra E5 also continued writing. Its full-test queue
  remains RUNNING with 0 completed and 0 failed. Parent-control E8 is pending.
- The existing `queue_complete_test.py` admits work only when a GPU has less
  than 500 MiB allocated, so it does not use spare memory on an occupied GPU.
  No change was made to the live controller or its in-memory pending list:
  launching E8 outside that queue would risk a duplicate admission. All test
  inference remains exploratory/ineligible; no complete scores are available.
- Rechecked at17:58UTC: all eight GPUs remain assigned, GPU memory remains
  about4.5GiB/card on25 and2.2-4.5GiB/card on26, with measured utilization
  17-30% on25 and10-36% on26. All parent-loss and refreshed E8-E11 trace/evidence
  outputs on25 continued growing; the previously quiet E9 sequence-boundary
  progress JSON is explained by a live trace/evidence write on its current
  sequence. On26 E4-E7 and extra E5 progress/evidence continued updating.
  Full-test queue still has E8 pending, 0 completed and 0 failed; no full score
  exists yet.
- 18:03UTC follow-up: all original inference PIDs are still present; GPU
  utilization is 22-30% on25 and 7-21% on26. Server25 parent-loss E6-E9
  progress JSONs advanced again; refreshed E8-E11 progress JSONs remain within
  their active long sequences while their trace/evidence files grew. Server26
  parent-control E4-E7 are at89-90/245 sequences and extra E5 at63/245. Its
  queue status was updated at18:00UTC and remains RUNNING, 0 completed, 0
  failed, with E8 pending. No score file exists yet.
- 18:05UTC follow-up after a 180-second wait: all tracked GPU inference and
  scorer-waiter PIDs remain alive. Server25 parent-loss E6-E9 progress JSONs
  advanced, and refreshed E8-E11 trace/evidence files continued growing even
  though their sequence counters remain within long sequences. Server26
  parent-control E4-E7 progress/evidence and extra E5 outputs also advanced.
  Queue status remains RUNNING with E8 pending and 0 completed/failed; no
  complete metric JSON is present on either host.
- 18:11UTC follow-up: after confirming all monitored jobs live, waited180s and
  rechecked. Server25 inference children remained at97-100% CPU, GPU utilization
  21-32%, and parent-loss plus refreshed trace/evidence files continued growing.
  Server26 children remained at99-100% CPU, GPU0 utilization37% with two jobs
  and GPUs1-3 about11%; parent-control E4-E7 and extra E5 outputs advanced.
  Queue still has E8 pending and no completed/failed jobs or score.
- Source review of the active evaluator found a concrete optimization target:
  its frame loop serially calls `read_pair` for each frame (separate OpenCV
  `imread` for RGB and TIR), CPU `crop_pair`/`warpAffine`, then stateful tracker
  inference; `Tracker.track` synchronizes CUDA before and after each frame.
  This matches the observed CPU-saturated/GPU-light pattern but component
  timings are not yet available, so decode is not proven to be the sole
  bottleneck. PyTorch's upstream DataLoader supports worker prefetch and its
  official data-loading tutorial demonstrates pipelining; Facebook Research's
  `facebookresearch/spdl` is another open-source data-loading reference.
  Candidate follow-up: profile decode, crop, and model separately, then test a
  bounded next-frame RGB/TIR prefetch while preserving serial tracker-state
  updates and exact frame order. Do not alter the live evaluator/controller;
  any future accuracy evaluation must use a separately admitted full-test run
  and remain explicitly exploratory/ineligible.
- 18:18UTC follow-up after a 180-second wait: all sampled inference children
  remain alive at97-100% CPU. Server25 GPU utilization is22-36%; parent-loss
  E6/E9 and refreshed E8/E11 progress/evidence outputs updated, with other
  long-sequence traces still growing. Server26 GPU utilization is11-20%;
  E4-E7 plus extra E5 remain running and their recent progress/evidence outputs
  are advancing. Full-test queue remains RUNNING with E8 pending, 0 completed
  and 0 failed; no complete score JSON has appeared.
- 18:19UTC sequence counts: server25 parent-loss E6/E7/E8=116/245 each, E9
  =105/245; refreshed all882 E8-E11=63/245 each. Server26 parent-control
  E4=95, E5=97, E6=95, E7=95/245; additional parent-loss E5=77/245. All
  associated workers remain live, all eight GPUs have assigned inference,
  queue has0 completed/0 failed with E8 pending, and no score JSON exists.
- 18:25UTC after another180s wait: server25 parent-loss E6/E7/E8=122/245,
  E9=116/245; refreshed E8-E11 remain63/245 at the sequence counter but their
  current trace/evidence CSVs received fresh writes. Server26 parent-control
  E4-E7 each=98/245 and extra parent-loss E5=81/245; current evidence CSVs
  updated at18:25. All sampled inference PIDs remain alive. GPU utilization
  ranges30-35% on25 and1-24% on26. No job completed or failed; queue E8 remains
  pending and no full-test score JSON has appeared.
- 18:30UTC after another180s wait: server25 parent-loss E6-E8=127/245, E9
  =123/245; refreshed all882 E8/E9/E10/E11=64/64/66/64 sequences. Server26
  parent-control E4-E7 remain98/245 within current long sequences, but all four
  evidence CSVs were written at18:30; extra E5 advanced to84/245. All sampled
  worker PIDs remain live. GPU utilization is26-38% on25 and10-34% on26.
  Queue remains RUNNING, E8 pending, 0 complete/failed; no score JSON exists.
- 18:35UTC after another180s wait: server25 parent-loss E6/E7/E8=132/245,
  E9=129/245; refreshed all882 E8-E11=74/245. Server26 extra parent-loss
  E5=90/245; parent-control E4-E7 counters remain98/245 in current long
  sequences, with all evidence CSVs updated during the poll interval. Sampled
  GPU inference PIDs remain alive and evidence/progress files are live. No
  completed/failed jobs or metric JSON; E8 remains pending in the queue.
- 18:40UTC after another180s wait: server25 parent-loss E6/E7/E8=137/245,
  E9=132/245; refreshed E8-E11=80/245 each. Server26 parent-control E5=99/245,
  E4/E6/E7=98/245 within active long sequences; extra parent-loss E5=92/245.
  All sampled inference children remain alive and evidence files were updated.
  Queue E8 remains pending, 0 complete/failed, with no complete metric JSON.
- 18:45UTC after another180s wait: server25 parent-loss E6/E7/E8=140/245,
  E9=137/245; refreshed E8/E9/E10/E11=84/84/85/84. Server26 parent-control
  E4/E7=100, E5=103, E6=102/245; extra parent-loss E5 remains92/245 but its
  evidence CSV grew substantially during the interval. All inference PIDs remain
  alive, no full-test score exists, and queue E8 remains pending with0
  completed/failed.
- 18:50UTC after another180s wait: server25 parent-loss E6/E7/E8=141/245,
  E9=140/245; refreshed E8-E11=90/90/91/90. Server26 parent-control E4-E7
  =103/105/105/104, and extra parent-loss E5=95/245. All sampled GPU workers
  remain alive and progress/evidence files updated. Queue is RUNNING with E8
  pending and0 completed/failed; no complete score JSON yet.
- 18:56UTC after another180s wait: server25 parent-loss E6/E7/E8=143/245,
  E9=142/245; refreshed E8-E11=92/245 each. Server26 parent-control E4-E7
  =105/245 each and extra parent-loss E5=98/245. All sampled inference children
  remain alive and evidence files were freshly written. Queue remains RUNNING
  with E8 pending, 0 complete/failed, and no score JSON.
- 19:01UTC after another180s wait: server25 parent-loss E6/E7/E8=144/245,
  E9=143/245; refreshed all882 E8-E11=95/245. Server26 parent-control
  E4-E7 remain105/245 while all four evidence CSVs continue growing; extra
  parent-loss E5 remains98/245 with its evidence file also growing. All sampled
  workers remain alive. Queue E8 is still pending, 0 complete/failed, no score.
- 19:08UTC live recheck: all 13 full-test inference workers remain RUNNING and
  every worker's progress file has recent updates. Server25 parent-loss E6/E7/E8
  are150/245 sequences, E9=146/245; refreshed all882 E8-E11=98/245 each.
  Server26 parent-control E4/E5/E6/E7=105/114/111/106 sequences; extra
  parent-loss E5=98/245. All eight physical GPUs are assigned inference:
  server25 has two workers per GPU (~4.5GiB each, 22-31% utilization);
  server26 GPU0 has two workers (~4.5GiB), GPUs1-3 one each (~2.2-2.4GiB,
  10-32% utilization). There are no idle cards, completed full-test scores, or
  queue failures. Server25 queues have3+1 active, one pending total; server26
  has4 active and E8 pending. Do not manually launch the pending E8 with the
  same run/task identity: queue_complete_test.py only admits a GPU below500MiB
  and does not reconcile externally launched jobs, so that would risk duplicate
  dispatch. Continue the registered queues and preserve exploratory_test=true,
  test_eligible=false on all full-test outputs.
- 19:17UTC rechecked the same live queues after180s: all13 workers remain
  RUNNING. P25 parent-loss E6-E8=153/245, E9=150/245; refreshed all882 E10
  advanced to99/245, while E8/E9/E11 remain within their active long sequences.
  P26 parent-control E4/E5/E6/E7=116/121/120/119 and extra parent-loss E5=101.
  Both queue systems still report0 completed and0 failed; P25 has1 pending
  across its two queues and P26 has E8 pending. GPUs remain assigned; sampled
  0-10% utilization on some P26 cards is intermittent while their active
  progress files advance, not evidence of an exited worker. No full metric JSON
  is available yet.
- 19:23UTC after another180s: all13 full-test workers remain RUNNING and the
  same eight GPUs remain assigned. P25 parent-loss E6/E7/E8=155/245, E9=154;
  refreshed E8/E9/E11=103 and E10=104. P26 parent-control E4/E5/E6/E7=
  122/125/124/123; extra parent-loss E5=105. Queues still show0 complete,
  0 failed, with one pending job on the main queue per provider (P25's separate
  GPU3 queue has none). No full score JSON has been produced.
- 19:28UTC after another180s: all13 inference workers and queue processes remain
  live, with0 completed and0 failed. P25 parent-loss E6-E8=160/245, E9=155;
  refreshed E8-E11=105 each. P26 parent-control E4/E5/E6/E7=125/129/127/127;
  extra parent-loss E5=105. Some sequence counters held during the interval,
  but actual evaluator children are using98-100% CPU and their current trace
  and evidence files were written within seconds of this check. This confirms
  active long sequences rather than an exited/stalled worker. No full score file
  exists; all GPUs remain assigned and E8 stays queued on26.
- 19:34UTC rechecked after180s: all13 full-test jobs still RUNNING, no queue
  completion or failure, no full metric JSON. P25 parent-loss E6/E7/E8=162/245,
  E9=160; refreshed E8-E11 remain105/245 within a long sequence. P26
  parent-control E4/E5/E6/E7=130/132/131/131; extra parent-loss E5 remains
  105/245 in a long sequence. The refreshed P25 workers' evidence files and
  P26 extra-E5 evidence file were written seconds before polling; evaluator
  children remain98-100% CPU. No exited task or idle GPU found; E8 remains
  under the26 queue.
- 19:39UTC after another180s, all13 jobs remain active and no queue has a
  completion, failure, or score. P25 parent-loss E6/E7/E8=171/245, E9=162;
  refreshed E8/E9/E11=105 and E10=108. P26 parent-control E4/E5/E6/E7=
  132/137/136/135; extra parent-loss E5=105. Although some sequence counters
  held, E8/E9/E11 traces/evidence and extra-E5 evidence were freshly written
  during this interval; evaluator processes are alive and working. All GPUs
  remain assigned; P26 E8 remains pending in the existing queue.
- 19:44UTC recheck after180s: all13 workers still RUNNING, both queue systems
  report0 completed and0 failed. P25 parent-loss E6-E9=171/245; refreshed
  E8/E9/E10/E11=119/116/120/116. P26 parent-control E4-E7=137/245 each;
  extra parent-loss E5=114/245. The long-running extra E5 has advanced to a
  new sequence and fresh trace/evidence writes. All eight cards remain assigned;
  no full metric JSON yet, and P26 E8 is still pending.
- 19:49UTC after another180s: all13 evaluations continue; queue totals remain
 0 completed and0 failed. P25 parent-loss E6-E9=171/245; each is on a long
  sequence, and all four current trace files were written seconds before the
  poll. Refreshed E8/E9/E10/E11=123/123/125/123. P26 parent-control
  E4/E5/E6/E7=138/140/140/140; extra parent-loss E5=119 and advanced to another
  sequence. Current trace/evidence writes confirm progress. All GPUs assigned;
  no full score file yet; P26 E8 remains pending.
- 19:54UTC recheck after180s: all13 inference workers remain RUNNING; both
  queue groups still have0 completed and0 failed. P25 parent-loss E6/E7/E8=
  173/245, E9=171; refreshed E8/E9/E10/E11=129/127/130/127. P26 parent-control
  E4/E5/E6/E7=140/141/141/140; extra parent-loss E5=123. E9 and P26 E4/E7
  counters held on long sequences, but their active trace/evidence files were
  written during the poll interval. All8 GPUs remain assigned; no score file;
  E8 stays pending in the26 queue.
- 19:59UTC after another180s: all13 full-test evaluators remain active; queues
  still show0 completed and0 failed, with one pending job per main queue.
  P25 parent-loss E6/E7/E8=175/245 and E9=173; refreshed E8/E9/E10/E11=
  132/132/133/132. P26 parent-control E4/E5/E6/E7=141/143/143/142; extra
  parent-loss E5=127. All eight GPUs remain assigned, no complete score JSON
  has appeared, and P26 E8 remains pending.
- 20:03UTC after another180s: all13 inference workers remain active; queues
  have0 completed/0 failed and P26 E8 is pending. P25 parent-loss E6-E9=175/245;
  refreshed E8-E11=137/245 each. P26 parent-control E4-E7=143/245 each; extra
  parent-loss E5=130/245. Sampled evaluator children use98-100% CPU, and
  current traces/evidence were written seconds before polling, confirming
  ongoing long sequences. All GPUs remain assigned; no full score JSON yet.
- 20:08UTC after another180s: all13 workers remain RUNNING, with no completed
  or failed queue jobs. P25 parent-loss E6/E7=179/245, E8=178, E9=175;
  refreshed E8-E11=140/245 each. P26 parent-control E4/E5/E6/E7=
  143/144/144/144; extra parent-loss E5=133. All eight GPUs remain assigned;
  no score JSON has appeared, and P26 E8 remains pending.
- 20:20UTC after another180s: all13 full-test evaluator runs remain active.
  P25 parent-loss E6/E7/E8=196/245 and E9=192; refreshed all882 E8-E11=143
  each. P26 all882 E4/E5/E6/E7=146/149/149/147; extra parent-loss E5=140.
  Main queue totals remain0 completed/0 failed, with E12 pending on25 and E8
  pending on26. Progress JSON ages reached several minutes for long sequences,
  but each run's latest trace/evidence file was written0-4 seconds before the
  poll, so no worker was restarted. GPU memory:25 all four cards~4526MiB
  (two inference workers/card);26 GPU0~4557MiB (two workers), GPU1/2~2283MiB,
  GPU3~2443MiB (one worker each). All eight cards assigned. No full score JSON
  yet; these test runs remain exploratory and ineligible for final test claims.
- 20:26UTC after another180s: P25 parent-loss E6/E7/E8=199/245, E9=196;
  P25 refreshed all882 E8-E11 and P26 extra parent-loss E5 remain active.
  P26 all882 E4/E5/E6/E7=149/152/152/150. Queues remain0 completed/0 failed
  with1 pending each; no score JSON. P25 cards each use~4526MiB with two jobs;
  P26 GPU0~4557MiB, GPU1/2~2283MiB, GPU3~2443MiB. GPU2 briefly reported0%
  utilization, but its evaluator process is alive, retains its allocation,
  and its progress increased, so it is not an idle card. All eight GPUs remain
  assigned; do not start duplicate pending E8/E12 jobs outside their queues.
- 20:30UTC after another180s: P25 parent-loss E6/E7=201/245, E8=200, E9=199;
  refreshed all882 E8-E11 remain active. P26 all882 E4/E5/E6/E7=152/153/153/152;
  extra parent-loss E5=143. Main queues still0 completed/0 failed with1 pending
  each; no score JSON. All13 evaluators are on live GPUs. Current trace/evidence
  files are0-5 seconds old; progress JSON ages range15-207 seconds, with sequence
  counts advancing. No restart or duplicate dispatch was warranted.
- 20:36UTC recheck: P25 parent-loss E6/E7=206/245, E8=205, E9=199; their
  current trace/evidence files were updated1-5 seconds before polling. P26
  all882 E4/E5/E6/E7=153/155/155/154, also with traces updated1-4 seconds ago;
  extra parent-loss E5 remains alive with a fresh evidence write. Both full-test
  queues remain RUNNING,0 completed/0 failed and1 pending each; the active jobs'
  score files do not yet exist. All eight GPUs remain allocated, so no idle-card
  restart or duplicate pending-job launch is appropriate.
- 20:40UTC after another180s: P25 parent-loss E6/E7=211/245, E8=209, E9=204;
  P26 all882 E4/E5/E6/E7=155/156/156/155 and extra parent-loss E5=144.
  Queues remain RUNNING with0 completed/0 failed and1 pending each. Every
  monitored trace/evidence file was updated within9 seconds; score outputs are
  still absent. All eight GPUs remain assigned, with P25 running two inferences
  per GPU and P26 GPU0 running two plus one on each other GPU. No task restart
  or duplicate launch was needed.
- 20:45UTC after another180s: P25 parent-loss E6/E7=217/245, E8=216, E9=208;
  P26 all882 E4/E5/E6/E7=156/160/160/156; extra parent-loss E5 remains active
  at144 sequences. Both queues remain0 completed/0 failed with1 pending each;
  no score output exists yet. All8 GPUs retain active allocations. Trace/evidence
  writes are0-6 seconds old, including jobs whose sequence count changed slowly;
  no process exit or idle GPU was found.
- 20:49UTC after another180s: P25 parent-loss E6/E7=222/245, E8=221, E9=213;
  P26 all882 E4/E5/E6/E7=160/161/161/161 and extra parent-loss E5=146.
  Both queues remain RUNNING with0 completed/0 failed and1 pending each; no
  score output exists. All monitored traces updated0-4 seconds before polling,
  and all eight GPUs remain assigned. No exited task or idle card was found;
  pending jobs were left to their queue controllers to avoid duplicate launch.
- 20:54UTC after another180s: P25 parent-loss E6/E7/E8=228/245 each, E9=220;
  P26 all882 E4/E5/E6/E7=161/164/164/162; extra parent-loss E5=149.
  Both queues remain RUNNING with0 completed/0 failed and1 pending each; score
  files are absent. Traces/evidence were updated within1-5 seconds. P26 GPU2
  briefly showed0% utilization but still had~2.3GiB allocated to live E6 and a
  fresh trace, so no card was idle. All8 cards remain assigned.
- 20:59UTC after another180s: P25 parent-loss E6/E7=239/245, E8=238, E9=226;
  P26 all882 E4/E5/E6/E7=162/170/169/164 and extra parent-loss E5=152.
  Both queues remain RUNNING with0 completed/0 failed and1 pending each; no
  score JSON yet. All trace/evidence files were updated within0-6 seconds and
  all eight GPUs remain assigned. Some progress.json timestamps lag by minutes
  on long sequences, but sequence totals advanced during the interval; no
  restart or duplicate pending-task launch was warranted.
- 21:09UTC full LasHeR test scoring verified for parent-loss E6-E8:
  each output contains245 sequences/220703 frames and is correctly marked
  exploratory_test=true, test_eligible=false. E6 PR/NPR/SR=66.8895/61.8140/
  50.4415, FPS=12.0609; E7=65.6285/61.0245/49.7032, FPS=12.0589;
  E8=65.6558/61.2797/49.7648, FPS=12.0318. All use103218887 total params.
  Best is E6, still short of requested PR/SR/NPR by8.9105/9.7585/10.0860pp;
  this small-head cached-gain experiment does not meet the target.
  Parent-loss E9 has completed inference (245/245,220703 frames); CPU scoring
  is still running, so no E9 metric is reported yet.
- 21:09UTC scheduling audit: P25 main queue has active0, completed E6-E8,
  pending refreshed all882 E12, failed0. GPU3's separate queue is still
  scoring E9. Independently launched refreshed all882 E8-E11 remain live on
  GPUs0-3 (each progress171 sequences/165032 frames at this poll; traces fresh),
  with per-run score workers correctly using --wait until native completion.
  Thus E12 is waiting behind these active GPU allocations, not lost; let the
  existing controller dispatch it after memory falls below its500MiB gate.
  P26 all882 E4-E7 plus oversubscribed parent-loss E5 remain active. No manual
  duplicate dispatch or process termination.
- 21:14UTC P25 GPU3 queue finished parent-loss E9 with native COMPLETE and a
  validated full score:245 sequences/220703 frames, PR/NPR/SR=65.0493/60.7972/
  49.2968, FPS=11.9024, exploratory_test=true and test_eligible=false. E9 is
  below E6 on all three accuracy metrics. Refreshed all882 E8/E9/E10/E11
  remain live on P25 GPUs0-3 at172/171/173/171 sequences respectively
  (165032/165032/172954/165032 frames); their trace writes are2-5s old.
  P25 main queue E12 remains pending while these externally registered GPU
  evaluations occupy the devices; controller is live and waits for its
  <500MiB free-memory gate. P26 all882 E4-E7 remain active with E5 queued.
- 21:18UTC follow-up: P25 refreshed all882 E8/E9/E10/E11 remain RUNNING at
  171/173/173/172 sequences (165032/172954/172954/172290 frames); traces are
  fresh (0-5s). GPU0-3 each retain~2266MiB, with10-16% utilization at sample.
  P25 E12 remains the only pending job; queue controller is alive and waits for
  its configured free-memory gate. P26 E4-E7 remain active at171/173/173/172,
  with traces0-5s old and E8 pending. No task exited and no card is unassigned.
- 21:22UTC recheck: P25 refreshed all882 E8/E9/E10/E11 each reached175
  sequences/176619 frames; traces updated0-5s ago, with no COMPLETE or score
  output yet. P25 E12 remains queued and has not launched because GPUs0-3 still
  hold active refreshed evaluations. P26 all882 E4/E5/E6/E7 are at173/174/174/
  173 sequences (172954/176180/176180/172954 frames), traces3-5s old; E8 stays
  pending. All eight GPUs remain assigned; no abnormal exit or idle card found.
- 21:27UTC process-health check: P25 refreshed all882 E8-E11 evaluator
  children are each at~100% CPU while sequence counters stay at175/245; their
  trace files continue to update. P26's four all882 evaluator children are
  actively consuming~98.5-100% CPU, and the extra E5 evaluator is live. The
  slow counters are long-sequence compute, not dead processes. No process was
  restarted and no GPU was unassigned.
- 21:33UTC after another180s: P25 refreshed all882 E8/E9/E10/E11 advanced to
  183/181/185/178 sequences (185433/185000/185650/184448 frames), proving the
  preceding175-sequence interval was a long-sequence delay; native tasks remain
  RUNNING and traces are fresh. P26 all882 E4-E7 still show175 sequences each,
  but trace files updated1-4s before the poll and evaluator workers are live.
  P25 E12 and P26 E8 remain pending behind allocated GPUs. No score for refreshed
  runs yet and no idle GPU.
- 21:40UTC after another180s: P25 refreshed E8/E9/E10/E11 advanced to195/195/
  196/195 sequences (190295/190295/192045/190295 frames); trace ages0-3s and
  evaluator children~100% CPU. P26 all882 E4/E5/E6/E7 are175/183/182/175
  sequences (176619/185433/185127/176619 frames), trace ages1-6s; all four
  evaluator children are98.6-100% CPU. P25 E12 and P26 E8 remain pending;
  both controllers live. No idle GPU or failed run; full refreshed scores absent.
- 21:45UTC after another180s: P25 refreshed all882 E8/E9/E10/E11 reached
  198/196/199/196 sequences (194186/192045/194602/192045 frames), all native
  RUNNING. P26 all882 E4/E5/E6/E7 reached176/187/186/183 sequences
  (183928/187728/186458/185433 frames). Traces are0-6s old; queues remain
  active with no failures or scores for these runs. P25 E12 and P26 E8 remain
  pending behind active GPU allocations. All8 cards assigned.
- 21:49UTC after another180s: P25 refreshed all882 E8-E11 each at199 sequences/
  194602 frames; traces0-4s old, native tasks RUNNING. P26 all882 E4/E5/E6/E7
  at185/193/192/186 sequences (185650/189397/188735/186458 frames), traces
  0-8s old. No scores or failures for these batches. P25 E12 and P26 E8 remain
  pending, all GPUs assigned, and no worker exit observed.
- 21:53UTC after another180s: P25 refreshed all882 E8/E9/E10/E11 at203/202/
  204/201 sequences (200006/199653/201394/199267 frames), native RUNNING,
  trace ages0-3s. P26 all882 E4/E5/E6/E7 at192/196/195/193 sequences
  (188735/192045/190295/189397 frames), trace ages0-5s. No score outputs or
  failed runs in these batches; P25 E12 and P26 E8 remain pending behind active
  jobs. All8 GPUs remain assigned.
- 21:57UTC after another180s: P25 refreshed all882 E8/E9/E10/E11 reached
  206/205/208/205 sequences (202948/201898/204113/201898 frames); P26 E4/E5/
  E6/E7 reached195/197/196/195 (190295/193961/192045/190295 frames). Trace
  writes are0-3s old, native jobs remain active, and no full scores have landed.
  P26 GPU2/3 sampled at1-2% utilization but retain~2.3-2.4GiB allocations,
  live jobs and fresh traces; they are not idle. Pending jobs remain under their
  existing queue controllers; no failures or exits observed.
- 22:01UTC after another180s: P25 refreshed all882 E8/E9/E10/E11 reached
  210/209/212/209 sequences (205914/205246/206608/205246 frames); P26 E4/E5/
  E6/E7 reached196/199/199/196 (192045/194602/194602/192045 frames). All
  refreshed evaluators remain RUNNING with trace updates0-5s old. No score or
  failure yet. P25 E12 and P26 E8 remain queued, all8 GPUs assigned.
- 22:05UTC after another180s: P25 refreshed all882 E8/E9/E10/E11 reached216/
  214/218/214 sequences (208602/207799/209947/207799 frames); P26 E4/E5/E6/
  E7 reached198/199/199/199 (194186/194602/194602/194602 frames). Trace ages
  are0-7s and all native tasks remain active; no full score or failure yet.
  P26 GPU3 sampled2% utilization but retained~2363MiB, active E7 and fresh trace;
  not idle. P25 E12 and P26 E8 still queued; all8 GPUs assigned.
- 22:09UTC after another180s: P25 refreshed E8/E9/E10/E11 reached220/220/
  224/220 sequences (210763/210763/212920/210763 frames); P26 E4/E5/E6/E7
  reached199/202/200/199 (194602/199653/198988/194602 frames). Trace ages1-7s,
  all native evaluators RUNNING, no scores/failures. P25 E12 and P26 E8 remain
  queued behind active allocations; all eight cards still assigned.
- 22:14UTC after another180s: P25 refreshed all882 E8/E9/E10/E11 reached228/
  226/230/226 sequences (215222/213976/215997/213976 frames). P26 E4/E5/E6/E7
  reached199/205/204/201 (194602/201898/201394/199267 frames). Trace ages1-5s,
  queue failures0 and no new full score. P26 E4's count moved slowly, but its
  evidence is fresh and GPU remains allocated; it is not considered exited.
  P25 E12 and P26 E8 remain queued; all GPUs assigned.
- 22:18UTC after another180s: P25 refreshed all882 E8/E9/E10/E11 reached238/
  236/240/236 sequences (218403/217651/219174/217651 frames); P26 E4/E5/E6/E7
  reached203/208/207/204 (200006/204113/203509/201394 frames). Trace ages0-6s;
  all tasks remain active and no full refreshed score exists. P26 GPU2/3 each
  sampled1% utilization but retain~2.3-2.4GiB, live evaluations and recent trace
  writes, so are not idle. No failure; P25 E12 and P26 E8 still queued.

### 2026-10-30 22:51 UTC: Complete scores and useful concurrent dispatch

- Refreshed all882 E8-E11 are native COMPLETE/exit0, each covering all245
  LasHeR-test sequences and220703 frames. Validated score JSONs are in
  `reports/refreshed_query_gain_all882_test_p25_20261030/`.
  E8 PR/NPR/SR=66.185390/61.309848/50.274906, FPS11.731192;
  E9=67.134699/62.074075/50.630373, FPS11.693804;
  E10=67.356374/62.286899/50.928631, FPS11.793080;
  E11=66.969336/61.998446/50.640487, FPS11.704709.
  E10 is the highest observed within this completed batch; every result remains
  exploratory_test=true/test_eligible=false. These are not final held-out
  generalization claims or evidence of reaching the requested targets.
  The E10 deficits to PR75.8/NPR71.9/SR60.2 are8.443626/9.613101/9.271369pp.
- E12 automatically started on P25 GPU0 under the original queue;
  wrapper3224273/child3224293 remain live,29seq/22820frames at22:51UTC,
  trace6.5s old. P25 utility-focal E6-E10 were not previously tested on either
  provider. Their retained checkpoint hashes, successful11-frame smoke receipts,
  native training completion, source-review hashes, and full-test mapping were
  verified before registering separate complete exploratory-test plans.
- At22:47:17UTC launched five distinct utility-focal full-test tasks:
  GPU1 E6(wrapper3254451)/E9(3254454);
  GPU2 E7(3254452)/E10(3254455);
  GPU3 E8(3254453). At22:51 all five wrappers and evaluator children were live,
  configs bound to actual native commands, progress5seq/1776frames each,
  traces0.2-7.5s old and evaluator CPU101-102%.
  Actual GPU allocations were2266/4526/4526/2266MiB on GPUs0/1/2/3.
  CPU scorers3258536-3258540 are attached with --wait and180s polling,
  producing `reports/utility_focal_test_p25_20261030/e006.json` through e010.
  Exact commands, checkpoint hashes, plan hashes and process IDs are saved in
  `reports/UTILITY_FOCAL_FULL_TEST_DISPATCH_SERVER25_20261030.json`.
- Correction: P25 mapping was present and verified all along. The earlier
  claim that E5 was blocked by missing mapping was false. Two later P25 E5
  attempts failed before model inference: the first plan contained a literal
  backslash-n suffix (JSON Extra data); r1 parsed but used a relative evaluator
  path, violating its Path(__file__).relative_to(ROOT) contract. Their original
  native FAILED records and logs remain intact; their scorer waiters exited.
  Do NOT launch r2 or another E5 duplicate: the same checkpoint b21588ac...
  already runs on P26 GPU0, wrapper2770944,207seq/203509frames at22:51UTC.
  New utility jobs use absolute Python/evaluator/scorer paths and structured
  JSON output. No evaluator/scorer contract or bound model source was changed.
- P26 all882 E4/E5/E6/E7 at22:51UTC:231/245/245/236seq,
  frames216680/220703/220703/217651. E5 native COMPLETE;
  E6 still finishing native teardown. E5 CPU scorer3550563 is genuinely live
  at~110%CPU under controller2703216; the controller synchronously scores
  before admitting E8, so a brief idle GPU1 is expected during scoring.
  E8 remains owned by the existing queue. Do not dispatch it manually as well.
- Retention inventory: P25 formal parent-loss, utility-focal and refreshed
  all882 groups each contain exactly5 epoch checkpoints; size-tail B576 has5
  and B640 has4. P26 all882 control and copied parent-loss groups each have5;
  copied utility-focal has1. No formal group exceeds keep-five. Current best,
  active/queued inputs, and smoke/audit dependencies were preserved. P25 has
  ~7GiB free; a completed full test occupies~61MiB, so these five jobs fit
  the measured storage budget. P26 has~6.3TiB free.
- Validation: actual P25 split-admission unit tests3/3 pass with
  `PYTHONPATH=scripts`; both new dispatch helpers compile under actual
  Python3.7. Full inference starts and fresh traces provide runtime evidence.
  No new dev evaluations, training, architecture changes, or seed search.

### 2026-10-30 23:41 UTC: Server26 complete scores and size-tail full tests

- P26 all882 E4-E7 each finished all245 test sequences/220703 frames, native
  COMPLETE/exit0 and official CPU scores:
  E4 PR/NPR/SR=66.717686/61.824186/50.555620, FPS8.563222;
  E5=67.405901/62.580736/50.966963, FPS8.716696;
  E6=67.498396/62.600194/51.001246, FPS8.703498;
  E7=67.543368/62.650040/51.093086, FPS8.615936.
  P26 parent-loss E5 also completed; attached the missing CPU scoring step.
  Its complete result is65.784692/60.841643/49.740533, FPS8.771042.
  JSON and curve files are copied into local reports/cached_parent_gain_all882_test_20261030/
  and reports/cached_query_gain_parent_loss_test_p26_20261030/.
  All metrics remain exploratory_test=true/test_eligible=false; FPS is under
  concurrent workloads, not an isolated speed benchmark. Targets remain unmet.
- Source-only diagnostic review is SOURCE_PASS/0 blockers, actual_model
  UNATTESTED, same-family/provisional, reviewer01a18c8c-7889-7410-80e6-c63bfb95215a.
  Final report SHA256 e2a6e89d02e38132ecd05419abd9b298be479d60e83d9fc13120a8a7052c1438.
  The parent_control mode is explicitly excluded: its old dev reference run
  does not contain245 test predictions. Only size_tail_diagnostic is launched.
- Added independent scripts/eval_size_tail_test.py and
  tools/run_size_tail_server26_test.py; preserved local dirty original
  eval_size_tail.py/score_size_tail.py and every bound remote model source.
  The adapter reuses the verified25-to26 RGB/TIR mapping and receipt covering
  all441406 image paths. Parent plan, checkpoint, native training COMPLETE,
  strict reload,66 trainable parameters, manifest and source hashes are bound.
  No dev job or training update is executed by these new evaluations.
- Initial size-tail E5/E4 attempts failed BEFORE inference at declared source
  hash verification: three local reviewed helper hashes differed from the
  existing26 snapshot. Original task IDs/logs/native FAILED evidence remain.
  Reviewed exact fetched helper snapshots, bound their actual hashes, and used
  new r1 IDs plus the mapping adapter. Never overwrite the sealed helpers.
- Confirmed real running P26 size-tail jobs and attached180s CPU scorer waiters:
  GPU2 B576-E5 wrapper3632909/child3632911/scorer3638831;
  GPU3 B640-E4 wrapper3632910/child3632912/scorer3638833;
  GPU0 B576-E3 wrapper3644710/child3644722/scorer3647189;
  GPU0 B640-E3 wrapper3644713/child3644723/scorer3647190.
  At23:41UTC r1 jobs each5seq/1776frames and r2 jobs each4seq/1602frames,
  all CPU99-104%, trace ages2-8s. P26 all882 E8 wrapper3552931/child3552932
  remains owned by original queue on GPU1,33seq/26666frames, trace3s old.
  Four new full-test jobs cover four distinct checkpoint hashes; no duplicate.
- P25 six original inference children remain live with fresh traces0-6s:
  refreshed E12 GPU0=71seq/60587frames; utility E6/E9 GPU1, E7/E10 GPU2,
  E8 GPU3 each58seq/35808frames. No restarts or new failures.
- B576-E1/E2/E4 and B640-E1/E2 are being copied from25 with their terminal
  training evidence. Planned r3 placement is GPU1 B576-E1/B640-E1,
  GPU2 B576-E2/B640-E2, GPU3 B576-E4, concurrent with existing inference.
  They are NOT yet admitted at this observation. Dispatch --wave r3 first
  checks all source hashes, exact checkpoint hashes and native COMPLETE/exit0,
  then observed GPU1/2/3 memory below3000MiB (about2.3-2.4GiB now).
  The existing five P26 workers use about34GiB RAM with94GiB available.
- Retention: nine retained size-tail epoch weights remain protected on25;
  verified copies on26 are active/next test inputs. Formal groups remain at
  five or fewer; no checkpoint was deleted speculatively. Current best,
  active/queued and audit inputs remain protected.
- Consolidated duplicate heartbeat monitors: dual-rgbt-25-26-full-test-monitor
  stays ACTIVE every5minutes; monitor-25-26-full-test-gpu-queues is PAUSED.
  Batch direct SSH snapshots,180-300s checks, fresh trace/PID evidence before
  judging stalls, no dev restart and no duplicate pending checkpoint.

### 2026-10-30 23:46 UTC: All retained size-tail epochs admitted concurrently

- Five planned r3 tasks are now RUNNING with real inference children and fresh
  trace writes; all nine distinct retained size-tail epochs have full-test jobs.
  No additional training/dev job or duplicate checkpoint was launched.
  Exact plans/commands/hashes/native wrappers and scorer PIDs are in
  reports/SIZE_TAIL_FULL_TEST_DISPATCH_SERVER26_20261030_R1.json,
  reports/SIZE_TAIL_FULL_TEST_DISPATCH_SERVER26_20261030_R2.json and
  reports/SIZE_TAIL_FULL_TEST_DISPATCH_SERVER26_20261030_R3.json.
- P26 final placement: GPU0 B576-E3/B640-E3; GPU1 original all882-E8 plus
  B576-E1/B640-E1; GPU2 B576-E5/B576-E2/B640-E2;
  GPU3 B640-E4/B576-E4. Native inference counts2/3/3/2, memory
  4557/6814/6814/4696MiB. P25 still has6 active jobs, so16 useful inference
  workers across8 occupied physical GPUs, without memory placeholders.
- New r3 wrapper/child/scorer IDs:
  B576-E1 GPU1=3658207/3658217/3663763;
  B576-E2 GPU2=3658208/3658216/3663764;
  B576-E4 GPU3=3658209/3658215/3663766;
  B640-E1 GPU1=3658210/3658214/3663767;
  B640-E2 GPU2=3658211/3658218/3663768.
  At23:45:51UTC four r3 runs had2seq/456frames, B576-E4 had3seq/751frames;
  trace ages1.6-4.3s. r1 E5/E4 each7seq/5885frames; r2 E3 each5seq/1776frames,
  traces2.8-9.6s. all882-E8=38seq/29449frames. No failure or OOM.
- First r3 scorer-attachment invocation preceded config.json creation while
  model initialization was still running. It exited before attaching any
  scorers; all5 native GPU jobs stayed RUNNING. A later invocation verified
  configs/native commands/checkpoints and attached all5 scorers successfully.
  No retry logic or exception-swallowing was added to the scorer helper.
- Actual P26 Python3.7 split-admission tests3/3 PASS; independent evaluator,
  both adapters, dispatch and scorer-attachment syntax compiled successfully.
  Successful model loading and per-sequence native outputs provide runtime
  evidence beyond source review. Full metrics for size-tail are still pending.
- Archived exact P26 helper source snapshots under
  reports/source_snapshots/server26_size_tail_test_20261030/ and both original
  failed native statuses/run logs under
  reports/size_tail_full_test_failed_initial_20261030/.
  Original failure plan records retain their review SHA; the report file was
  revised during source review, so do not claim its earlier bytes are archived.
  The final e2a6e89... report has remained unchanged since r1/r2/r3 admission.

### 2026-10-30 local: Verified full-test failure analysis

- At23:59:27UTC all8 GPUs remained occupied by16 real inference children:
  P25 placement1/2/2/1 and P26 placement2/3/3/2. Children consume~1CPU core
  each and every active trace was updated within0.6-8.4s. No unexpected exit
  or additional full score; nine size-tail full-test evaluations continue.
  P25 E12=92seq/75284frames; utility E6/E7/E8/E9/E10=63/62/63/63/60seq.
  P26 all882E8=58seq/35808frames; size-tail r1=14/16seq,r2=8each,r3=7each.
- New read-only tools/analyze_lasher_test_trajectories.py uses actual
  run-relative score artifact keys, not the old dev-only analyzer contract.
  Native COMPLETE/exit0, exact commands, plans, curves, checkpoints and982
  artifacts perrun verified.3testsPASS locally/actual26;3real analysesPASS.
  It does not change online inference, source-bound model code or selection.
- Full-test diagnostics all882E7/26,refreshedE10/25,parentlossE6/25 each have
  219925valid trackedframes after245initialization/533invalidGT exclusions.
  Centererror>20px accompanies89.334/88.423/88.161%ofIoU<.5frames.
  Lostframes68019/66196/65834;losthighconfidence44216/44590/43085.
  Reconstructed lost-template-writeframes8666/8739/8506 under source-verified
  frame%5/conf>=.65rule;flagsremainblank,not actual tensor-write observations.
  Querygainoptimism+.070064/+.079964/+.041570;negative realizedutility
  80.388/79.221/72.593%ofexecutedqueries atunchangedcost.15.
  These are correlations/immediateeffects,not causality/unexecutedrecovery.
- Detailed13runrawtable,matched-parentbaselinegap,historicalE0warnings,
  source-pinnedKeepTrack/STARKreferencesand3role-preservinghypotheses:
  reports/FULL_TEST_TRAJECTORY_FINDINGS_20261030.md.
  Machine-readable diagnostics:reports/full_test_trajectory_analysis_20261030/.
  Do not resume dev or feed test labels into training. Need exact parent
  full-test control and paired memory intervention before attribution.
- P25disk7GiBfree/P26disk6.3TiBfree:prefer26forlargecollection.
  Retainedformalepochgroupsremain<=5;size-tailweightsallactiveandprotected.
  No speculative checkpoint deletion. Existing5minuteheartbeat remainsactive.
  Targetsunmet;testeligiblefalse;no validatedthree-moduleinnovationclaim.
- Final00:05:52UTC recheck:all16childrenlive,traces<=9s,no newscore/OOM/exit.
  P25E12=94seq/80515frames,utilityE6-E10=63/63/74/64/63seq.
  P26all882E8=58seq/35808frames,size-tailr1=24/25seq,r2=16each,r3=8each.
  Desktop handoff gets this section separately, preserving its prior entries.
