# Trace-weighted MAC measurement

Measure the unchanged server25quality-k8 parent, evaluation cost0.15, using
all97hash-verified dev traces:54549tracking frames plus97initializations.
The observed route/shape histogram has23distinct (primary modality,action,
candidate count) groups. Most haveK8; STOP also occurs with1-7candidates.

An independent process loads the exact parent checkpoint. It computes the
original preview and gain head, then forces only the diagnostic routing outputs
to exercise each observed group. A reduced maximum K reproduces the measured
candidate-count shape on a fixed first frame. This is not an accuracy replay.
Each case is warmed, reset and profiled. Assert the observed modality/action/
candidate count and auxiliary token count match the requested group.

Record positive profiler FLOP events. Count only conv2d,mm,bmm,addmm as MAC
families, dividing their multiply/add FLOPs by2. Keep pointwise add/mul FLOPs
separate. Reject unclassified positive FLOP event types rather than silently
calling them MACs. Report all event totals for inspection. Norm,softmax,
resampling,CPUdecoding and unsupported operations are outside this MAC count.
It is a convolution/matrix-multiply MAC estimate, not an exhaustive operation
count or latency prediction. No profiling FPS is reported.

Average the measured MACs using the actual23group frequencies. Provide both
the per-tracking-frame mean and the mean including one representative
initialization weighted across97sequences.
Forcing/reduced-K controls affect routing/shape only and retain the expensive
original computation. They do not modify model parameters or active jobs.
Frozen tensors, absent gradients, source/report/trace/plan hashes and native
command identity are verified. A source audit precedes execution.

Existing official parent dev numbers remain75.9676/70.7621/56.0472(PR/NPR/SR),
FPS13.8815,total parameters103218887. The earlier report's STOP40.1277G and
wide80.1748G divided all profiler FLOPs by2; preserve that historical scope
and distinguish the new conv/matrix-only MAC convention. Neither should be
mistaken for a measured end-to-end speedup under different concurrent loads.
