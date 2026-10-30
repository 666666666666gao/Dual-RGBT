# Frozen query coverage expansion

The20sequence pilot has28recoverable sampled states in fit but none in its
four-sequence holdout. Keep those partitions intact. Use the next80names in
the same seed2027 shuffle of all882train sequence names. Every fifth new name
is holdout. Assign whole sequences to four shards greedily by frame count,
breaking ties by shuffled position and shard index. This uses dataset size,
not performance. No old sequence is recollected or relabelled into holdout.

The added64fit/16holdout sequences contain44545frames. Four shards have
11166/11128/11125/11126frames. Their fit/holdout counts differ because balancing
uses whole-sequence lengths. Combine partitions across shards before judging
coverage; do not interpret shard-local holdout performance as independent
model selection. Sampling RNG restarts at2027 per shard, and the exact
sequence order is saved in each plan. Each shard requires its own31frame
original-versus-observed smoke and hash-bound full admission. Parent weights,
feature definition, query selection and teacher remain unchanged.

The pilot gain diagnostic verifies every label file against its completion
receipt. Among3408sampled fit query rows,47queries would recover a lost target
toIoU>=0.5. Their mean true gain is0.61809 but mean predicted gain is-0.00849;
none has positive predicted cost-adjusted utility. These47rows correspond to
the28states already reported, not47independent recoveries.41of47are local
requests.729sampled queries destroy a primary-good result; none is predicted
positive. Broadly increasing query frequency could lose this useful rejection.
These conditional row means are unweighted, sampled-family diagnostics, not
exhaustive-candidate calibration or future trajectory outcomes.

Once expanded coverage is verified, first compare refitting the existing
955->64->1gain head with its unchanged regression objective against a
utility-sign objective that emphasizes hard decisions while retaining gain
regression. The concrete loss, sampling weights, training budget and acceptance
thresholds must be registered before fitting. Keep all other model tensors
frozen. A stronger head architecture is not justified by this pilot alone.

The official[fvcore focal loss implementation](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py)
provides a reference for downweighting easy classification examples. This is
a candidate optimization technique, not evidence that it fixes tracking or
a claimed core innovation. No external code has been copied. Preserve
cost-adjusted gain semantics; a binary logit must not silently replace an IoU
gain at inference. Any fitted candidate still requires full LasHeRdev tracking
against the frozen parent and protectedE1, including harmful-query and
template-contamination diagnostics. Offline recovery recall alone is not
acceptance. Training is currently unadmitted.

Separately, the25recovery queue's fixed500MiB launch condition prevented refill
while owned2.3GiB dev tasks remained. The controller now accepts an explicit
launch-memory-limit-mib setting, retaining500by default.25uses5000after checking
current owned tasks and24GiB capacity. This permits roughly2.3GiB additional
recovery tasks alongside existing work. It does not change model configuration
or imply higher throughput; preserve old controller state and adopt live
wrappers when replacing only the scheduler process.26keeps its existing
controller and setting. Both modes passed Linux completed-task adoption/refill tests. Those fixtures
do not exercise adoption of a live wrapper; the real scheduler transition
had no active wrappers and preserved all task identities.
