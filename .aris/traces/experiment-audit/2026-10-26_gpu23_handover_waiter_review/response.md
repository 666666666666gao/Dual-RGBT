# GPU2/3 handover waiter source review

**SOURCE_PASS; blocking issues:0.** Requested Astra/max; actual route UNATTESTED, same-family/provisional.

Source SHA256: `c75c125131008971918cb852437d94203e6ce3079c915faf2a2e7a2a960d57df`.

The helper creates one SSH waiter, checks GNUtail capabilities and a bounded dead-PID probe before READY, then waits original owners1028233 and1028232 at240s intervals. It reads no training logs or GPU metrics and does not terminate either owner. Both process exits are required; training success is not inferred.

The handshake records local waiter/SSH and remote waiter PIDs. Terminal stdout/stderr and SSH exit are saved; failed SSH, malformed state or failed reserved driver cannot produce the final started state. Output is bounded to two small JSON records and tail follows /dev/null. Queue registration is unique for the intended single invocation; no retry/fallback exists.

The reviewed driver hash is unchanged. Actual capacity audit hash/blockers match locally. After owner exit the driver independently rechecks capacity/audit, empty compute ownership/<500MiB, original PID absence, and each plan's actual M0/native0/reload/frozen/replay/formal startup gates.

Two remote-sequence and six local-handshake CPU mock cases passed. No actual helper, tail, SSH, remote query, subprocess or NN was executed by this review. Actual tail behavior is checked by the bounded startup probe before READY.
