# Wave6 capacity result audit

**PASS — same-family / provisional; actual model UNATTESTED. Audit blocking issues: 0.**

The saved 24-task node is internally consistent. The effective admission decision remains **HOLD_ADDITIONAL_ADMISSIONS_HOST_RAM**. The throughput-health result ADMIT_NEXT_WAVE does not override that capacity hold; the eight phase8 configurations are absent from the audited snapshots.

Scope: seven saved JSON reports and the measurement/scoring/observer source only. No SSH, weight reads, active-run access, NN execution or new experiments. This audit checks historical saved observations, not later live state.

| Quantity | Recomputed result |
|---|---:|
| Saved identities | 24; six per GPU |
| Logged update delta / elapsed | 21,700 / 1031.522028684616 s |
| Aggregate throughput | 21.036875022118 updates/s |
| Previous sixteen-task reference | 15.700198288483 updates/s |
| Aggregate/reference ratio | 1.339911422491 |
| Original eight, per-task retention versus wave4 | 0.834242330897 to 0.957956676559 |
| HEALTH minimum retention versus older eight-task baseline | 0.746601205226 |
| Sum of trainer Swap, start / end | 7,262,528 / 7,261,004 KiB |
| Swap change | -1,524 KiB |
| CPU elapsed interval | 25.02456521987915 s |
| Whole-host CPU busy | 31.150732092995% |

The 24 saved wrapper/child identities, commands and GPUs match the start, endpoint and node plan. Every update delta is positive and both snapshots record finite losses. The total is a historical operational aggregate across evolving training states, not a paired speedup experiment.

HEALTH uses two different reference scopes: its aggregate ratio compares against the previous sixteen-task wave4 aggregate, while original_eight_rate_retention remains relative to the earlier eight-task baseline. The original eight retain 83.4242%–95.7957% of their wave4 per-task rates; their HEALTH minimum of 74.6601% has the older denominator. The older raw baseline was intentionally not reopened: its implied rates were cross-checked through the saved wave4 rate/retention pairs.

CPU utilization recomputes from the first eight /proc/stat counters, with idle and iowait excluded. The before/after clocks span 25.02456521987915 seconds because the interval includes the endpoint snapshot. It is neither a ten-second-only result nor utilization attributable only to these training processes.

| Memory observation | MemAvailable (KiB) | Timestamp provenance |
|---|---:|---|
| Startup RAM hold | 14,734,688 | 1793084235.8255396 |
| Raw endpoint node | 15,031,996 | Snapshot observed 1793085267.3475683; sequential reads |
| Extra health-scoring read | 14,645,592 | Separate later read; exact timestamp not retained |

These are different observations and must not be substituted for one another. The endpoint also records only 136 KiB SwapFree. GPU memory is 14158/14150/14150/14310 MiB, so the roughly 18 GiB-per-card target is not established. GPU utilization columns are instantaneous samples.

The RAM hold predates the throughput node; its statement that throughput was pending is historical. The later health predicate checks aggregate growth, retained rate, swap growth and at least 8 GiB MemAvailable. It does not budget the memory of the final eight jobs in addition to the reserve. Nothing in the allowed evidence releases the separate RAM hold.

## Evidence

- identity and updates: `reports/VRAM18_WAVE6_EXECUTION_20261027.json:2965`; `reports/VRAM18_WAVE6_HEALTH_NODE_20261027.json:2`; `reports/VRAM18_WAVE6_HEALTH_NODE_PLAN_20261027.json:7`; `tools/launch_vram18_serial_training_wave.py:20`; `tools/launch_vram18_serial_training_wave.py:46`.
- aggregate and reference: `reports/VRAM18_WAVE6_HEALTH_20261027.json:8`; `reports/VRAM18_WAVE4_HEALTH_20261027.json:9`; `tools/launch_vram18_serial_training_wave.py:47`; `tools/launch_vram18_serial_training_wave.py:52`; `tools/launch_vram18_serial_training_wave.py:55`.
- cpu: `reports/VRAM18_WAVE6_HEALTH_NODE_20261027.json:1224`; `reports/VRAM18_WAVE6_HEALTH_NODE_20261027.json:1255`; `reports/VRAM18_WAVE6_HEALTH_NODE_20261027.json:1256`; `tools/observe_vram18_capacity.py:20`; `tools/observe_vram18_capacity.py:26`.
- memory: `reports/VRAM18_WAVE6_HEALTH_NODE_20261027.json:1160`; `reports/VRAM18_WAVE6_HEALTH_20261027.json:24`; `reports/VRAM18_24_TASK_HOST_CAPACITY_DECISION_20261027.json:12`; `tools/launch_vram18_serial_training_wave.py:29`; `tools/launch_vram18_serial_training_wave.py:58`.
- capacity hold: `reports/VRAM18_24_TASK_HOST_CAPACITY_DECISION_20261027.json:2`; `reports/VRAM18_24_TASK_HOST_CAPACITY_DECISION_20261027.json:14`; `tools/launch_vram18_serial_training_wave.py:60`; `reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json:379`.

The JSON companion includes all 24 update deltas and rates, both retention denominators, raw CPU tick deltas, nine input hashes and the eight phase8 plan paths absent from the snapshots. No full-training equivalence, accuracy or online FPS claim follows from these observations.
