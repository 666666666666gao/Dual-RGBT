"""Small local audit of seven saved capacity receipts and two source files only."""
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
STEMS = ['VRAM18_WAVE6_EXECUTION', 'VRAM18_WAVE6_HEALTH_NODE_PLAN', 'VRAM18_WAVE6_HEALTH_NODE',
         'VRAM18_WAVE6_HEALTH', 'VRAM18_WAVE4_HEALTH', 'VRAM18_TRAINING_PLAN_INDEX',
         'VRAM18_24_TASK_HOST_CAPACITY_DECISION']
SOURCES = ['tools/launch_vram18_serial_training_wave.py', 'tools/observe_vram18_capacity.py']
paths = ['reports/' + stem + '_20261027.json' for stem in STEMS] + SOURCES
content = {name: (ROOT / name).read_bytes() for name in paths}
hashes = {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}
docs = {stem: json.loads(content[path]) for stem, path in zip(STEMS, paths)}


def cite(name, needle, occurrence=0):
    hits = [i for i, line in enumerate(content[name].decode('utf-8-sig').splitlines(), 1) if needle in line]
    return name + ':' + str(hits[occurrence])


def same(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


execution, plan, node, health, previous, index, hold = [docs[stem] for stem in STEMS]
execution_path, plan_path, node_path, health_path, previous_path, index_path, hold_path = paths[:7]
assert execution['state'] == 'FORMAL_STARTUP_PASS' and execution['phase'] == 6
assert execution['controller_sha256'] == hashes[SOURCES[0]]
assert plan['observer_sha256'] == hashes[SOURCES[1]]
assert plan['execution_sha256'] == health['execution_sha256'] == hold['execution_sha256'] == hashes[execution_path]
assert node['node_plan_sha256'] == hashes[plan_path]
assert health['raw_node_sha256'] == hashes[node_path]
assert health['index_sha256'] == execution['index_sha256'] == previous['index_sha256'] == hashes[index_path]
assert execution['next_observation_epoch'] == plan['observe_at_epoch']
assert node['observed'] >= plan['observe_at_epoch']
start = execution['measurement_start']
before = {task['native']['task']: task for task in start['tasks']}
after = {task['native']['task']: task for task in node['tasks']}
registered = {task['task']: task for task in plan['tasks']}
assert len(start['tasks']) == len(node['tasks']) == len(plan['tasks']) == 24
assert len(before) == 24 and set(before) == set(after) == set(registered)
original = {task['task'] for task in index['original_owners']}
assert len(original) == 8 and len(previous['rates']) == 16 and set(previous['rates']) < set(before)
interval = node['observed'] - start['observed']
rows, rates, retention_vs16, retention_vs_earlier8 = [], {}, {}, {}
active_plan_paths = set()
for name, a in before.items():
    b, r = after[name], registered[name]
    assert a['native']['state'] == b['native']['state'] == 'RUNNING'
    assert a['native']['pid'] == b['native']['pid'] == r['pid']
    assert a['child_pid'] == b['child_pid'] == r['child_pid']
    assert a['native']['physical_gpu'] == b['native']['physical_gpu'] == r['gpu']
    assert a['native']['command'] == b['native']['command'] == r['command']
    assert a['finite_logged_losses'] and b['finite_logged_losses']
    delta = b['logged_updates'] - a['logged_updates']
    assert delta > 0
    rate = delta / interval
    rates[name] = rate
    same(rate, health['rates'][name])
    command = b['native']['command']
    active_plan_paths.add(command[command.index('--plan') + 1])
    if name in original:
        retention_vs16[name] = rate / previous['rates'][name]
        # The older raw baseline is outside this deliberately limited audit.
        # Recover its implied rate from the already saved wave4 rate/retention pair.
        earlier_rate = previous['rates'][name] / previous['original_eight_rate_retention'][name]
        retention_vs_earlier8[name] = rate / earlier_rate
        same(retention_vs_earlier8[name], health['original_eight_rate_retention'][name])
    rows.append(dict(task=name, gpu=b['native']['physical_gpu'], pid=b['native']['pid'], child_pid=b['child_pid'],
        start_updates=a['logged_updates'], end_updates=b['logged_updates'], delta_updates=delta, updates_per_second=rate,
        original_eight=name in original, rate_retention_vs_wave4=retention_vs16.get(name),
        health_retention_vs_earlier_eight=health['original_eight_rate_retention'].get(name)))
assert set(rates) == set(health['rates'])
counts = dict(Counter(row['gpu'] for row in rows))
assert counts == {'0': 6, '1': 6, '2': 6, '3': 6}
total_delta = sum(row['delta_updates'] for row in rows)
aggregate = sum(rates.values())
same(total_delta / interval, aggregate)
same(interval, health['interval_seconds'])
same(aggregate, health['aggregate_updates_per_second'])
same(previous['aggregate_updates_per_second'], health['previous_wave_reference_updates_per_second'])
ratio = aggregate / previous['aggregate_updates_per_second']
same(ratio, health['aggregate_ratio'])
same(min(retention_vs_earlier8.values()), health['min_original_rate_retention'])
swap_start = sum(int(task['host_memory']['Swap'].split()[0]) for task in before.values())
swap_end = sum(int(task['host_memory']['Swap'].split()[0]) for task in after.values())
assert swap_end - swap_start == health['swap_growth_kib']

cpu = node['cpu_sample']
ticks = [b - a for a, b in zip(cpu['before']['cpu_ticks'][:8], cpu['after']['cpu_ticks'][:8])]
assert all(value >= 0 for value in ticks) and sum(ticks) > 0
cpu_interval = cpu['after']['observed'] - cpu['before']['observed']
busy = 100 * (sum(ticks) - ticks[3] - ticks[4]) / sum(ticks)
same(cpu_interval, cpu['interval_seconds'])
same(busy, cpu['busy_percent'])
assert cpu['before']['observed'] < node['observed'] < cpu['after']['observed']
assert hold['status'] == 'HOLD_ADDITIONAL_ADMISSIONS_HOST_RAM' and hold['tasks'] == 24
assert hold['observed'] == start['observed']
assert hold['mem_available_kib'] == start['meminfo_kib']['MemAvailable']
assert hold['swap_free_kib'] == start['meminfo_kib']['SwapFree']
assert hold['gpu_csv'] == start['gpu_csv']
phase8 = [item['path'] for item in index['plans'] if item['phase'] == 8]
phase6 = {item['path'] for item in index['plans'] if item['phase'] == 6}
assert len(phase8) == 8 and not (set(phase8) & active_plan_paths)
assert phase6 <= active_plan_paths and len(execution['cases']) == 8
assert {case['plan'] for case in execution['cases']} == phase6
assert health['status'] == 'ADMIT_NEXT_WAVE'
assert ratio > 1 and health['min_original_rate_retention'] >= .4
assert health['swap_growth_kib'] <= 64 * 1024 and health['mem_available_kib'] >= 8 * 1024**2

wave = dict(start_observed=start['observed'], end_observed=node['observed'], interval_seconds=interval,
    tasks=24, tasks_per_gpu=counts, total_update_delta=total_delta, aggregate_updates_per_second=aggregate,
    wave4_reference_updates_per_second=previous['aggregate_updates_per_second'], aggregate_ratio_vs_wave4=ratio,
    original_eight_retention_vs_wave4=retention_vs16,
    original_eight_retention_vs_wave4_min=min(retention_vs16.values()),
    original_eight_retention_vs_wave4_max=max(retention_vs16.values()),
    health_original_eight_retention_reference='Earlier eight-task baseline, NOT the preceding sixteen-task wave4 interval.',
    health_min_original_rate_retention=health['min_original_rate_retention'],
    earlier_baseline_crosscheck='Consistent with implied older baseline rates recovered from wave4 rates/retention. Older raw baseline files were intentionally not reopened.',
    swap_start_kib=swap_start, swap_end_kib=swap_end, swap_growth_kib=swap_end-swap_start, per_task=rows)
resources = dict(cpu_interval_seconds=cpu_interval, cpu_tick_deltas_first_eight=ticks,
    cpu_total_tick_delta=sum(ticks), cpu_busy_tick_delta=sum(ticks)-ticks[3]-ticks[4], cpu_busy_percent=busy,
    cpu_scope='Whole-host aggregate logical-CPU ticks, idle and iowait excluded; interval includes endpoint snapshot overhead. Not a ten-second-only or training-process-only percentage.',
    startup_mem_available_kib=hold['mem_available_kib'], startup_observed=hold['observed'],
    endpoint_mem_available_kib=node['meminfo_kib']['MemAvailable'], endpoint_snapshot_observed=node['observed'],
    score_extra_read_mem_available_kib=health['mem_available_kib'], score_extra_read_exact_timestamp=None,
    memory_scope='Three different recorded resource readings. Endpoint snapshot time bounds the observation; fields are read sequentially. Extra score-time reading has no independently saved exact timestamp.',
    endpoint_gpu_csv=node['gpu_csv'], gpu_scope='Memory used and utilization are instantaneous device snapshots; utilization is not an interval average.',
    endpoint_swap_free_kib=node['meminfo_kib']['SwapFree'])
evidence = {
    'identity_and_updates':[cite(execution_path, '"measurement_start"'), cite(node_path, '"observed"'), cite(plan_path, '"tasks"'), SOURCES[0]+':20', SOURCES[0]+':46'],
    'aggregate_and_reference':[health_path+':8', previous_path+':9', SOURCES[0]+':47', SOURCES[0]+':52', SOURCES[0]+':55'],
    'cpu':[cite(node_path, '"cpu_sample"'), cite(node_path, '"interval_seconds"'), cite(node_path, '"busy_percent"'), SOURCES[1]+':20', SOURCES[1]+':26'],
    'memory':[cite(node_path, '"MemAvailable"'), cite(health_path, '"mem_available_kib"'), cite(hold_path, '"mem_available_kib"'), SOURCES[0]+':29', SOURCES[0]+':58'],
    'capacity_hold':[hold_path+':2', cite(hold_path, '"scope"'), SOURCES[0]+':60', cite(index_path, '"scope"')]
}
checks = {
    'artifact_seals':dict(status='PASS',details='Execution/index/node-plan/node/controller/observer hashes and references match across the nine allowed files.'),
    'identities_counts_rates':dict(status='PASS',details='All 24 identities, commands, GPUs and child PIDs match both snapshots and the node plan; six per GPU, positive deltas and saved finite-loss flags. Aggregate recomputes exactly within floating-point roundoff.',evidence=evidence['identity_and_updates']),
    'reference_denominators':dict(status='PASS',details='Aggregate denominator is wave4 sixteen-task throughput. HEALTH original_eight_rate_retention still uses the older eight-task baseline. Original-eight retention versus wave4 is separately recomputed from per-task rates.',evidence=evidence['aggregate_and_reference']),
    'cpu_and_swap':dict(status='PASS',details='CPU interval and busy percentage recompute from saved clocks/ticks; summed trainer Swap delta recomputes. Whole-host CPU interval includes snapshot overhead.',evidence=evidence['cpu']),
    'memory_reading_provenance':dict(status='PASS',details='Startup, endpoint and score-time MemAvailable values are correctly kept separate. The extra score-time read has no independently saved exact timestamp.',evidence=evidence['memory']),
    'capacity_decision':dict(status='PASS',details='ADMIT_NEXT_WAVE passes the recorded rate/retention/swap/minimum-free-memory predicate. It does not reserve RAM for eight additional tasks or supersede HOLD_ADDITIONAL_ADMISSIONS_HOST_RAM. Eight registered phase8 plans are absent from both saved 24-task snapshots.',evidence=evidence['capacity_hold']),
    'claim_scope':dict(status='PASS',details='Historical operational observation only; no paired speedup, completed full-training equivalence, accuracy or online FPS conclusion. This limited audit makes no assertion about later live state.')
}
now = datetime.now(timezone.utc).isoformat()
report = dict(audit_skill='experiment-audit', status='SAVED_CAPACITY_RESULT_AUDIT_PASS', verdict='PASS', overall_verdict='pass',
    integrity_status='pass', deterministic_verdict='PASS', blocking_issues=0, blockers=[], date='2026-10-27',generated_at=now,
    review_independence='same-family', acceptance_status='provisional', actual_model='UNATTESTED', reviewer_model='UNATTESTED',
    reviewer_family='openai', agent_id='/root/finish_vram18_result_audit_20261027',
    scope='Independent small-scope audit of seven saved JSON reports and two local source files. No SSH, weights, active-run files, model calls, new NN or other experiments; no whole-repository review.',
    inspected_files=paths, audited_input_hashes=hashes, checks=checks, wave6=wave, resources=resources,
    capacity_decision=dict(throughput_health_status=health['status'], effective_admission_status=hold['status'],
        retained_task_count=24, final_eight_started_in_audited_snapshots=False, unlaunched_phase8_plan_paths=phase8,
        chronology='RAM hold was recorded at startup before the subsequent throughput node. The earlier text that throughput was pending describes that earlier moment; the later throughput PASS is not a RAM-hold release.',
        meaning='Retain existing 24. No authority to start the final eight is inferred from throughput ADMIT_NEXT_WAVE.'),
    reporting_qualifiers=['Compare original-eight per-task rates against wave4 separately from HEALTH older-baseline retention.',
        'Use the actual 25.02456521987915-second whole-host CPU interval including snapshot overhead.',
        'Keep endpoint 15031996 KiB and untimestamped score-time 14645592 KiB MemAvailable distinct.',
        'Do not promote operational aggregate ratios to paired speedup, full-training equivalence, accuracy or online FPS.'],
    evidence=evidence, trace_path=str(TRACE.relative_to(ROOT)).replace('\\','/'))

lines=['# Wave6 capacity result audit','',
    '**PASS — same-family / provisional; actual model UNATTESTED. Audit blocking issues: 0.**','',
    'The saved 24-task node is internally consistent. The effective admission decision remains **HOLD_ADDITIONAL_ADMISSIONS_HOST_RAM**. The throughput-health result ADMIT_NEXT_WAVE does not override that capacity hold; the eight phase8 configurations are absent from the audited snapshots.','',
    'Scope: seven saved JSON reports and the measurement/scoring/observer source only. No SSH, weight reads, active-run access, NN execution or new experiments. This audit checks historical saved observations, not later live state.','',
    '| Quantity | Recomputed result |','|---|---:|',
    f'| Saved identities | 24; six per GPU |',
    f'| Logged update delta / elapsed | {total_delta:,} / {interval:.12f} s |',
    f'| Aggregate throughput | {aggregate:.12f} updates/s |',
    f'| Previous sixteen-task reference | {previous["aggregate_updates_per_second"]:.12f} updates/s |',
    f'| Aggregate/reference ratio | {ratio:.12f} |',
    f'| Original eight, per-task retention versus wave4 | {min(retention_vs16.values()):.12f} to {max(retention_vs16.values()):.12f} |',
    f'| HEALTH minimum retention versus older eight-task baseline | {health["min_original_rate_retention"]:.12f} |',
    f'| Sum of trainer Swap, start / end | {swap_start:,} / {swap_end:,} KiB |',
    f'| Swap change | {swap_end-swap_start:,} KiB |',
    f'| CPU elapsed interval | {cpu_interval:.14f} s |',
    f'| Whole-host CPU busy | {busy:.12f}% |','',
    'The 24 saved wrapper/child identities, commands and GPUs match the start, endpoint and node plan. Every update delta is positive and both snapshots record finite losses. The total is a historical operational aggregate across evolving training states, not a paired speedup experiment.','',
    f'HEALTH uses two different reference scopes: its aggregate ratio compares against the previous sixteen-task wave4 aggregate, while original_eight_rate_retention remains relative to the earlier eight-task baseline. The original eight retain {100*min(retention_vs16.values()):.4f}%–{100*max(retention_vs16.values()):.4f}% of their wave4 per-task rates; their HEALTH minimum of {100*health["min_original_rate_retention"]:.4f}% has the older denominator. The older raw baseline was intentionally not reopened: its implied rates were cross-checked through the saved wave4 rate/retention pairs.','',
    'CPU utilization recomputes from the first eight /proc/stat counters, with idle and iowait excluded. The before/after clocks span 25.02456521987915 seconds because the interval includes the endpoint snapshot. It is neither a ten-second-only result nor utilization attributable only to these training processes.','',
    '| Memory observation | MemAvailable (KiB) | Timestamp provenance |','|---|---:|---|',
    f'| Startup RAM hold | {hold["mem_available_kib"]:,} | {hold["observed"]} |',
    f'| Raw endpoint node | {node["meminfo_kib"]["MemAvailable"]:,} | Snapshot observed {node["observed"]}; sequential reads |',
    f'| Extra health-scoring read | {health["mem_available_kib"]:,} | Separate later read; exact timestamp not retained |','',
    'These are different observations and must not be substituted for one another. The endpoint also records only 136 KiB SwapFree. GPU memory is 14158/14150/14150/14310 MiB, so the roughly 18 GiB-per-card target is not established. GPU utilization columns are instantaneous samples.','',
    'The RAM hold predates the throughput node; its statement that throughput was pending is historical. The later health predicate checks aggregate growth, retained rate, swap growth and at least 8 GiB MemAvailable. It does not budget the memory of the final eight jobs in addition to the reserve. Nothing in the allowed evidence releases the separate RAM hold.','',
    '## Evidence','']
for label, refs in evidence.items():
    lines += ['- '+label.replace('_',' ')+': '+'; '.join('`'+ref+'`' for ref in refs)+'.']
lines += ['', 'The JSON companion includes all 24 update deltas and rates, both retention denominators, raw CPU tick deltas, nine input hashes and the eight phase8 plan paths absent from the snapshots. No full-training equivalence, accuracy or online FPS claim follows from these observations.','']
out=ROOT/'reports/VRAM18_WAVE6_CAPACITY_RESULT_AUDIT_20261027'
out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
out.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
(TRACE/'001-capacity.response.md').write_text('\n'.join(lines),encoding='utf-8')
(TRACE/'001-capacity.meta.json').write_text(json.dumps(dict(timestamp=now,status='ok',verdict='PASS',blocking_issues=0,
    agent_id=report['agent_id'],actual_model='UNATTESTED',review_independence='same-family',acceptance_status='provisional',
    inspected_files=paths,report=str(out.relative_to(ROOT)).replace('\\','/')+'.json'),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=report['status'],verdict='PASS',blocking_issues=0,total_updates=total_delta,
    interval_seconds=interval,aggregate_updates_per_second=aggregate,ratio=ratio,
    original_eight_retention_vs_wave4_range=[min(retention_vs16.values()),max(retention_vs16.values())],
    cpu_interval_seconds=cpu_interval,cpu_busy_percent=busy,swap_growth_kib=swap_end-swap_start)))
