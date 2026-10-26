"""Summarize completed, fixed-work native controller training prefixes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_bytes())
    directory = ROOT / plan['output']
    complete = json.loads((directory / 'COMPLETE.json').read_bytes())
    config = json.loads((directory / 'config.json').read_bytes())
    native = json.loads((ROOT / 'runs/pipeline_tasks' / plan['task'] / 'status.json').read_bytes())
    assert native['state'] == 'COMPLETE' and native['exit_code'] == 0
    assert complete['status'] == 'COMPLETE' and complete['total_parameters'] == 103218887
    assert config['plan_sha256'] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert config['plan'] == plan
    assert complete['source_sha256'] == plan['source_sha256'] and complete['input_sha256'] == plan['input_sha256']
    assert len(complete['arms']) == len(plan['arms'])
    rows = []
    reference = None
    for registered, arm in zip(plan['arms'], complete['arms']):
        assert arm['arm'] == registered['name'] and arm['processes'] == registered['processes']
        assert arm['child_exit_codes'] == [0] * arm['processes']
        streams = sorted([v for worker in arm['workers'] for v in worker['streams']], key=lambda v: v['index'])
        assert [v['sequence'] for v in streams] == plan['sequence_names']
        assert all(v['updates'] == plan['updates_per_stream'] for v in streams)
        assert arm['total_updates'] == sum(v['updates'] for v in streams) == 256
        for worker in arm['workers']:
            assert worker['total_parameters'] == 103218887 and worker['frozen_state_tensors'] == 251
            assert len(worker['changed_state_tensors']) == 14
            assert worker['updates'] == sum(v['updates'] for v in worker['streams'])
            assert worker['tracked_frames'] == sum(v['tracked_frames'] for v in worker['streams'])
        for stream in streams:
            raw = directory / arm['arm'] / ('stream%d_steps.json' % stream['index'])
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == stream['steps_sha256']
        fingerprint = [(v['tracked_frames'], v['prediction_route_sha256']) for v in streams]
        if reference is None:
            reference = fingerprint
        before, after = arm['coexisting_training_before'], arm['coexisting_training_after']
        interval = after['observed'] - before['observed']
        observed_rates = []
        for first, last in zip(before['runs'], after['runs']):
            assert first['task'] == last['task'] and first['gpu'] == last['gpu']
            observed_rates.append(dict(gpu=first['gpu'], logged_update_increment=last['logged_updates']-first['logged_updates'],
                                       approximate_updates_per_second=(last['logged_updates']-first['logged_updates'])/interval))
        resident_mib, total_mib, utilization = map(int, arm['gpu_after_work_csv'].split(','))
        rows.append(dict(arm=arm['arm'], additional_processes=arm['processes'],
            total_gpu1_processes=arm['processes']+1,
            total_parameters_per_model=103218887, optimizer_updates=256,
            training_wall_seconds=arm['training_wall_seconds'],
            additional_aggregate_updates_per_second=arm['aggregate_updates_per_second'],
            additional_training_track_frames_per_second=arm['aggregate_training_track_frames_per_second'],
            spawn_to_join_seconds=arm['spawn_to_join_seconds'],
            spawn_to_join_updates_per_second=256/arm['spawn_to_join_seconds'],
            gpu1_resident_after_work_mib=resident_mib, gpu1_total_mib=total_mib,
            gpu1_utilization_at_end_percent=utilization,
            worker_peak_allocated_bytes=[v['peak_allocated_bytes'] for v in arm['workers']],
            worker_peak_reserved_bytes=[v['peak_reserved_bytes'] for v in arm['workers']],
            all_stream_predictions_routes_equal_first=fingerprint==reference,
            coexisting_formal_logged_rates=observed_rates))
    first, last = rows[0]['additional_aggregate_updates_per_second'], rows[-1]['additional_aggregate_updates_per_second']
    result = dict(status='COMPLETED_PREFIX_TIMING_SUMMARY_PENDING_RESULT_AUDIT',
        total_parameters=103218887, arms=rows,
        repeated_additional1_relative_drift_percent=100*(last/first-1),
        measured_highest_additional_throughput_arm=max(rows,key=lambda v:v['additional_aggregate_updates_per_second'])['arm'],
        scope='Same eight TRAIN prefixes /256 native optimizer updates per arm; additional1/2/4/8/1 workers beside one existingGPU1 formal=2/3/5/9 total. CPU/model reset and optimizer setup inside training timer; model construction before barrier excluded and spawn/join timing separately included. Resident NVML snapshot after work is not a continuous peak; utilization is one endpoint snapshot. Worker allocator peaks exclude driver/context/otherprocesses. Existing formal rates use CSV buffers flushed every25updates and are approximate, not same-state causal timings. Route hashes cover only32 update frames per stream, not all tracked frames; no full-frame auxiliary fraction. Warm-prefix/cache/arm-order and evolving original trajectories limit long-run extrapolation. Training track throughput is not online inferenceFPS or completed full882 training. No new checkpoints.')
    destination = ROOT / args.output
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
