"""Wait for the two recorded prior owners, then run the admitted reserved wave."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from complete_pipeline import SSH


def main():
    record_path = ROOT / 'reports/GPU23_TRAIN_HANDOVER_QUEUE_20261026.json'
    assert not record_path.exists()
    review = json.loads((ROOT / 'reports/CONCURRENT_WAVE_DRIVER_REVIEW_20261026.json').read_bytes())
    driver = ROOT / 'tools/launch_concurrent_training_wave.py'
    assert review['status'] == 'SOURCE_PASS' and review['blocking_issues'] == 0
    assert hashlib.sha256(driver.read_bytes()).hexdigest() == review['driver_sha256']
    capacity = json.loads((ROOT / 'reports/TRAIN_CONCURRENCY_CAPACITY_DECISION_20261026.json').read_bytes())
    assert capacity['status'] == 'ADMIT_BOUNDED_CONFIG_TRAINING'
    assert capacity['result_audit_blocking_issues'] == 0
    code = """import json,os,subprocess,time
from pathlib import Path
help_text=subprocess.check_output(['tail','--help'],text=True)
assert '--pid' in help_text and '--sleep-interval' in help_text
assert not Path('/proc/2147483647').exists()
subprocess.run(['tail','--pid=2147483647','--sleep-interval=0.1','-f','/dev/null'],check=True,timeout=3)
print(json.dumps(dict(state='WAITING_FOR_PRIOR_GPU_PROCESS_EXIT',remote_waiter_pid=os.getpid(),started=time.time(),owner_pids=[1028233,1028232],pid_check_seconds=240)),flush=True)
for pid in [1028233,1028232]:
 subprocess.run(['tail','--pid='+str(pid),'--sleep-interval=240','-f','/dev/null'],check=True)
print(json.dumps(dict(state='PRIOR_GPU_PROCESSES_EXITED',observed=time.time())),flush=True)
"""
    child = subprocess.Popen(SSH, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding='utf-8')
    child.stdin.write(code)
    child.stdin.close()
    first = child.stdout.readline()
    ready = json.loads(first)
    assert ready['state'] == 'WAITING_FOR_PRIOR_GPU_PROCESS_EXIT'
    record = dict(state=ready['state'], registered=time.time(), local_waiter_pid=os.getpid(),
        local_ssh_pid=child.pid, remote=ready,
        reserved_phase='reserved23', final_gpu_admission='Existing driver requires priorPID absent, no unexpected GPU compute PID and memory.used<500MiB before each launch; no interruption/retry.',
        scope='One persistent SSH wait using GNUtail PID existence checks every240s. No repeated SSH/LLM progress polling, no training logs/metrics, no GPU queries until prior owners have exited. Both prior processes must exit before the reserved pair is attempted; this does not assert their training succeeded.')
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    last = child.stdout.read()
    stderr = child.stderr.read()
    exit_code = child.wait()
    raw = ROOT / 'reports/GPU23_PRIOR_OWNER_EXIT_WAIT_20261026.json'
    raw.write_text(json.dumps(dict(first=ready, remaining_stdout=last, stderr=stderr, ssh_exit_code=exit_code), indent=2)+'\n')
    assert exit_code == 0
    ended = json.loads(last)
    assert ended['state'] == 'PRIOR_GPU_PROCESSES_EXITED'
    record.update(state='PRIOR_OWNERS_EXITED_HARDWARE_PENDING', owners_exit=ended)
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    subprocess.run([sys.executable, str(driver), '--phase', 'reserved23'], cwd=str(ROOT), check=True)
    record.update(state='RESERVED_FORMALS_STARTED', completed=time.time())
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
