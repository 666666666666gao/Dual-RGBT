"""Local stdlib mocks only; never import or run the actual handover helper."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[4]
TRACE=Path(__file__).resolve().parent
SOURCE='tools/wait_for_gpu23_training.py'
DRIVER='tools/launch_concurrent_training_wave.py'
SOURCE_SHA='c75c125131008971918cb852437d94203e6ce3079c915faf2a2e7a2a960d57df'
DRIVER_SHA='cbcd5a4465cabd55dfa95d8797b8ff9832b3e7d3766f37600ec58ccae4324f6b'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(ROOT/SOURCE)==SOURCE_SHA and sha(ROOT/DRIVER)==DRIVER_SHA
review=read(ROOT/'reports/CONCURRENT_WAVE_DRIVER_REVIEW_20261026.json')
assert review['status']=='SOURCE_PASS' and review['blocking_issues']==0 and review['driver_sha256']==DRIVER_SHA
capacity=read(ROOT/'reports/TRAIN_CONCURRENCY_CAPACITY_DECISION_20261026.json')
assert capacity['status']=='ADMIT_BOUNDED_CONFIG_TRAINING' and capacity['result_audit_blocking_issues']==0 and capacity['max_formal_tasks_per_gpu01']>=1
assert sha(ROOT/capacity['result_audit'])==capacity['result_audit_sha256']=='5f2f6d858a92b8cbbc23d0bccfed6b05cd87e41b473879ef5b51e8d6b151ebb9'
audit=read(ROOT/capacity['result_audit']);assert audit['blocking_issues']==0
tree=ast.parse((ROOT/SOURCE).read_text())
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
remote_code=next(n.value for n in ast.walk(main) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('import json,os,subprocess,time'))
remote_tree=ast.parse(remote_code)
assert not any(isinstance(n,(ast.Try,ast.While)) for n in ast.walk(tree))
assert not any(isinstance(n,(ast.Try,ast.While)) for n in ast.walk(remote_tree))
remote_body=[n for n in remote_tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
remote_compiled=compile(ast.fix_missing_locations(ast.Module(body=remote_body,type_ignores=[])),'<mock-remote-waiter>','exec')
remote_results=[]
for fail_owner in [False,True]:
    calls=[]
    def tail_run(command,**kwargs):
        calls.append(dict(command=command,kwargs=kwargs))
        assert kwargs['check'] is True
        if fail_owner and '--pid=1028233' in command:
            raise subprocess.CalledProcessError(1,command)
    mock=SimpleNamespace(check_output=lambda command,**kwargs:'GNU tail --pid --sleep-interval',run=tail_run)
    ns=dict(json=json,os=SimpleNamespace(getpid=lambda:99001),time=SimpleNamespace(time=lambda:1.),
        subprocess=mock,Path=lambda p:SimpleNamespace(exists=lambda:False))
    stdout=io.StringIO();failed=False
    with contextlib.redirect_stdout(stdout):
        try:exec(remote_compiled,ns)
        except subprocess.CalledProcessError:failed=True
    messages=[json.loads(line) for line in stdout.getvalue().splitlines()]
    assert calls[0]['command']==['tail','--pid=2147483647','--sleep-interval=0.1','-f','/dev/null'] and calls[0]['kwargs']['timeout']==3
    assert calls[1]['command']==['tail','--pid=1028233','--sleep-interval=240','-f','/dev/null']
    assert messages[0]['state']=='WAITING_FOR_PRIOR_GPU_PROCESS_EXIT'
    assert messages[0]['owner_pids']==[1028233,1028232] and messages[0]['pid_check_seconds']==240
    if fail_owner:
        assert failed and len(calls)==2 and len(messages)==1
    else:
        assert not failed and calls[2]['command']==['tail','--pid=1028232','--sleep-interval=240','-f','/dev/null']
        assert messages[1]['state']=='PRIOR_GPU_PROCESSES_EXITED'
    remote_results.append(dict(fail_first_owner_wait=fail_owner,tail_calls=len(calls),ready_emitted=True,exit_message_emitted=not fail_owner))

main_compiled=compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),'<mock-local-waiter>','exec')
local_results=[]
for scenario in ['success','ssh_nonzero','bad_terminal','bad_ready','driver_failure','queue_exists']:
    fixture=TRACE/('MOCK_ONLY_'+scenario)
    (fixture/'reports').mkdir(parents=True,exist_ok=False)
    (fixture/'tools').mkdir()
    for name in [DRIVER,'reports/CONCURRENT_WAVE_DRIVER_REVIEW_20261026.json','reports/TRAIN_CONCURRENCY_CAPACITY_DECISION_20261026.json']:
        (fixture/name).write_bytes((ROOT/name).read_bytes())
    queue=fixture/'reports/GPU23_TRAIN_HANDOVER_QUEUE_20261026.json'
    if scenario=='queue_exists':queue.write_text('{"mock_preexisting":true}')
    ready=dict(state='BAD' if scenario=='bad_ready' else 'WAITING_FOR_PRIOR_GPU_PROCESS_EXIT',remote_waiter_pid=99001,started=1.,owner_pids=[1028233,1028232],pid_check_seconds=240)
    ended=dict(state='BAD' if scenario=='bad_terminal' else 'PRIOR_GPU_PROCESSES_EXITED',observed=2.)
    fake_child=SimpleNamespace(pid=99002,stdin=io.StringIO(),stdout=io.StringIO(json.dumps(ready)+'\n'+json.dumps(ended)+'\n'),stderr=io.StringIO('synthetic SSH failure' if scenario=='ssh_nonzero' else ''),wait=lambda:7 if scenario=='ssh_nonzero' else 0)
    operations=[]
    def popen(command,**kwargs):
        assert command==['MOCK_ONLY_SSH'] and kwargs['stdin']==subprocess.PIPE and kwargs['stdout']==subprocess.PIPE and kwargs['stderr']==subprocess.PIPE
        operations.append('mock_SSH');return fake_child
    def driver_run(command,**kwargs):
        assert command==['MOCK_PYTHON',str(fixture/DRIVER),'--phase','reserved23'] and kwargs['cwd']==str(fixture) and kwargs['check'] is True
        operations.append('mock_driver')
        if scenario=='driver_failure':raise subprocess.CalledProcessError(1,command)
    mock=SimpleNamespace(Popen=popen,run=driver_run,PIPE=subprocess.PIPE)
    ns=dict(ROOT=fixture,Path=Path,json=json,hashlib=hashlib,os=SimpleNamespace(getpid=lambda:99003),time=SimpleNamespace(time=lambda:3.),subprocess=mock,SSH=['MOCK_ONLY_SSH'],sys=SimpleNamespace(executable='MOCK_PYTHON'))
    exec(main_compiled,ns)
    failure=None
    with contextlib.redirect_stdout(io.StringIO()):
        try:ns['main']()
        except (AssertionError,subprocess.CalledProcessError) as exc:failure=type(exc).__name__
    if scenario=='success':
        assert failure is None and operations==['mock_SSH','mock_driver']
        q=read(queue);assert q['state']=='RESERVED_FORMALS_STARTED' and q['local_waiter_pid']==99003 and q['local_ssh_pid']==99002 and q['remote']['remote_waiter_pid']==99001
    elif scenario=='driver_failure':
        assert failure=='CalledProcessError' and operations==['mock_SSH','mock_driver'] and read(queue)['state']=='PRIOR_OWNERS_EXITED_HARDWARE_PENDING'
    elif scenario=='queue_exists':assert failure=='AssertionError' and not operations
    else:assert failure=='AssertionError' and operations==['mock_SSH']
    if scenario in ['success','ssh_nonzero','bad_terminal','driver_failure']:
        raw=read(fixture/'reports/GPU23_PRIOR_OWNER_EXIT_WAIT_20261026.json')
        assert raw['ssh_exit_code']==(7 if scenario=='ssh_nonzero' else 0) and raw['first']==ready
    if scenario=='bad_ready':assert not queue.exists()
    local_results.append(dict(scenario=scenario,operations=operations,outcome='PASS'))

cpu=dict(status='CPU_WAITER_CHECK_PASS',source_sha256=SOURCE_SHA,driver_sha256=DRIVER_SHA,
    capacity_decision_sha256=sha(ROOT/'reports/TRAIN_CONCURRENCY_CAPACITY_DECISION_20261026.json'),result_audit_sha256=capacity['result_audit_sha256'],
    remote_waiter_mock_cases=remote_results,local_handshake_handover_mock_cases=local_results,
    actual_helper_import_or_execution=False,actual_remote_query=False,actual_subprocess_launch=False,NN_launch=False,source_edit=False)
(TRACE/'cpu_verification.json').write_text(json.dumps(cpu,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat()
route=dict(reviewer_model_requested='gpt-6-astra',reviewer_reasoning_requested='max',reviewer_model='UNATTESTED',reviewer_reasoning='UNATTESTED',actual_route_attestation='UNATTESTED',review_independence='same-family',acceptance_status='provisional')
report=dict(status='SOURCE_PASS',blocking_issues=0,blockers=[],verdict='PASS',audit_skill='experiment-audit',generated_at=now,date='2026-10-26',
    agent_id='/root/review_joint_quality_train_20261026',verdict_id='/root/review_joint_quality_train_20261026:gpu23-waiter:1',**route,
    source_path=SOURCE,source_sha256=SOURCE_SHA,driver_path=DRIVER,driver_sha256=DRIVER_SHA,
    capacity_decision_sha256=cpu['capacity_decision_sha256'],result_audit_sha256=capacity['result_audit_sha256'],
    checks=[
        dict(status='PASS',lines='16-24',finding='Existing queue prevents a second registered invocation. At registration the driver must match its SOURCE_PASS review hash and capacity must record admitted bounded training with zero audit blockers. Actual local capacity audit SHA matches and blockers are zero; the driver rechecks audit hash and reserved23 capacity>=1 at handover.'),
        dict(status='PASS',lines='25-34',finding='One remote Python checks tail flags and performs a bounded3s dead-PID exit probe before flushing READY. It then waits original PID1028233 and PID1028232 sequentially with240s intervals; both checked subprocesses must return before the exit message. Tail waits /dev/null without reading training logs, GPU data or sending termination signals. Process exit does not establish training success.'),
        dict(status='PASS',lines='36-56',finding='One SSH child receives and closes stdin; READY records actual local waiter, SSH and remote waiter PIDs. Remote program emits only two small JSON lines; tail follows /dev/null, so no recurring stdout stream is left undrained while waiting. Terminal stdout/stderr and SSH return code are recorded, and nonzero SSH or wrong terminal state cannot hand over. A pre-READY failure aborts before queue registration.'),
        dict(status='PASS',lines='57-61',finding='Only after successful owner-exit receipt is the reserved23 driver called with check=True. It independently requires old PIDs absent, exact empty compute ownership/<500MiB per reserved card, unique paths and per-plan real M0/native0/reload/frozen/replay before formal startup. RESERVED_FORMALS_STARTED is written only after driver success. There is no retry, fallback, GPU allocation or stop-existing-owner operation in this helper.')],
    CPU_checks=dict(remote_sequence_cases=2,local_handshake_and_handover_cases=6,actual_GNU_tail_probe_performed_by_reviewer=False),
    runtime_admission='SOURCE_ONLY: waiter and hardware/M0/startup gates not executed by reviewer. GNUtail capability/dead-PID behavior is checked at actual startup before READY.',
    trace_path=TRACE.relative_to(ROOT).as_posix(),cpu_verification_sha256=sha(TRACE/'cpu_verification.json'),
    operations={k:cpu[k] for k in ['actual_helper_import_or_execution','actual_remote_query','actual_subprocess_launch','NN_launch','source_edit']})
report_path=ROOT/'reports/GPU23_HANDOVER_WAITER_SOURCE_REVIEW_20261026.json'
md_path=report_path.with_suffix('.md')
with report_path.open('x',encoding='utf-8') as stream:stream.write(json.dumps(report,indent=2)+'\n')
md=f'''# GPU2/3 handover waiter source review

**SOURCE_PASS; blocking issues:0.** Requested Astra/max; actual route UNATTESTED, same-family/provisional.

Source SHA256: `{SOURCE_SHA}`.

The helper creates one SSH waiter, checks GNUtail capabilities and a bounded dead-PID probe before READY, then waits original owners1028233 and1028232 at240s intervals. It reads no training logs or GPU metrics and does not terminate either owner. Both process exits are required; training success is not inferred.

The handshake records local waiter/SSH and remote waiter PIDs. Terminal stdout/stderr and SSH exit are saved; failed SSH, malformed state or failed reserved driver cannot produce the final started state. Output is bounded to two small JSON records and tail follows /dev/null. Queue registration is unique for the intended single invocation; no retry/fallback exists.

The reviewed driver hash is unchanged. Actual capacity audit hash/blockers match locally. After owner exit the driver independently rechecks capacity/audit, empty compute ownership/<500MiB, original PID absence, and each plan's actual M0/native0/reload/frozen/replay/formal startup gates.

Two remote-sequence and six local-handshake CPU mock cases passed. No actual helper, tail, SSH, remote query, subprocess or NN was executed by this review. Actual tail behavior is checked by the bounded startup probe before READY.
'''
with md_path.open('x',encoding='utf-8') as stream:stream.write(md)
(TRACE/'request.json').write_text(json.dumps(dict(task='Source-only review of one persistent GPU23 owner-exit waiter; no helper execution, remote active query or NN.',source=SOURCE,**route),indent=2)+'\n',encoding='utf-8')
(TRACE/'response.md').write_text(md,encoding='utf-8')
(TRACE/'meta.json').write_text(json.dumps(dict(skill='experiment-audit',generated_at=now,agent_id=report['agent_id'],verdict_id=report['verdict_id'],status='SOURCE_PASS',blocking_issues=0,trace_mode='full',**route),indent=2)+'\n',encoding='utf-8')
(TRACE/'events.jsonl').write_text(json.dumps(dict(event='review_complete',at=now,status='SOURCE_PASS',blocking_issues=0,report_path=report_path.relative_to(ROOT).as_posix()))+'\n',encoding='utf-8')
files=[report_path,md_path]+[TRACE/name for name in ['verify_and_report.py','cpu_verification.json','request.json','response.md','meta.json','events.jsonl']]
(TRACE/'output_seals.json').write_text(json.dumps(dict(file_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files}),indent=2)+'\n',encoding='utf-8')
assert sha(ROOT/SOURCE)==SOURCE_SHA and sha(ROOT/DRIVER)==DRIVER_SHA
print(json.dumps(dict(status='SOURCE_PASS',blocking_issues=0,source_sha256=SOURCE_SHA,report_sha256=sha(report_path),markdown_sha256=sha(md_path)),indent=2))
