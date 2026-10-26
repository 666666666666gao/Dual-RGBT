param([Parameter(Mandatory=$true)][string]$Plan)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$planPath = Join-Path $projectRoot $Plan
$configuration = Get-Content -LiteralPath $planPath -Raw -Encoding utf8 | ConvertFrom-Json
$review = Get-Content -LiteralPath (Join-Path $projectRoot $configuration.source_review) -Raw -Encoding utf8 | ConvertFrom-Json
if ($review.status -ne 'SOURCE_PASS' -or $review.blocking_issues -ne 0 -or $review.plan_sha256 -ne (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant()) { throw 'Parallel operations review not passed.' }
foreach ($item in $configuration.source_sha256.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath (Join-Path $projectRoot $item.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.Value) { throw "Source changed: $($item.Name)" }
}
foreach ($item in $configuration.local_input_sha256.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath (Join-Path $projectRoot $item.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.Value) { throw "Input changed: $($item.Name)" }
}
$statePath = Join-Path $projectRoot $configuration.execution_state
if (Test-Path -LiteralPath $statePath) { throw 'Adoption already attempted; inspect its state.' }
$owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($configuration.old_owner.controller_pid)"
if ($null -eq $owner -or $owner.CommandLine -ne $configuration.old_owner.command -or $owner.ParentProcessId -ne $configuration.old_owner.launcher_pid) { throw 'Original controller identity changed.' }
if ($owner.CreationDate.ToUniversalTime() -ne ([datetime]$configuration.old_owner.created_utc).ToUniversalTime()) { throw 'Original controller birth changed.' }
$queue = Get-Content -LiteralPath (Join-Path $projectRoot $configuration.old_owner.status_path) -Raw -Encoding utf8 | ConvertFrom-Json
if ($queue.state -ne 'FULL_DEV_RUNNING' -or $queue.tasks.Count -ne 2 -or $queue.tasks[-1].pid -ne $configuration.adopted_task.pid -or $queue.metrics.Count -ne 0) { throw 'Original queue advanced; do not adopt this snapshot.' }
if ($queue.next_observation_epoch -lt ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + 300)) { throw 'Original controller is too close to its next action.' }
$python = 'C:\Users\gb\AppData\Local\uv\cache\archive-v0\7oLWbeZ6v0oUodecKf4d1\Scripts\python.exe'
$beforeCode = @'
import json,sys
from pathlib import Path
sys.path.insert(0,'scripts')
from complete_pipeline import ROOT,REMOTE,remote
p=json.loads(Path(sys.argv[1]).read_bytes()); e=p['adopted_task']
code="""import csv,json,os,subprocess,time
from pathlib import Path
r=Path(%r); e=%r
n=json.loads((r/'runs/pipeline_tasks'/e['task']/'status.json').read_bytes())
assert n['state']=='RUNNING' and n['pid']==e['pid'] and n['command']==e['command']
assert n['physical_gpu']=='0' and n['started']==e['started']
wrapper=(Path('/proc')/str(e['pid'])/'cmdline').read_bytes().decode().split(chr(0))[:-1]
assert wrapper==[e['command'][0],'scripts/run_task.py','--task',e['task'],'--gpu','0','--']+e['command']
children=[int(x) for x in (Path('/proc')/str(e['pid'])/'task'/str(e['pid'])/'children').read_text().split()]
assert children==[e['child_pid']]
assert (Path('/proc')/str(e['child_pid'])/'cmdline').read_bytes().decode().split(chr(0))[:-1]==e['command']
assert int(next(x.split()[1] for x in (Path('/proc')/str(e['child_pid'])/'status').read_text().splitlines() if x.startswith('PPid:')))==e['pid']
used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],universal_newlines=True).strip()); assert used<500,used
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'; os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
torch.manual_seed(2027)
x=torch.randn(8,8,device='cuda'); y=x@x
assert torch.isfinite(y).all(); torch.cuda.synchronize()
rows=json.loads((r/'runs'/e['task']/'progress.json').read_bytes())
trace_frames=0; service=[]
for path in sorted((r/'runs'/e['task']/'traces').glob('*.csv'),key=lambda q:q.stat().st_mtime):
 if path.stat().st_size:
  with path.open(newline='') as f: header,*values=csv.reader(f)
  values=[v for v in values if len(v)==len(header)]
  trace_frames+=max([int(v[0]) for v in values]+[0])+1
  service.extend(float(v[header.index('seconds_including_decode')]) for v in values)
tail=service[-256:]; assert tail and min(tail)>0
print(json.dumps(dict(native=n,wrapper_argv=wrapper,child_pid=e['child_pid'],observed=time.time(),completed_frames=sum(x['frames'] for x in rows),completed_sequences=len(rows),trace_frames_including_init=trace_frames,recent_primary_service_fps=len(tail)/sum(tail),recent_service_frames=len(tail),gpu1_kernel='PASS',gpu1_name=torch.cuda.get_device_name())))
""" % (REMOTE,e)
snapshot=remote(code)
Path(p['before_receipt']).write_text(json.dumps(snapshot,indent=2)+'\n')
print(json.dumps(snapshot))
'@
$before = $beforeCode | & $python - $planPath
if ($LASTEXITCODE -ne 0) { throw 'GPU/task preflight failed; original controller was not stopped.' }
$owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($configuration.old_owner.controller_pid)"
if ($null -eq $owner -or $owner.CommandLine -ne $configuration.old_owner.command) { throw 'Controller changed before retirement.' }
Stop-Process -Id $owner.ProcessId -Force
Start-Sleep -Seconds 2
if ($null -ne (Get-CimInstance Win32_Process -Filter "ProcessId=$($configuration.old_owner.controller_pid)")) { throw 'Old controller is still present.' }
$exitPath = Join-Path $projectRoot $configuration.old_owner.exit_receipt
$oldExit = Get-Content -LiteralPath $exitPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($oldExit.launcher_pid -ne $configuration.old_owner.launcher_pid) { throw 'Old launcher terminal receipt mismatch.' }
$retired = [ordered]@{ state='OWNER_RETIRED_NN_CONTINUES'; controller_pid=$owner.ProcessId; launcher_pid=$oldExit.launcher_pid; actual_exit_code=$oldExit.exit_code; observed_utc=[DateTime]::UtcNow.ToString('o'); source_queue_status=$configuration.old_owner.status_path; reason='Explicit operator handoff for two-GPU full-dev execution; do not classify as an NN failure.' }
[IO.File]::WriteAllText((Join-Path $projectRoot $configuration.retirement_receipt), (($retired | ConvertTo-Json -Depth 8) + [char]10), [Text.UTF8Encoding]::new($false))
$launchCode = @'
import json,sys,time
from pathlib import Path
sys.path.insert(0,'scripts')
from complete_pipeline import ROOT,REMOTE,launch,remote
p=json.loads(Path(sys.argv[1]).read_bytes()); e=p['adopted_task']
native=remote("""import json
from pathlib import Path
r=Path(%r); e=%r
n=json.loads((r/'runs/pipeline_tasks'/e['task']/'status.json').read_bytes())
assert n['state']=='RUNNING' and n['pid']==e['pid'] and n['command']==e['command']
assert n['physical_gpu']=='0' and n['started']==e['started']
wrapper=(Path('/proc')/str(e['pid'])/'cmdline').read_bytes().decode().split(chr(0))[:-1]
assert wrapper==[e['command'][0],'scripts/run_task.py','--task',e['task'],'--gpu','0','--']+e['command']
children=[int(x) for x in (Path('/proc')/str(e['pid'])/'task'/str(e['pid'])/'children').read_text().split()]
assert children==[e['child_pid']]
assert (Path('/proc')/str(e['child_pid'])/'cmdline').read_bytes().decode().split(chr(0))[:-1]==e['command']
assert int(next(x.split()[1] for x in (Path('/proc')/str(e['child_pid'])/'status').read_text().splitlines() if x.startswith('PPid:')))==e['pid']
print(json.dumps(n))
""" % (REMOTE,e))
assert native['command']==e['command'] and native['started']==e['started']
evaluation=json.loads((ROOT/p['evaluation_plan']).read_bytes()); entry=evaluation['runs']['0.05']
command=[REMOTE+'/scripts/run_rankstop_dev.py','--plan',p['evaluation_plan'],'--task',entry['task'],'--run',entry['run'],'--cost-weight','0.05']
new=launch(entry['task'],1,command)
state=dict(state='PARALLEL_FULL_DEV_LAUNCHED',owner='root goal operator; original Windows controller retired',started=time.time(),adopted_native=native,new_task=new,pending_cost=0.15,next_observation_epoch=time.time()+p['first_parallel_observation_seconds'],evaluation_plan=p['evaluation_plan'],experiment_unchanged=True,timing_scope='Cost0 includes solo then parallel execution. Full FPS must retain this load disclosure; no paired speedup claim.')
(ROOT/p['execution_state']).write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state))
'@
$launchCode | & $python - $planPath
if ($LASTEXITCODE -ne 0) { throw 'Parallel launch did not complete; adopted GPU0 NN remains detached. Inspect receipts, do not retry blindly.' }
