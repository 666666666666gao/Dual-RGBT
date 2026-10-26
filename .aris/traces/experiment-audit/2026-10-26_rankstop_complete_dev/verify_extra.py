import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
root=Path('C:/Users/gb/projects/Dual-RGBT');out=Path(__file__).parent
hashes={}
def read(p):
    p=root/p;data=p.read_bytes();hashes[p.relative_to(root).as_posix()]=hashlib.sha256(data).hexdigest();return json.loads(data)
plan=read('reports/rankstop_dev_plan_server26_20261026.json')
plan_sha=hashes['reports/rankstop_dev_plan_server26_20261026.json']
review=read(plan['source_review'])
assert review['status']=='SOURCE_PASS' and review['blocking_issues']==0
assert review['plan_sha256']==plan_sha and review['source_sha256']==plan['source_sha256']
sm=plan['runs']['smoke'];sr=read('runs/'+sm['run']+'/SMOKE.json')
sc=read('runs/'+sm['run']+'/config.json');sn=read('runs/pipeline_tasks/'+sm['task']+'/status.json')
assert sr['frames']==11 and sr['sequences']==1 and sr['smoke_only']
assert sr['plan_sha256']==sc['plan_sha256']==plan_sha
assert sr['checkpoint_sha256']==sc['checkpoint_sha256']==plan['checkpoint_sha256']
assert sn['state']=='COMPLETE' and sn['exit_code']==0 and sn['command']==sc['native_command']
assert sr['reload']['status']=='PASS' and sr['reload']['strict_reload_box_route_bit_equal'] and sr['reload']['strict_reload_templates_history_bit_equal']
training_audit=read(plan['training_audit'])
assert training_audit['actual_process_exit_code'] is None
native=[]
for cost in [0.,.05,.15]:
    e=plan['runs'][str(cost)];n=read('runs/pipeline_tasks/'+e['task']+'/status.json')
    expected=['/data2/gb/Dual-RGBT/conda-envs/dual_rgbt/bin/python','/data2/gb/Dual-RGBT/scripts/run_rankstop_dev.py','--plan','reports/rankstop_dev_plan_server26_20261026.json','--task',e['task'],'--run',e['run'],'--cost-weight',str(cost)]
    assert n['command']==expected
    assert n['started']>sn['completed'] and n['started']>datetime.fromisoformat(review['generated_at']).timestamp()
    for f in ['config.json','COMPLETE.json']:
        c=read('runs/'+e['run']+'/'+f)
        assert c['training_provenance_warnings']==training_audit['warnings']
    m=read(e['metrics']);assert m['training_provenance_warnings']==training_audit['warnings']
    native.append(n)
overlap=max(0,min(native[0]['completed'],native[1]['completed'])-max(native[0]['started'],native[1]['started']))
assert overlap>0
launch=read('reports/RANKSTOP_COST15_LAUNCH_20261026.json');startup=read('reports/RANKSTOP_COST15_STARTUP_20261026.json')
assert launch['pid']==startup['native']['pid']==native[2]['pid']
assert launch['command'][6:]==['--']+native[2]['command']
assert launch['opencv_env_threads']==startup['opencv_env_threads']==1
assert startup['native']['command']==native[2]['command']
u=read('reports/RANKSTOP_UTILITY_TARGET_DIAGNOSTIC_20261026.json')
g=np.float32(.02);p=np.float32(.05);c=np.float32(.05)*np.float32(193/385)
toy=dict(raw_gain=float(g),query_cost=float(c),calibrated_utility=float(g-c),isolated_old_optimum_prediction=float(p),isolated_predicted_utility=float(p-c),old_loss_at_calibrated=float(.25*(.05-float(g))),old_loss_at_isolated_optimum=float(.5*(float(p)-float(g))**2))
assert toy==u['toy']
grid=np.linspace(-.1,.1,20001);loss=.5*(grid-.02)**2+.25*np.maximum(.05-grid,0)
assert abs(grid[np.argmin(loss)]-.05)<1e-12
reference=read('reports/candidate_complete_adopted_20261016/e1_dev_cost0p15.json')
rp=root/'reports/candidate_complete_adopted_20261016/e1_dev_cost0p15.npz'
assert rp.exists()
hashes[rp.relative_to(root).as_posix()]=hashlib.sha256(rp.read_bytes()).hexdigest()
curves=np.load(rp,allow_pickle=False)
for k in ['PR','NPR','SR']:
    value=100*float(curves[k].mean() if k=='SR' else curves[k][:,20].mean())
    assert value==reference[k]==plan['selection']['reference_dev'][k]
selection=read('reports/E1_THREE_DEV_SELECTION_20261017.json')
selected=next(r for r in selection['rows'] if r['cost_weight']==.15)
assert selected['source_sha256']==hashes['reports/candidate_complete_adopted_20261016/e1_dev_cost0p15.json']
report=dict(status='PASS',smoke_native_exit_code=0,smoke_receipt_frames=11,smoke_receipt_reload='PASS_NOT_NEURALLY_REPLAYED',native_commands_registered_exact=True,source_review_before_all_dev=True,native_receipt_overlap_seconds=overlap,cost15_env_thread_request=1,thread_runtime_not_stamped_in_dev_receipts=True,training_exit_code=None,training_warnings_preserved_in_all_9_artifacts=True,reference_npz_values_exact=True,toy_exact=True,toy_numerical_optimum=float(grid[np.argmin(loss)]),fixed_margin_conflicts=sum(r['exact_gain_violates_fixed_margin'] for r in u['label_groups']),isolated_wrong_call_cost0p05=sum(r['idealized_wrong_call_cost_0.05'] for r in u['label_groups']),hashes=hashes)
(out/'extra_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
