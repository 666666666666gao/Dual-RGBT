"""Expand launcher strings with inert remote/time stubs; never execute them."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('C:/Users/gb/projects/Dual-RGBT')
OUT = ROOT / '.aris/traces/experiment-audit/2026-10-27_vram18_source_review'
index = json.loads((ROOT/'reports/VRAM18_TRAINING_PLAN_INDEX_20261027.json').read_bytes())
tree = ast.parse((ROOT/'tools/launch_vram18_training_wave.py').read_text())
launch_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'launch')
remote_root = '/data2/gb/Dual-RGBT'
python = remote_root + '/conda-envs/dual_rgbt/bin/python'
recorded = []
def capture(code):
    ast.parse(code)
    recorded.append(code)
    return {'source_only': True}
namespace = dict(ROOT=ROOT,Path=Path,json=json,REMOTE=remote_root,PYTHON=python,
                 owned=index['original_owners'],temporary=[],compact_peak=1,
                 remote=capture,time=SimpleNamespace(sleep=lambda _:None))
exec(compile(ast.Module(body=[launch_node],type_ignores=[]),'isolated_launcher_builder','exec'),namespace)
assignments = [n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets)]
witnesses = [n for n in assignments if n.lineno in (94,112)]
programs=[]
for row in index['plans']:
    plan = json.loads((ROOT/row['path']).read_bytes())
    case=dict(plan=row['path'],plan_sha256=row['sha256'],gpu=row['gpu'],runs=plan['runs'],tasks=plan['tasks'])
    namespace['args']=SimpleNamespace(phase=row['phase'])
    for sanity in [True,False]:
        stage='sanity' if sanity else 'formal'
        namespace['launch'](case,sanity,None if sanity else '0'*64)
        code=recorded[-1]
        command_node=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen')
        command=ast.literal_eval(command_node.args[0])
        trainer='scripts/train_candidate_'+row['objective']+'_trajectory.py'
        expected=[python,plan['launch_wrapper'],'--task',plan['tasks'][stage],'--gpu',str(row['gpu']),'--',python,remote_root+'/'+trainer,'--plan',row['path'],'--run',plan['runs'][stage]]
        expected += ['--sanity'] if sanity else ['--sanity-complete-sha256','0'*64]
        assert command==expected
        assert command[7:]==expected[7:]
        programs.append(dict(plan=row['path'],kind=stage+'_launch',syntax='PASS',command=command,
            source_sha256=hashlib.sha256(code.encode()).hexdigest(),source=code))
        if not sanity:case['formal']=dict(task=plan['tasks']['formal'],pid=1,gpu=row['gpu'],command=command,launched=0)
    for node in witnesses:
        code=eval(compile(ast.Expression(node.value),'isolated_witness_string','eval'),dict(REMOTE=remote_root,c=case))
        ast.parse(code)
        programs.append(dict(plan=row['path'],kind='sanity_admission' if node.lineno==94 else 'formal_startup',syntax='PASS',
            source_sha256=hashlib.sha256(code.encode()).hexdigest(),source=code))

# State-machine counterexample uses only synthetic dictionaries, no runtime observations.
# The native wrapper persists RUNNING before subprocess.run creates the training child.
starting=dict(native_state='RUNNING',wrapper_child_pid=101,cuda_context_created=False)
expected={starting['wrapper_child_pid']} if starting['native_state']=='RUNNING' else set()
actual=set()
counterexample=dict(label='SOURCE_STATE_MODEL_ONLY_NOT_A_RUNTIME_RESULT',
    admissible_state=starting,expected_gpu_pids=sorted(expected),actual_gpu_pids=sorted(actual),
    ownership_assertion_passes=actual==expected,
    reason='RUNNING does not imply CUDA context readiness; trainer verifies source/input hashes before its first .cuda(). Fixed two-second sleep is not a lifecycle acknowledgment.')
(OUT/'generated_remote_programs.json').write_text(json.dumps(dict(programs=programs,state_counterexample=counterexample),indent=2)+'\n')
print(json.dumps(dict(status='SOURCE_SYNTAX_ARGV_PASS',programs=len(programs),launch_argv_count=48,remote_execution=False,counterexample=counterexample),indent=2))
