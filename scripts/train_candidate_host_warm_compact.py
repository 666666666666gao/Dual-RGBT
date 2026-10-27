"""Release unused host heap once after four real optimizer updates warm CUDA kernels."""
import ctypes
import gc
import importlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    plan=json.loads((ROOT/sys.argv[sys.argv.index('--plan')+1]).read_bytes())
    name={'joint_preview_relative_quality':'train_candidate_joint_quality_trajectory','signed_gain_and_same_state_difference_regression':'train_candidate_joint_gainpair_trajectory'}[plan['repair']['kind']]
    module=importlib.import_module(name);original=module.trajectory_stage;original_step=module.step_optimizer
    def watched(model,optimizer,plan,config,run,before,sequences,sanity):
        count=0
        def step(loss,model,optimizer,prefixes,sanity):
            nonlocal count
            result=original_step(loss,model,optimizer,prefixes,sanity);count+=1
            if count==4:
                def memory():return {v.split(':',1)[0]:v.split(':',1)[1].strip() for v in Path('/proc/self/smaps_rollup').read_text().splitlines()[1:]}
                start=memory();gc.collect();libc=ctypes.CDLL('libc.so.6');libc.malloc_trim.argtypes=[ctypes.c_size_t];libc.malloc_trim.restype=ctypes.c_int
                trimmed=libc.malloc_trim(0);end=memory()
                (run/'host_memory_warmup.json').write_text(json.dumps(dict(before=start,after=end,malloc_trim_return=trimmed,optimizer_updates=count,scope='One release of unused CPU heap after four exact original updates; no tensor/optimizer/RNG/data/model changes.'),indent=2))
            return result
        module.step_optimizer=step
        return original(model,optimizer,plan,config,run,before,sequences,sanity)
    module.trajectory_stage=watched;module.main()
if __name__=='__main__':main()
