"""Run the unchanged joint trainer after releasing unused host heap pages once."""
import ctypes
import gc
import importlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    plan=json.loads((ROOT/sys.argv[sys.argv.index('--plan')+1]).read_bytes())
    module_name={'joint_preview_relative_quality':'train_candidate_joint_quality_trajectory',
        'signed_gain_and_same_state_difference_regression':'train_candidate_joint_gainpair_trajectory'}[plan['repair']['kind']]
    module=importlib.import_module(module_name)
    original=module.trajectory_stage
    def trimmed(model,optimizer,plan,config,run,before,sequences,sanity):
        def memory():
            return {v.split(':',1)[0]:v.split(':',1)[1].strip() for v in Path('/proc/self/smaps_rollup').read_text().splitlines()[1:]}
        start=memory();gc.collect()
        libc=ctypes.CDLL('libc.so.6');libc.malloc_trim.argtypes=[ctypes.c_size_t];libc.malloc_trim.restype=ctypes.c_int
        result=libc.malloc_trim(0);end=memory()
        (run/'host_memory_startup.json').write_text(json.dumps(dict(before=start,after=end,malloc_trim_return=result,
            scope='One post-load host-heap release; no model, tensor values, optimizer, data, RNG or trajectory changes.'),indent=2))
        return original(model,optimizer,plan,config,run,before,sequences,sanity)
    module.trajectory_stage=trimmed
    module.main()
if __name__=='__main__':main()
