import json,os,sys
from pathlib import Path
task,exitcode=sys.argv[1],int(sys.argv[2])
r=json.loads((Path('runs/pipeline_tasks')/task/'status.json').read_text())
assert r['state']=='RUNNING' and r['pid']==os.getppid()
assert r['command']==[sys.executable]+sys.argv
assert r['physical_gpu']==os.environ['CUDA_VISIBLE_DEVICES']
assert os.environ['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
assert 'torch' not in sys.modules
print(json.dumps(dict(gpu=os.environ['CUDA_VISIBLE_DEVICES'],pid=os.getpid(),ppid=os.getppid(),torch_imported=False)))
sys.exit(exitcode)
