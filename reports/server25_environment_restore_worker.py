import json,subprocess,time,sys,os
from pathlib import Path
r=Path('/home/gaob/Dual-RGBT');source=Path('/data2/gaob/Dual-RGBT_archive/retired_server25_project_20261026/conda-envs');out=r/'reports/SERVER25_ENVIRONMENT_RESTORE_NATIVE_20261027.json'
state=dict(state='RUNNING',pid=os.getpid(),started=time.time(),source=str(source),destination=str(r/'conda-envs'));out.write_text(json.dumps(state,indent=2)+'\n')
a=subprocess.run(['cp','-a','--reflink=auto','--',str(source),str(r/'conda-envs')]);state.update(exit_code=a.returncode,completed=time.time(),state='COMPLETE' if a.returncode==0 else 'FAILED');out.write_text(json.dumps(state,indent=2)+'\n');sys.exit(a.returncode)
