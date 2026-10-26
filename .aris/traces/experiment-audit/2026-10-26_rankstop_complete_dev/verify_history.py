import ast
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

root=Path('C:/Users/gb/projects/Dual-RGBT')
out=Path(__file__).parent
remote=json.loads((out/'remote_verification.json').read_text(encoding='utf-8-sig'))
records=[]
for name,expected in remote['checkpoint_config']['source_sha256'].items():
    current=(root/name).read_bytes()
    if hashlib.sha256(current).hexdigest()==expected:continue
    commits=subprocess.check_output(['git','log','--all','--format=%H','--',name],cwd=root).decode().splitlines()
    found=None
    for commit in commits:
        p=subprocess.run(['git','show',commit+':'+name],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if p.returncode:continue
        raw=p.stdout
        forms={'raw':raw,'LF':raw.replace(b'\r\n',b'\n'),'CRLF':raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')}
        for form,data in forms.items():
            if hashlib.sha256(data).hexdigest()==expected:
                found=(commit,form,data);break
        if found:break
    assert found,name
    commit,form,data=found
    snapshot=out/('historical_'+name.replace('/','_'))
    snapshot.write_bytes(data)
    old=data.decode();now=current.decode()
    diff=''.join(difflib.unified_diff(old.splitlines(True),now.splitlines(True),fromfile=commit+':'+name,tofile='current:'+name))
    (out/(name.replace('/','_')+'.diff')).write_text(diff,encoding='utf-8')
    funcs=[]
    oldast=ast.parse(old);newast=ast.parse(now)
    for n in ast.walk(oldast):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            matches=[x for x in ast.walk(newast) if isinstance(x,type(n)) and x.name==n.name]
            if len(matches)==1 and ast.dump(n,include_attributes=False)==ast.dump(matches[0],include_attributes=False):funcs.append(n.name)
    records.append(dict(path=name,historical_sha256=expected,current_sha256=hashlib.sha256(current).hexdigest(),commit=commit,newline_transform=form,snapshot=snapshot.name,exact_byte_hash_match=True,ast_identical_functions=funcs,diff=diff))
(out/'source_history_verified.json').write_text(json.dumps(dict(status='PASS',records=records),indent=2)+'\n')
for r in records:print(json.dumps(r),flush=True)
