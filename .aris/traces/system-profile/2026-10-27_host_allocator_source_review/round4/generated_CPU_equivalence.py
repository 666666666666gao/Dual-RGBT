import json,torch
from pathlib import Path
r=Path('/data2/gb/Dual-RGBT');cases=[{'name': 'default', 'plan': 'reports/host_allocator_r4_default_plan_20261027.json', 'plan_sha256': 'f6ed9381a8141c08c5aa41371c9e9321da4594e5d6891a7114025936023c0d03', 'sample_output': 'reports/HOST_ALLOCATOR_R4_DEFAULT_SAMPLES_20261027.json'}, {'name': 'compact', 'plan': 'reports/host_allocator_r4_compact_plan_20261027.json', 'plan_sha256': 'c3d46362069ffab78dc725f63e1c35679a3f1dc81e29c2154ad6db41c0269f3b', 'sample_output': 'reports/HOST_ALLOCATOR_R4_COMPACT_SAMPLES_20261027.json'}];torch.set_num_threads(1)
paths=[r/'runs'/json.loads((r/c['plan']).read_bytes())['runs']['sanity'] for c in cases]
a,b=[torch.load(p/'latest.pth',map_location='cpu') for p in paths]
assert tuple(a['net'])==tuple(b['net']) and len(a['net'])==265
assert all(torch.equal(a['net'][n],b['net'][n]) for n in a['net'])
def equal(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
assert equal(a['optimizer'],b['optimizer'])
for name in ['trajectory_steps.csv','query_labels.csv']:assert (paths[0]/name).read_bytes()==(paths[1]/name).read_bytes()
print(json.dumps(dict(state_tensors_bit_equal=265,optimizer_bit_equal=True,trajectory_csv_bit_equal=True,query_labels_bit_equal=True,scope='Four actual GPU updates/20trackingframes only; not full training equivalence or accuracy.')))
