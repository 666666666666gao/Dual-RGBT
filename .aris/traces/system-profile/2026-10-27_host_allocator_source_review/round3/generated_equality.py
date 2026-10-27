import json,torch
from pathlib import Path
r=Path('/data2/gb/Dual-RGBT');cases=[{'name': 'default', 'plan': 'reports/host_allocator_default_plan_20261027.json', 'plan_sha256': '22d196dd73e1fc929467c0cf453e7c4ed84c65a38ccf199a9eb2b5251071ffd8', 'sample_output': 'reports/HOST_ALLOCATOR_DEFAULT_SAMPLES_20261027.json'}, {'name': 'compact', 'plan': 'reports/host_allocator_compact_plan_20261027.json', 'plan_sha256': '08c5a5ec9d32c812fb966674d0ba60f11ac14f5bf99cb5555c9a95e37449c2cb', 'sample_output': 'reports/HOST_ALLOCATOR_COMPACT_SAMPLES_20261027.json'}];torch.set_num_threads(1)
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
