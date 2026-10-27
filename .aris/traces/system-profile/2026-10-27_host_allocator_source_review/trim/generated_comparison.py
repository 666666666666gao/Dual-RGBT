import json,torch
from pathlib import Path
r=Path('/data2/gb/Dual-RGBT');cases=[{'name': 'default', 'plan': 'reports/host_allocator_r4_default_plan_20261027.json', 'plan_sha256': 'f6ed9381a8141c08c5aa41371c9e9321da4594e5d6891a7114025936023c0d03', 'sample_output': 'reports/HOST_ALLOCATOR_R4_DEFAULT_SAMPLES_20261027.json', 'kind': 'completed_reference', 'reference_sample': 'reports/HOST_ALLOCATOR_R4_DEFAULT_SAMPLES_20261027.json', 'reference_sample_sha256': '9c4bee40860d6cb80ac4ccbfa0c16c8cc9c977143596cd94c8872d21e903d0d8'}, {'name': 'compact', 'kind': 'new_admission', 'plan': 'reports/host_allocator_trim_case_plan_20261027.json', 'plan_sha256': 'afbd6498e8b65a46a9b8a69fcc40f49f2ee77432d95e9a5682b88935cbdb9b3f', 'sample_output': 'reports/HOST_ALLOCATOR_TRIM_COMPACT_SAMPLES_20261027.json'}];torch.set_num_threads(1)
paths=[r/'runs'/json.loads((r/c['plan']).read_bytes())['runs']['sanity'] for c in cases]
a,b=[torch.load(p/'latest.pth',map_location='cpu') for p in paths]
assert tuple(a['net'])==tuple(b['net']) and len(a['net'])==265
assert all(torch.equal(a['net'][n],b['net'][n]) for n in a['net'])
errors=[]
def equal(a,b):
 if isinstance(a,torch.Tensor):
  assert a.shape==b.shape and a.dtype==b.dtype
  errors.append(dict(bit_equal=torch.equal(a,b),max_abs=float((a-b).abs().max())))
  return torch.isfinite(a).all().item() and torch.isfinite(b).all().item() and torch.allclose(a,b,atol=1e-8,rtol=1e-6)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
assert equal(a['optimizer'],b['optimizer'])
for name in ['trajectory_steps.csv','query_labels.csv']:assert (paths[0]/name).read_bytes()==(paths[1]/name).read_bytes()
print(json.dumps(dict(state_tensors_bit_equal=265,optimizer_bit_equal=all(v['bit_equal'] for v in errors),optimizer_tolerance_pass=True,optimizer_atol=1e-8,optimizer_rtol=1e-6,optimizer_max_abs=max(v['max_abs'] for v in errors),trajectory_csv_bit_equal=True,query_labels_bit_equal=True,scope='Four actual GPU updates/20trackingframes only; not full training equivalence or accuracy.')))
