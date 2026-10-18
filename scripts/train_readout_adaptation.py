"""LasHeR readout adaptation: freeze backbone/policy, train only decode+box_head."""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_rgbt.data import LasHeRPairs
from dual_rgbt.losses import executor_loss, localization_loss
from dual_rgbt.model import DualRGBT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--checkpoint', default='/home/gaob/Dual-RGBT/runs/stage_b_refreshed_20261013/latest.pth')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--samples', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--lr', type=float, default=1e-5)
    args = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.json').write_text(json.dumps(vars(args), indent=2))
    model = DualRGBT('/home/gaob/CADTrack_ep0014.pth.tar').cuda()
    checkpoint_state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint_state['net'], strict=True)
    del checkpoint_state
    for module in (model.backbone, model.preview, model.quality, model.gain):
        module.requires_grad_(False)
    model.decode.requires_grad_(True)
    model.box_head.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(p.requires_grad for p in list(model.decode.parameters()) + list(model.box_head.parameters()))
    assert not any(p.requires_grad for p in list(model.backbone.parameters()) + list(model.preview.parameters()) + list(model.quality.parameters()) + list(model.gain.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    fields=['epoch','step','modality','action','loss','iou','seconds','primary_tokens','auxiliary_tokens']
    with (run/'steps.csv').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=fields); writer.writeheader()
        step=0; best=float('inf')
        for epoch in range(1,args.epochs+1):
            ds=LasHeRPairs(ROOT/'data/manifest.json','train',args.samples,args.seed+(epoch-1)*args.samples)
            loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True,drop_last=True)
            model.train(); model.preview.eval(); model.quality.eval(); model.gain.eval()
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d): module.eval()
            for batch in loader:
                start=time.perf_counter(); templates,search,target=[batch[k].cuda(non_blocking=True) for k in ('templates','search','target')]
                modality,action=step%2,(step//2)%5; output=model.forward_action(templates,search,modality,action); loss=executor_loss(output,target)
                assert torch.isfinite(loss); optimizer.zero_grad(set_to_none=True); loss.backward(); norm=torch.nn.utils.clip_grad_norm_(trainable,.1); assert torch.isfinite(norm); optimizer.step()
                _,iou=localization_loss(output['box'],target)
                writer.writerow(dict(epoch=epoch,step=step,modality=modality,action=action,loss=float(loss),iou=float(iou.mean()),seconds=time.perf_counter()-start,primary_tokens=output['primary']['primary_tokens'],auxiliary_tokens=output['auxiliary_tokens']))
                if step%50==0: fh.flush(); print(json.dumps({'epoch':epoch,'step':step,'loss':float(loss),'iou':float(iou.mean())}),flush=True)
                step+=1
            model.eval(); dev=LasHeRPairs(ROOT/'data/manifest.json','dev',100,args.seed); summaries=[]
            with torch.no_grad():
                for modality in range(2):
                    for action in range(5):
                        losses=[]; ious=[]
                        for batch in DataLoader(dev,batch_size=args.batch_size,num_workers=args.workers):
                            templates,search,target=[batch[k].cuda() for k in ('templates','search','target')]
                            out=model.forward_action(templates,search,modality,action); values,overlap=localization_loss(out['box'],target); losses+=values.cpu().tolist(); ious+=overlap.cpu().tolist()
                        summaries.append({'modality':modality,'action':action,'localization_loss':float(np.mean(losses)),'sample_iou':float(np.mean(ious)),'samples':len(losses)})
            (run/f'dev_epoch{epoch:02d}.json').write_text(json.dumps(summaries,indent=2))
            state={'net':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'step':step,'config':vars(args),'stage':'readout_adaptation'}; torch.save(state,run/'latest.pth')
            dev_loss=float(np.mean([x['localization_loss'] for x in summaries]));
            if dev_loss<best: best=dev_loss; torch.save({'net':model.state_dict(),'epoch':epoch,'step':step,'config':vars(args),'stage':'readout_adaptation'},run/'best.pth'); (run/'best.json').write_text(json.dumps({'epoch':epoch,'step':step,'dev_loss':dev_loss},indent=2))
            print('EPOCH_COMPLETE',epoch,json.dumps(summaries),flush=True)
    (run/'COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','epochs':args.epochs,'steps':step,'trainable_parameter_count':sum(x.numel() for x in trainable),'frozen_backbone':True,'frozen_policy_heads':True,'scope':'LasHeR readout adaptation only; architecture and module roles unchanged.'},indent=2))

if __name__=='__main__': main()
