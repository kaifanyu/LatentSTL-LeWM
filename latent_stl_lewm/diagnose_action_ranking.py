"""Diagnose action rankings using held-out same-start visual trajectories."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F
from .cem_models import CEMModel
from .train import file_hash, write_json
from .world_model import load_checkpoint


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    base,_=load_checkpoint(args.original,args.device)
    data={};banks=defaultdict(list)
    with h5py.File(args.dataset,'r') as h:
        for key,ep in h['trajectories'].items():
            if str(ep.attrs['split']) not in ('train','val'):continue
            rgb=torch.as_tensor(ep['rgb'][:],device=args.device).permute(0,3,1,2)
            with torch.autocast('cuda',dtype=torch.float16):
                z=torch.cat([base.encode(x).float() for x in rgb.split(128)])
            data[key]={'z':z.cpu(),'a':torch.as_tensor(ep['action'][:]),'split':str(ep.attrs['split']),
                       'bank':str(ep.attrs['bank_id']),'policy':str(ep.attrs['policy'])}
            if data[key]['split']=='val':banks[data[key]['bank']].append(key)
        with torch.autocast('cuda',dtype=torch.float16):
            goal=base.encode(torch.as_tensor(h['references/goal_rgb'][:],device=args.device).permute(2,0,1)[None]).float()[0]
    torch.save({'episodes':data,'encoder_checkpoint_sha256':file_hash(args.original),
                'dataset_sha256':file_hash(args.dataset),'goal':goal.cpu()},out/'encoded_train_val.pt')
    rows=[]
    for name,path in [('original',args.original),('multistep',args.multistep)]:
        model,_=load_checkpoint(path,args.device);adapter=CEMModel(model,kind='local',action_repeat=1)
        for bank,ids in banks.items():
            arc=next(key for key in ids if data[key]['policy']=='safe_arc')
            actions=torch.stack([data[key]['a'][:32] for key in ids]).to(args.device)
            z=torch.stack([data[key]['z'] for key in ids]).to(args.device)
            with torch.autocast('cuda',dtype=torch.float16):pred=adapter.rollout(z[:,0],actions)
            for horizon in (1,4,8,16,32):
                actual=z[:,horizon];estimated=pred[:,horizon]
                real_delta=actual-z[:,0];pred_delta=estimated-z[:,0]
                row={'model':name,'bank':bank,'horizon':horizon,'ids':ids,
                     'cosine_error':float((1-F.cosine_similarity(actual,estimated)).mean()),
                     'persistence_cosine_error':float((1-F.cosine_similarity(actual,z[:,0])).mean()),
                     'delta_direction_cosine':float(F.cosine_similarity(real_delta,pred_delta).mean()),
                     'delta_norm_ratio':float(pred_delta.norm(dim=-1).mean()/real_delta.norm(dim=-1).mean().clamp_min(1e-8))}
                for target_name,target in [('goal',goal),('arc_waypoint',data[arc]['z'][min(horizon*2,96)].to(args.device))]:
                    truth=1-F.cosine_similarity(actual,target[None]);score=1-F.cosine_similarity(estimated,target[None])
                    row[target_name]={'spearman':float(spearmanr(truth.cpu(),score.cpu()).statistic),
                                      'best_actual':data[ids[int(truth.argmin())]]['policy'],
                                      'best_predicted':data[ids[int(score.argmin())]]['policy'],
                                      'regret':float(truth[score.argmin()]-truth.min()),
                                      'actual_distances':truth.cpu().tolist(),'predicted_distances':score.cpu().tolist()}
                rows.append(row)
        del model,adapter
    summary=[]
    for name in ('original','multistep'):
        for horizon in (1,4,8,16,32):
            subset=[r for r in rows if r['model']==name and r['horizon']==horizon]
            record={'model':name,'horizon':horizon}
            for key in ('cosine_error','persistence_cosine_error','delta_direction_cosine','delta_norm_ratio'):
                record[key]=float(np.mean([r[key] for r in subset]))
            for target in ('goal','arc_waypoint'):
                record[target+'_rank_spearman']=float(np.mean([r[target]['spearman'] for r in subset]))
                record[target+'_correct_top1']=sum(r[target]['best_actual']==r[target]['best_predicted'] for r in subset)
                record[target+'_regret']=float(np.mean([r[target]['regret'] for r in subset]))
            summary.append(record)
    write_json(out/'report.json',{'dataset':args.dataset,'evaluation_split':'val','banks':len(banks),
                                 'rows':rows,'summary':summary})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('original','multistep','dataset','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda')
    run(p.parse_args())
