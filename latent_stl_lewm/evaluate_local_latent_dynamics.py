"""Reproducible validation-only comparison of local ridge transition settings."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F
from .local_latent_dynamics import LocalLatentDynamics
from .train import file_hash, write_json
from .world_model import load_checkpoint


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(4)
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    cache=torch.load(args.encoded,weights_only=True,map_location='cpu')
    episodes={key:value for key,value in cache['episodes'].items() if value['split']=='val'}
    banks=sorted({value['bank'] for value in episodes.values()})
    identifiers=[key for bank in banks for key,value in episodes.items() if value['bank']==bank]
    z=torch.stack([episodes[key]['z'] for key in identifiers]).to(args.device)
    actions=torch.stack([episodes[key]['a'][:16] for key in identifiers]).to(args.device)
    goal=cache['goal'].to(args.device)
    base,_=load_checkpoint(args.checkpoint,args.device)
    artifact=torch.load(args.artifact,weights_only=True,map_location='cpu')
    results=[]
    for neighbors,ridge in itertools.product(args.neighbors,args.ridge):
        model=LocalLatentDynamics(base,artifact,neighbors=neighbors,ridge=ridge,device=args.device)
        pred=model.rollout(z[:,0],actions)
        metrics=[]
        for horizon in (1,4,8,16):
            actual,estimated=z[:,horizon],pred[:,horizon]
            row={'horizon':horizon,'cosine_error':float((1-F.cosine_similarity(actual,estimated)).mean()),
                 'goal_ranks':[],'waypoint_ranks':[]}
            for bank in banks:
                indices=[i for i,key in enumerate(identifiers) if episodes[key]['bank']==bank]
                arc=next(key for key in identifiers if episodes[key]['bank']==bank and episodes[key]['policy']=='safe_arc')
                waypoint=episodes[arc]['z'][min(horizon*2,96)].to(args.device)
                for name,target in [('goal',goal),('waypoint',waypoint)]:
                    truth=(1-F.cosine_similarity(actual[indices],target[None])).cpu().numpy()
                    score=(1-F.cosine_similarity(estimated[indices],target[None])).cpu().numpy()
                    row[name+'_ranks'].append(float(spearmanr(truth,score).statistic))
            for key in ('goal_ranks','waypoint_ranks'):row[key]=float(np.mean(row[key]))
            metrics.append(row)
        result={'neighbors':neighbors,'ridge':ridge,'metrics':metrics}
        results.append(result);print(json.dumps(result),flush=True)
    write_json(output,{'args':vars(args),'results':results,'evaluation_split':'val',
                       'artifact_sha256':file_hash(args.artifact),'encoded_sha256':file_hash(args.encoded),
                       'source_sha256':file_hash(__file__)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('encoded','checkpoint','artifact','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--neighbors',nargs='+',type=int,default=[64,128,256])
    parser.add_argument('--ridge',nargs='+',type=float,default=[.01,.1,1.])
    parser.add_argument('--device',default='cuda')
    evaluate(parser.parse_args())
