"""Read-only audit of predicted versus observed executed-prefix visual motion."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train import file_hash, write_json


def audit(args):
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    goal=torch.load(args.encoded,weights_only=True,map_location='cpu')['goal']
    results=[]
    for directory in args.runs:
        directory=Path(directory);rows=[]
        for path in sorted(directory.glob('scene_*/prediction_*.npz')):
            with np.load(path) as data:
                predicted=data['latent'][0];actual=data['executed_prefix_actual_latent']
                z0=torch.from_numpy(predicted[0]);estimate=torch.from_numpy(predicted[len(actual)])
                observed=torch.from_numpy(actual[-1])
            rows.append({'scene':path.parent.name,'cycle':path.stem,
                         'predicted_goal_gain':float(F.cosine_similarity(estimate,goal,dim=0)-F.cosine_similarity(z0,goal,dim=0)),
                         'actual_goal_gain':float(F.cosine_similarity(observed,goal,dim=0)-F.cosine_similarity(z0,goal,dim=0)),
                         'direction_cosine':float(F.cosine_similarity(estimate-z0,observed-z0,dim=0)),
                         'predicted_delta_norm':float((estimate-z0).norm()),'actual_delta_norm':float((observed-z0).norm())})
        if not rows:continue
        keys=('predicted_goal_gain','actual_goal_gain','direction_cosine','predicted_delta_norm','actual_delta_norm')
        arrays={key:np.asarray([row[key] for row in rows]) for key in keys}
        results.append({'run':str(directory.resolve()),'cycles':len(rows),
                        'mean':{key:float(value.mean()) for key,value in arrays.items()},
                        'predicted_positive_actual_negative':int(((arrays['predicted_goal_gain']>0)&(arrays['actual_goal_gain']<0)).sum()),
                        'gain_correlation':float(np.corrcoef(arrays['predicted_goal_gain'],arrays['actual_goal_gain'])[0,1]),
                        'config_sha256':file_hash(directory/'config.json'),'rows':rows})
    write_json(output,{'source_sha256':file_hash(__file__),'encoded_sha256':file_hash(args.encoded),
                       'metric':'Cosine similarity to the task goal; matched actually executed prefix only',
                       'interpretation':'Visual prediction accuracy, not physical success or contact safety',
                       'results':results})
    print(json.dumps([{key:value for key,value in result.items() if key!='rows'} for result in results],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoded',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--runs',nargs='+',required=True)
    audit(parser.parse_args())
