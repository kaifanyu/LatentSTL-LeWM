"""Frozen-image goal-conditioned inverse dynamics fitted to training actions."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from .train import file_hash, write_json


class GoalInverseDynamics(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=256):
        super().__init__()
        self.config={'latent_dim':latent_dim,'hidden_dim':hidden_dim}
        self.register_buffer('latent_mean',torch.zeros(latent_dim))
        self.register_buffer('latent_std',torch.ones(latent_dim))
        self.network=nn.Sequential(nn.Linear(latent_dim*2+1,hidden_dim),nn.SiLU(),
                                    nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),
                                    nn.Linear(hidden_dim,3),nn.Tanh())

    def forward(self,current,target,horizon):
        horizon=torch.as_tensor(horizon,device=current.device,dtype=current.dtype)
        horizon=torch.broadcast_to(horizon,current.shape[:-1])[...,None]
        features=torch.cat([(current-self.latent_mean)/self.latent_std,
                            (target-current)/self.latent_std,horizon/8.],dim=-1)
        return self.network(features)

    @classmethod
    def load(cls,path,device='cpu'):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=cls(**saved['model_config'])
        model.load_state_dict(saved['model_state_dict']);model.to(device).eval()
        return model,saved


def make_pairs(cache,split):
    current,target,actions,horizons,identifiers=[],[],[],[],[]
    for key,episode in cache['episodes'].items():
        if episode['split']!=split:continue
        z,a=episode['z'].float(),episode['a'].float()
        # Online controller is translation-only. Rotating commands are excluded.
        for horizon in (1,2,4,8):
            for start in range(len(a)-horizon+1):
                segment=a[start:start+horizon]
                if segment[:,3:6].abs().max()>1e-6 or not torch.allclose(segment[:,-1],-torch.ones(horizon)):
                    continue
                current.append(z[start]);target.append(z[start+horizon]);actions.append(segment[:,:3].mean(0))
                horizons.append(horizon)
        identifiers.append(key)
    return {'current':torch.stack(current),'target':torch.stack(target),'actions':torch.stack(actions),
            'horizon':torch.tensor(horizons,dtype=torch.float32),'trajectory_ids':identifiers}


def train(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    cache=torch.load(args.encoded,map_location='cpu',weights_only=True)
    train_pairs=make_pairs(cache,'train');val_pairs=make_pairs(cache,'val')
    assert not set(train_pairs['trajectory_ids'])&set(val_pairs['trajectory_ids'])
    device=torch.device(args.device)
    model=GoalInverseDynamics().to(device)
    model.latent_mean.copy_(train_pairs['current'].mean(0).to(device))
    model.latent_std.copy_(train_pairs['current'].std(0).clamp_min(.1).to(device))
    datasets=[{k:v.to(device) for k,v in pairs.items() if torch.is_tensor(v)} for pairs in (train_pairs,val_pairs)]
    train_data,val_data=datasets
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    config={'args':vars(args),'encoded_sha256':file_hash(args.encoded),
            'train_trajectories':train_pairs['trajectory_ids'],'validation_trajectories':val_pairs['trajectory_ids'],
            'train_pairs':len(train_data['current']),'val_pairs':len(val_data['current']),
            'runtime_inputs':['current RGB latent','target RGB latent','requested prediction horizon'],
            'training_targets':'Mean raw XYZ commands from original training RGB/action trajectories',
            'privileged_state_inputs':False,'source_sha256':file_hash(__file__)}
    write_json(out/'config.json',config)
    best=float('inf');best_step=0;started=time.time()
    @torch.no_grad()
    def validation():
        model.eval();pred=[]
        for begin in range(0,len(val_data['current']),1024):
            sl=slice(begin,begin+1024)
            pred.append(model(val_data['current'][sl],val_data['target'][sl],val_data['horizon'][sl]))
        predicted=torch.cat(pred);actual=val_data['actions'];moving=actual.norm(dim=-1)>.02
        return {'mse':float(F.mse_loss(predicted,actual)),
                'zero_baseline_mse':float(actual.square().mean()),
                'moving_direction_cosine':float(F.cosine_similarity(predicted[moving],actual[moving]).mean()),
                'moving_examples':int(moving.sum())}
    initial=validation()
    with (out/'metrics.jsonl').open('x',buffering=1) as log:
        for step in range(1,args.steps+1):
            model.train();indices=torch.randint(len(train_data['current']),(args.batch_size,),device=device)
            predicted=model(train_data['current'][indices],train_data['target'][indices],train_data['horizon'][indices])
            loss=F.mse_loss(predicted,train_data['actions'][indices])
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if step%250==0 or step==args.steps:
                val=validation();row={'step':step,'train_mse':float(loss.detach()),'validation':val}
                if val['mse']<best:
                    best,best_step=val['mse'],step
                    checkpoint={'model_config':model.config,'model_state_dict':copy.deepcopy(model.state_dict()),
                                'config':config,'step':step,'validation':val}
                    torch.save(checkpoint,out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
                write_json(out/'status.json',{'status':'running','step':step,'best_step':best_step,'best_mse':best,
                                             'last_validation':val,'elapsed_seconds':time.time()-started})
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
    write_json(out/'status.json',{'status':'complete','step':step,'best_step':best_step,'best_mse':best,
                                 'initial_validation':initial,'last_validation':val,
                                 'elapsed_seconds':time.time()-started,'best_checkpoint_sha256':file_hash(out/'best.pt')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoded',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=6721)
    train(p.parse_args())
