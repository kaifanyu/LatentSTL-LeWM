"""Train-only local linear latent transitions with explicit action sensitivity.

The encoder remains the trained LeWM encoder. At each predicted state a weighted
ridge fit uses nearby training RGB transitions, current/previous visual features,
and commanded actions. No simulator measurements or evaluation actions are fit.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import torch
from torch.nn import functional as F
from .train import file_hash, write_json
from .world_model import load_checkpoint
from .reference_predicates import encoder_feature_hash


class LocalLatentDynamics:
    def __init__(self, base, artifact, *, neighbors=96, ridge=.1, latent_dims=8,
                 velocity_dims=4, delta_clip=2., device='cuda'):
        self.model=base.eval().requires_grad_(False)
        self.config=base.config
        self.artifact=artifact
        self.neighbors=neighbors;self.ridge=ridge;self.latent_dims=latent_dims
        self.velocity_dims=velocity_dims;self.delta_clip=delta_clip
        self.z=artifact['z'].to(device).float();self.normalized=F.normalize(self.z,dim=-1)
        self.delta=artifact['delta'].to(device).float()
        self.velocity=artifact['velocity'].to(device).float()
        self.actions=artifact['actions'].to(device).float()
        self.basis=artifact['basis'].to(device).float()
        self.z_scale=artifact['z_scale'].to(device).float()
        self.v_scale=artifact['v_scale'].to(device).float()
        self.action_scale=artifact['action_scale'].to(device).float()
        self.features=torch.cat([(self.z@self.basis[:,:latent_dims])/self.z_scale[:latent_dims],
                                 (self.velocity@self.basis[:,:velocity_dims])/self.v_scale[:velocity_dims],
                                 self.actions/self.action_scale],dim=-1)
        self.previous_z=None

    @classmethod
    def load(cls, checkpoint, artifact, device='cuda', **kwargs):
        model,_=load_checkpoint(checkpoint,device)
        saved=torch.load(artifact,map_location='cpu',weights_only=True)
        if encoder_feature_hash(model)!=saved['encoder_feature_sha256']:
            raise ValueError('Transition artifact encoder mismatch')
        return cls(model,saved,device=device,**kwargs)

    def encode(self,rgb):return self.model.encode(rgb).float()

    def set_history(self,latents,actions):
        if latents.ndim!=2 or len(latents)!=len(actions)+1:
            raise ValueError('Misaligned observed history')
        self.previous_z=latents[-2:-1].clone() if len(latents)>1 else None

    def predict_next(self,z,velocity,actions):
        # Disable FP16 around the small least-squares system for stable solves.
        with torch.autocast(device_type=z.device.type,enabled=False):
            z,velocity,actions=z.float(),velocity.float(),actions.float()
            distance=(1-F.normalize(z,dim=-1)@self.normalized.T).clamp_min(0)
            nearest,indices=distance.topk(min(self.neighbors,len(self.z)),largest=False)
            bandwidth=(nearest[:,-1:]*.25).clamp_min(.001)
            weights=torch.softmax(-(nearest-nearest[:,:1])/bandwidth,dim=-1)*indices.shape[1]
            query=torch.cat([(z@self.basis[:,:self.latent_dims])/self.z_scale[:self.latent_dims],
                             (velocity@self.basis[:,:self.velocity_dims])/self.v_scale[:self.velocity_dims],
                             actions/self.action_scale],dim=-1)
            features=self.features[indices]-query[:,None]
            features=torch.cat([torch.ones_like(features[:,:,:1]),features],dim=-1)
            gram=features.transpose(1,2)@(features*weights[:,:,None])
            eye=torch.eye(gram.shape[-1],device=z.device)[None]*self.ridge
            eye[:,0,0]=self.ridge*.01
            right=features.transpose(1,2)@(self.delta[indices]*weights[:,:,None])
            coefficients=torch.linalg.solve(gram+eye,right)
            delta=coefficients[:,0]
            if self.delta_clip:
                limit=self.delta[indices].norm(dim=-1).amax(dim=1)*self.delta_clip
                delta=delta*torch.minimum(torch.ones_like(limit),limit/delta.norm(dim=-1).clamp_min(1e-8))[:,None]
            return z+delta

    def rollout(self,z0,actions):
        velocity=torch.zeros_like(z0) if self.previous_z is None else z0-self.previous_z
        states=[z0.float()]
        for step in range(actions.shape[1]):
            successor=self.predict_next(states[-1],velocity,actions[:,step])
            velocity=successor-states[-1];states.append(successor)
        return torch.stack(states,dim=1)


def build(args):
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    cached=torch.load(args.encoded,map_location='cpu',weights_only=True)
    zs,velocities,actions,deltas,ids=[],[],[],[],[]
    for key,episode in cached['episodes'].items():
        if episode['split']!='train':continue
        z,a=episode['z'].float(),episode['a'].float()
        v=torch.cat([torch.zeros_like(z[:1]),z[1:]-z[:-1]])
        # Near-duplicate no-op frames otherwise swamp neighboring transitions.
        stride=12 if episode['policy']=='noop' else args.stride
        selected=torch.arange(0,len(a),stride)
        zs.append(z[:-1][selected]);velocities.append(v[:-1][selected]);actions.append(a[selected])
        deltas.append((z[1:]-z[:-1])[selected]);ids.append(key)
    z=torch.cat(zs);velocity=torch.cat(velocities);action=torch.cat(actions);delta=torch.cat(deltas)
    _,_,basis=torch.pca_lowrank(z-z.mean(0),q=24,center=False,niter=4)
    base,_=load_checkpoint(args.checkpoint)
    artifact={'z':z,'velocity':velocity,'actions':action,'delta':delta,'basis':basis,
              'z_scale':(z@basis).std(0).clamp_min(.1),'v_scale':(velocity@basis).std(0).clamp_min(.03),
              'action_scale':action.std(0).clamp_min(.1),'trajectory_ids':ids,
              'training_split':'train','encoder_feature_sha256':encoder_feature_hash(base),
              'encoded_sha256':file_hash(args.encoded),'dataset_sha256':cached['dataset_sha256'],
              'training_inputs':['RGB latents','previous RGB latent difference','actions'],
              'privileged_inputs':False}
    torch.save(artifact,out)
    report={k:v for k,v in artifact.items() if not torch.is_tensor(v)}
    report['transitions']=len(z);report['artifact_sha256']=file_hash(out)
    write_json(out.with_suffix('.json'),report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('encoded','checkpoint','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--stride',type=int,default=2)
    build(p.parse_args())
