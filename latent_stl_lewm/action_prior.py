"""Training-only visual retrieval prior for CEM initialization.

This is explicitly a demonstration-assisted planner variant. Offline simulator
labels select safe reaching training demonstrations. Runtime retrieval uses
only current RGB latents, never simulator state, test actions, or a waypoint
controller. CEM still evaluates and refines new Gaussian samples.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from .evaluate import sha256, write_json
from .world_model import load_checkpoint


def visual_state_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for component_name in ('encoder', 'projector'):
        component = getattr(model.model, component_name)
        for key, value in sorted(component.state_dict().items()):
            value = value.detach().cpu().contiguous()
            digest.update(f'{component_name}.{key}:{value.dtype}:{list(value.shape)}'.encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class VisualActionPrior:
    def __init__(self, artifact, device='cpu'):
        self.artifact = artifact
        self.latents = F.normalize(artifact['latents'].to(device).float(), dim=-1)
        self.frame_episode = artifact['frame_episode'].to(device)
        self.frame_index = artifact['frame_index'].to(device)
        self.actions = artifact['actions'].to(device).float()

    @classmethod
    def load(cls, path, device='cpu'):
        return cls(torch.load(path, map_location='cpu', weights_only=True), device)

    @torch.no_grad()
    def retrieve(self, z0, horizon, count=8):
        """Nearest current visual states from distinct training demonstrations."""
        distance = 1 - (self.latents @ F.normalize(z0.reshape(-1).float(), dim=-1))
        ordering = distance.argsort()
        selected, episodes = [], set()
        for index in ordering.cpu().tolist():
            episode = int(self.frame_episode[index])
            if episode not in episodes:
                selected.append(index); episodes.add(episode)
                if len(selected) == count:
                    break
        sequence, info = [], []
        for index in selected:
            episode, frame = int(self.frame_episode[index]), int(self.frame_index[index])
            take = self.actions[episode, frame:frame+horizon]
            if len(take) < horizon:
                hold = torch.zeros(horizon-len(take), self.actions.shape[-1], device=take.device)
                hold[:, -1] = -1
                take = torch.cat([take, hold])
            sequence.append(take)
            info.append({'trajectory':self.artifact['trajectory_ids'][episode], 'frame':frame,
                         'cosine_distance':float(distance[index]), 'split':'train'})
        if not sequence:
            raise ValueError('Training visual action prior is empty')
        return torch.stack(sequence), info


@torch.no_grad()
def build(checkpoint, dataset, output, device='cuda', batch_size=32):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model, _ = load_checkpoint(checkpoint, device)
    latents, actions, episode_indices, frame_indices, identifiers = [], [], [], [], []
    with h5py.File(dataset,'r') as data:
        for key, ep in data['trajectories'].items():
            if ep.attrs['split'] != 'train':
                continue
            if not ep['diagnostics/safe'][:].all() or not ep['diagnostics/reach'][:].any():
                continue
            # Translation-only CEM cannot reproduce demonstrations requiring wrist rotation.
            raw=ep['action'][:]
            if np.abs(raw[:,3:6]).max() > 1e-6 or not np.allclose(raw[:,-1],-1):
                continue
            images=ep['rgb'][:]
            tensors=[]
            for offset in range(0,len(images),batch_size):
                rgb=torch.as_tensor(images[offset:offset+batch_size],device=device).permute(0,3,1,2)
                with torch.autocast('cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
                    tensors.append(model.encode(rgb).float().cpu())
            latents.append(torch.cat(tensors))
            actions.append(torch.as_tensor(raw))
            episode_indices.append(torch.full((len(images),),len(identifiers),dtype=torch.int64))
            frame_indices.append(torch.arange(len(images)))
            identifiers.append(key)
    if not identifiers:
        raise ValueError('No safe reaching translation-only training demonstrations')
    artifact={'latents':torch.cat(latents), 'actions':torch.stack(actions),
              'frame_episode':torch.cat(episode_indices), 'frame_index':torch.cat(frame_indices),
              'trajectory_ids':identifiers, 'checkpoint_sha256':sha256(checkpoint),
              'encoder_projector_sha256':visual_state_hash(model), 'dataset_sha256':sha256(dataset),
              'offline_supervision':'Train-split simulator safe/reach labels select demonstrations; no validation/test trajectories enter this bank',
              'runtime_inputs':['encoded RGB'], 'role':'Initialization prior for generated CEM search; demonstration-assisted variant'}
    torch.save(artifact,output)
    metadata={k:v for k,v in artifact.items() if not torch.is_tensor(v)}
    metadata.update(training_demonstrations=len(identifiers),reference_frames=len(artifact['latents']))
    write_json(output.with_suffix('.json'),metadata)
    print(metadata,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('checkpoint','dataset','output'):
        p.add_argument(f'--{arg}',required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    build(**vars(args))


if __name__=='__main__':
    main()
