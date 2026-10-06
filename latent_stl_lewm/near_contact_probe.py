"""Branch every specified action candidate from one near-contact simulator state.

This is an offline diagnostic, not an online controller. All branches are
executed and scored; simulator measurements label outcomes after execution.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import h5py
import torch
from .cem_models import CEMModel
from .exploratory_cem import physical_summary, save_trajectory
from .reference_predicates import ReferencePredicates
from .train import file_hash, write_json
from .world_model import load_checkpoint


@torch.no_grad()
def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    with h5py.File(args.dataset,'r') as data:
        cfg=json.loads(data.attrs['metadata_json']);sim=cfg['simulator_config'].copy();sim['render_gpu']=0
        episode=data['trajectories'][args.trajectory]
        if episode.attrs['split']!='train' or episode.attrs['policy']!='collision_straight':
            raise ValueError('Probe prefix must come from a training collision-straight demonstration')
        commands=episode['action'][:];goal_rgb=data['references/goal_rgb'][:]
    from .simulator import MugPreservationEnv
    env=MugPreservationEnv(sim)
    try:
        prefix_frames=[env.reset(args.scene_seed)];prefix_diagnostics=[env.diagnostics()]
        for action in commands[:args.prefix]:
            image,_,_=env.step(action);prefix_frames.append(image);prefix_diagnostics.append(env.diagnostics())
        snapshot=env.snapshot();np.savez_compressed(output/'branch_snapshot.npz',**snapshot)
        branches={}
        candidates={'continue':commands[args.prefix:args.prefix+16].copy()}
        for name,xyz in [('up',[0,0,.3]),('lateral',[.3,0,0]),('noop',[0,0,0])]:
            a=np.zeros((16,7),dtype=np.float32);a[:,:3]=xyz;a[:,-1]=-1;candidates[name]=a
        for name,actions in candidates.items():
            frames=[env.restore(snapshot)];diagnostics=[env.diagnostics()]
            for action in actions:
                image,_,_=env.step(action);frames.append(image);diagnostics.append(env.diagnostics())
            branch=output/name;branch.mkdir()
            save_trajectory(branch,prefix_frames+frames[1:],np.concatenate([commands[:args.prefix],actions]),
                            prefix_diagnostics+diagnostics[1:],fps=cfg['frame_freq_hz'])
            branches[name]={'rgb':np.asarray(frames),'diagnostics':diagnostics,'actions':actions}
        references={'frame':ReferencePredicates.from_artifact(args.references,args.device),
                    'temporal':ReferencePredicates.from_artifact(args.temporal_references,args.device)}
        rows=[]
        for label,checkpoint in [('original',args.original),('multistep',args.multistep)]:
            base,_=load_checkpoint(checkpoint,args.device);model=CEMModel(base,kind='local',action_repeat=1)
            def encode(images):
                pixels=torch.as_tensor(np.asarray(images),device=args.device).permute(0,3,1,2)
                with torch.autocast('cuda',dtype=torch.float16):return model.encode(pixels).float()
            history=encode(prefix_frames);goal=encode([goal_rgb])[0]
            model.set_history(history,torch.as_tensor(commands[:args.prefix],device=args.device))
            actions=torch.as_tensor(np.stack(list(candidates.values())),device=args.device)
            with torch.autocast('cuda',dtype=torch.float16):pred=model.rollout(history[-1:].expand(4,-1),actions)
            actual=torch.stack([encode(branch['rgb']) for branch in branches.values()])
            np.savez_compressed(output/f'{label}_latents.npz',predicted=pred.cpu().numpy(),actual=actual.cpu().numpy(),history=history.cpu().numpy())
            for horizon in (8,16):
                for reference,predicates in references.items():
                    predicted=predicates.score(pred[:,:horizon+1],goal,history=history[None,:-1])
                    observed=predicates.score(actual[:,:horizon+1],goal,history=history[None,:-1])
                    for index,(name,branch) in enumerate(branches.items()):
                        rows.append({'model':label,'reference':reference,'horizon':horizon,'candidate':name,
                                     'predicted_future_safe_min':float(predicted['future_safe_min'][index]),
                                     'actual_encoded_future_safe_min':float(observed['future_safe_min'][index]),
                                     'predicted_stl':float(predicted['stl'][index]),'actual_encoded_stl':float(observed['stl'][index]),
                                     'physical':physical_summary(prefix_diagnostics+branch['diagnostics'][1:horizon+1])})
            del model,base
        report={'args':vars(args),'purpose':'Offline near-contact counterfactual ranking; every fixed branch executed',
                'prefix_physical':physical_summary(prefix_diagnostics),'runtime_physical_action_selection':False,
                'dataset_sha256':file_hash(args.dataset),'source_sha256':file_hash(__file__),'rows':rows}
        write_json(output/'report.json',report)
        print(json.dumps(report,indent=2))
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','original','multistep','references','temporal-references','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--scene-seed',type=int,default=902140)
    parser.add_argument('--trajectory',default='000003');parser.add_argument('--prefix',type=int,default=24)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())
