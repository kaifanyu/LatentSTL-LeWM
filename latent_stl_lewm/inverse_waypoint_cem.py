"""Waypoint CEM with a learned image-goal inverse head near path completion.

This wrapper leaves the shared runner unchanged. The late phase is inferred
from RGB alignment; neither simulator pose nor physical goal error activates,
selects, or stops the controller. The head proposes controls, and LeWM/CEM
evaluates and refines them under the same visual safety preference.
"""
from pathlib import Path
import json
import shutil
import torch
from . import waypoint_cem
from .goal_inverse_dynamics import GoalInverseDynamics
from .train import file_hash, write_json


def main():
    parser=waypoint_cem.parser()
    parser.add_argument('--inverse-checkpoint',required=True)
    parser.add_argument('--inverse-start',type=int,default=75)
    parser.add_argument('--inverse-horizon',type=int,default=4)
    parser.add_argument('--inverse-bound',type=float,default=.15)
    parser.add_argument('--inverse-weight',type=float,default=1.)
    parser.add_argument('--inverse-zero-bias',action='store_true')
    args=parser.parse_args()
    head,saved=GoalInverseDynamics.load(args.inverse_checkpoint,args.device)
    base_guide=waypoint_cem.WaypointGuide
    class InverseGuide(base_guide):
        def update(self,latent,elapsed_steps):
            self.current_latent=latent.float().reshape(1,-1)
            result=super().update(latent,elapsed_steps)
            result['inverse_active']=self.progress>=args.inverse_start
            return result

        def action_sequence(self,horizon):
            nominal=super().action_sequence(horizon)
            if self.progress>=args.inverse_start:
                with torch.no_grad():
                    proposed=head(self.current_latent,self.path[-1:],args.inverse_horizon)[0]
                    if args.inverse_zero_bias:
                        proposed-=head(self.current_latent,self.current_latent,args.inverse_horizon)[0]
                    proposed=proposed.clamp(-args.inverse_bound,args.inverse_bound)
                nominal=nominal.clone()
                nominal[:,:3]=(1-args.inverse_weight)*nominal[:,:3]+args.inverse_weight*proposed[None]
            return nominal
    waypoint_cem.WaypointGuide=InverseGuide
    try:
        status=waypoint_cem.run(args)
    finally:
        waypoint_cem.WaypointGuide=base_guide
    output=Path(args.output)
    provenance={'inverse_checkpoint':str(Path(args.inverse_checkpoint).resolve()),
                'inverse_checkpoint_sha256':file_hash(args.inverse_checkpoint),
                'head_validation':saved['validation'],'head_training_provenance':saved['config'],
                'activation':'Monotone RGB path phase >= inverse_start',
                'target':'Final RGB latent of initial-image-selected training demonstration',
                'proposal':'Constant bounded learned XYZ over horizon, replan after execute_steps',
                'physical_state_inputs':False,'source_hashes':{name:file_hash(Path(__file__).parent/name)
                   for name in ('inverse_waypoint_cem.py','goal_inverse_dynamics.py')}}
    write_json(output/'inverse_provenance.json',provenance)
    for name in ('inverse_waypoint_cem.py','goal_inverse_dynamics.py'):
        shutil.copy2(Path(__file__).parent/name,output/'source_snapshot'/name)
    # Relabel only explicitly replaced nominal candidates, preserving Gaussian
    # and shifted-plan origins and the full original candidate populations.
    mapping={'training_demonstration':'learned_inverse_nominal',
             'demonstration_initial_mean':'learned_inverse_initial_mean'}
    for episode in status['completed_episodes']:
        folder=output/f"scene_{episode['scene_seed']}";counts={}
        for path in sorted(folder.glob('cycle_*.json')):
            row=json.loads(path.read_text())
            if row['alignment'].get('inverse_active'):
                row['selected_source']=mapping.get(row['selected_source'],row['selected_source'])
                for iteration in row.get('iterations',[]):
                    iteration['candidate_sources']=[mapping.get(x,x) for x in iteration['candidate_sources']]
                write_json(path,row)
            counts[row['selected_source']]=counts.get(row['selected_source'],0)+1
        episode['selected_sources']=counts
        write_json(folder/'summary.json',episode)
    write_json(output/'status.json',status)
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
