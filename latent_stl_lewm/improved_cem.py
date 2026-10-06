"""Reference-set CEM with fresh simulator starts and optional training prior."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
from PIL import Image
import torch

from .action_prior import VisualActionPrior, visual_state_hash
from .cem_models import CEMModel
from .evaluate import json_safe, sha256, write_json
from .exploratory_cem import atomic_json, physical_summary, save_trajectory
from .reference_cem import ReferenceCEMConfig, plan_reference_cem
from .reference_predicates import ReferencePredicates, encoder_feature_hash
from .world_model import load_checkpoint


def save_population(directory, cycle, iterations):
    arrays = {}
    summaries = []
    for i, row in enumerate(iterations):
        summary = dict(row)
        for key in ('candidate_actions','candidate_goal_utility','candidate_feasible','candidate_selection_keys'):
            value = summary.pop(key)
            arrays[f'iteration_{i}_{key}'] = np.asarray(value,dtype=np.float32)
        for key, value in summary.pop('candidate_scores').items():
            arrays[f'iteration_{i}_score_{key}'] = np.asarray(value,dtype=np.float32)
        summaries.append(summary)
    np.savez_compressed(directory/f'candidates_{cycle:03d}.npz',**arrays)
    return summaries


@torch.no_grad()
def run(args):
    if args.steps < 1 or not 1 <= args.execute_steps <= args.horizon:
        raise ValueError('Require positive trial budget and execute_steps <= horizon')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    started=time.time()
    status={'status':'initializing','pid':os.getpid(),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'variant':args.variant,'started_at':datetime.now(timezone.utc).isoformat(),
            'closed_loop_executed':False,'completed_episodes':[]}
    def update(**values):
        status.update(values,updated_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-started)
        atomic_json(output/'status.json',status)
    update()
    env=None
    try:
        base, checkpoint = load_checkpoint(args.checkpoint,args.device)
        model=CEMModel(base,kind='local',action_repeat=1)
        predicates=ReferencePredicates.from_artifact(args.references,args.device)
        predicate_artifact=torch.load(args.references,map_location='cpu',weights_only=True)
        # Feature equality matters: the dynamics-only variant deliberately changes
        # the predictor checkpoint while retaining precisely the same encoder.
        expected_hash=predicate_artifact.get('metadata',{}).get('encoder_feature_sha256')
        actual_hash=encoder_feature_hash(base)
        if expected_hash is None or expected_hash != actual_hash:
            raise ValueError('Reference predicate encoder differs from the planning encoder')
        prior=VisualActionPrior.load(args.action_prior,args.device) if args.action_prior else None
        if prior is not None and prior.artifact['encoder_projector_sha256'] != visual_state_hash(base):
            raise ValueError('Action prior encoder differs from the planning encoder')
        dataset_hash=sha256(args.dataset)
        if predicates.metadata.get('dataset_sha256') != dataset_hash:
            raise ValueError('Reference predicate dataset differs from the supplied dataset')
        if prior is not None and prior.artifact['dataset_sha256'] != dataset_hash:
            raise ValueError('Action prior dataset differs from the supplied dataset')
        with h5py.File(args.dataset,'r') as data:
            metadata=json.loads(data.attrs['metadata_json'])
            goal_rgb=data['references/goal_rgb'][:]
            train_actions=np.concatenate([ep['action'][:] for ep in data['trajectories'].values() if ep.attrs['split']=='train'])
            used_seeds={int(ep.attrs['scene_seed']) for ep in data['trajectories'].values()}
        if used_seeds.intersection(args.scene_seeds):
            raise ValueError('Fresh evaluation scene seeds overlap the saved dataset')
        low=np.maximum(train_actions.min(0),-args.translation_bound)
        high=np.minimum(train_actions.max(0),args.translation_bound)
        low[3:6]=high[3:6]=0.; low[6]=high[6]=-1.
        def encode(frames):
            pixels=torch.as_tensor(np.asarray(frames),device=args.device).permute(0,3,1,2)
            with torch.autocast('cuda',dtype=torch.float16,enabled=args.amp):
                return torch.cat([model.encode(part) for part in pixels.split(32)]).float()
        goal=encode([goal_rgb])[0]
        cem=ReferenceCEMConfig(horizon=args.horizon,population=args.population,iterations=args.iterations,
                               elites=args.elites,chunk_size=args.chunk_size,seed=args.seed,
                               initial_std_scale=args.initial_std_scale,min_std=args.min_std,
                               smoothness_weight=args.smoothness_weight,noise_correlation=.8,
                               objective='safe_goal',action_repeat=1,amp=args.amp)
        config={'args':vars(args),'cem':asdict(cem),'checkpoint_sha256':sha256(args.checkpoint),
                'encoder_feature_hash':actual_hash,'references_sha256':sha256(args.references),
                'dataset_sha256':dataset_hash,'action_prior_sha256':sha256(args.action_prior) if args.action_prior else None,
                'model_config':checkpoint['model_config'],'action_low':low.tolist(),'action_high':high.tolist(),
                'objective':'Predicted future-safe candidates ranked by terminal goal progress; least unsafe otherwise. Exact full-history STL reported separately.',
                'supervision':'Train-split simulator labels select safe/unsafe reference examples and successful demonstration priors; validation labels calibrate margins. No physical state/labels in online action selection.',
                'planning_inputs':['current and historical RGB latents','previously executed actions','goal RGB','fixed reference sets'],
                'prior_role':None if prior is None else 'Training visual neighbors initialize CEM and are evaluated alongside new Gaussian candidates; demonstration-assisted planning',
                'evaluation_split':'Fresh scene seeds disjoint from every original train/validation/test trajectory; no optimization on physical test outcomes',
                'stop_policy':'Fixed environment-step budget; physical success does not trigger stopping',
                'source_hashes':{name:sha256(Path(__file__).parent/name) for name in
                                 ('improved_cem.py','reference_cem.py','reference_predicates.py','action_prior.py','cem_models.py','world_model.py')}}
        write_json(output/'config.json',config)
        Image.fromarray(goal_rgb).save(output/'goal.png')
        sim_config=metadata['simulator_config'].copy();sim_config['render_gpu']=args.render_gpu
        from .simulator import MugPreservationEnv
        env=MugPreservationEnv(sim_config)
        if str(args.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        for episode_index, scene_seed in enumerate(args.scene_seeds):
            directory=output/f'scene_{scene_seed}'
            directory.mkdir()
            frames=[env.reset(scene_seed)]
            initial_snapshot=env.snapshot()
            np.savez_compressed(directory/'start_snapshot.npz',**initial_snapshot)
            Image.fromarray(frames[0]).save(directory/'start.png')
            diagnostics=[env.diagnostics()];actions=[];cycle_rows=[]
            history=encode(frames)
            warm=None;cycle=0
            update(status='running',scene_seed=scene_seed,cycle=0,environment_steps=0)
            while len(actions)<args.steps:
                begin=time.time()
                previous=torch.as_tensor(np.asarray(actions,dtype=np.float32).reshape(-1,7),device=args.device)
                model.set_history(history,previous)
                retrieved=None;retrieval_info=[]
                initial_mean=warm
                if prior is not None:
                    retrieved,retrieval_info=prior.retrieve(history[-1:],args.horizon,args.prior_count)
                    initial_mean=retrieved[0]
                    if warm is not None:
                        retrieved=torch.cat([retrieved,warm[None]])
                cem.seed=args.seed + episode_index*10000 + cycle
                plan=plan_reference_cem(model,history[-1:],goal,predicates,low,high,config=cem,
                                        history_latents=history[None,:-1],initial_mean=initial_mean,
                                        initial_candidates=retrieved,previous_action=previous[-1] if len(previous) else None)
                # Preserve exact origins: previous CEM solutions are not demonstrations,
                # and the first mean in a prior run is itself initialized from a demo.
                plan['algorithm_selected_source']=plan['selected_source']
                for iteration in plan['iterations']:
                    origins=iteration['candidate_sources']
                    if prior is not None and warm is not None:
                        origins[2+len(retrieval_info)]='shifted_previous_plan'
                    if iteration['iteration']==0:
                        if prior is not None:origins[0]='retrieved_initial_mean'
                        elif warm is not None:origins[0]='shifted_previous_mean'
                    else:origins[0]='cem_refit_mean'
                plan['selected_source']=plan['iterations'][plan['selected_iteration']]['candidate_sources'][plan['selected_index']]
                proposed=plan.pop('actions').cpu().numpy()
                controls=plan.pop('control_actions')
                predicted_latents=plan.pop('predicted_latents').numpy()
                take=min(args.execute_steps,args.steps-len(actions))
                warm=torch.cat([controls[take:],controls[-1:].expand(take,-1)])
                plan['iterations']=save_population(directory,cycle,plan['iterations'])
                segments=[]
                for action in proposed[:take]:
                    rgb,done,_=env.step(action)
                    frames.append(rgb);segments.append(rgb);actions.append(action.copy())
                    diagnostics.append(env.diagnostics())
                    if done:break
                actual_segment=encode(segments)
                history=torch.cat([history,actual_segment])
                actual=predicates.score(history[None],goal)
                prediction_error=1-torch.nn.functional.cosine_similarity(torch.as_tensor(predicted_latents[0,len(segments)],device=args.device)[None],actual_segment[-1:]).item()
                np.savez_compressed(directory/f'prediction_{cycle:03d}.npz',latent=predicted_latents,
                                    executed_prefix_actual_latent=actual_segment.cpu().numpy(),actions=proposed)
                summary={'cycle':cycle,'environment_steps':len(actions),'selected_source':plan['selected_source'],
                         'predicted_future_safe_min':plan['scores']['future_safe_min'][0],
                         'predicted_goal_progress':plan['scores']['goal_progress'][0],
                         'actual_stl':float(actual['stl'][0]),'actual_terminal':float(actual['terminal'][0]),
                         'execution_prediction_cosine_error':prediction_error,
                         'final_iteration_feasible_candidates':plan['iterations'][-1]['feasible_count'],
                         'physical_diagnostics':physical_summary(diagnostics),'elapsed_seconds':time.time()-begin}
                row={**plan,**summary,'retrieval':retrieval_info,'optimized_action_sequence':proposed.tolist(),
                     'executed_actions':np.asarray(actions[-len(segments):]).tolist(),
                     'actual_scores':{k:v.cpu().tolist() for k,v in actual.items()}}
                write_json(directory/f'cycle_{cycle:03d}.json',row)
                cycle_rows.append(summary)
                Image.fromarray(frames[-1]).save(directory/'latest_frame.png')
                update(status='running',closed_loop_executed=True,scene_seed=scene_seed,cycle=cycle+1,
                       environment_steps=len(actions),last_cycle=summary,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if str(args.device).startswith('cuda') else 0)
                print(json.dumps(json_safe({'variant':args.variant,'scene_seed':scene_seed,**summary})),flush=True)
                if cycle==0 or (cycle+1)%10==0:
                    save_trajectory(directory,frames,actions,diagnostics,fps=metadata['frame_freq_hz'],name='latest')
                cycle+=1
                if done:break
            save_trajectory(directory,frames,actions,diagnostics,fps=metadata['frame_freq_hz'])
            actual=predicates.score(history[None],goal)
            result={'scene_seed':scene_seed,'environment_steps':len(actions),'cycles':cycle,
                    'candidate_evaluations':cycle*args.population*args.iterations,
                    'actual_stl':float(actual['stl'][0]),'actual_terminal':float(actual['terminal'][0]),
                    **physical_summary(diagnostics),'video':str((directory/'trajectory.mp4').resolve()),
                    'selected_source_counts':{source:sum(r['selected_source']==source for r in cycle_rows)
                                               for source in sorted({r['selected_source'] for r in cycle_rows})}}
            write_json(directory/'summary.json',result);write_json(directory/'cycles.json',cycle_rows)
            status['completed_episodes'].append(result);update()
        update(status='completed',physical_successes=sum(x['physical_success'] for x in status['completed_episodes']))
    except BaseException as exc:
        update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        if env is not None:env.close()
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('checkpoint','dataset','references','output','variant'):
        p.add_argument(f'--{arg}',required=True)
    p.add_argument('--action-prior')
    p.add_argument('--device',default='cuda');p.add_argument('--render-gpu',type=int,default=0)
    p.add_argument('--scene-seeds',nargs='+',type=int,default=[900100,900101,900102])
    p.add_argument('--seed',type=int,default=20261005)
    p.add_argument('--horizon',type=int,default=8);p.add_argument('--execute-steps',type=int,default=2)
    p.add_argument('--steps',type=int,default=160);p.add_argument('--population',type=int,default=128)
    p.add_argument('--iterations',type=int,default=4);p.add_argument('--elites',type=int,default=16)
    p.add_argument('--chunk-size',type=int,default=32);p.add_argument('--prior-count',type=int,default=8)
    p.add_argument('--translation-bound',type=float,default=.6)
    p.add_argument('--initial-std-scale',type=float,default=.25);p.add_argument('--min-std',type=float,default=.02)
    p.add_argument('--smoothness-weight',type=float,default=.005)
    p.add_argument('--amp',action='store_true')
    print(json.dumps(json_safe(run(p.parse_args())),indent=2))


if __name__=='__main__':main()
