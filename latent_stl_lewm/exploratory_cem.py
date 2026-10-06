"""Explicitly exploratory CEM execution, with failed validation kept visible.

This separate entry point leaves the evidence-gated MPC path unchanged. It is
for the user's requested empirical test despite the known predicate failure.
Only RGB and action history are supplied to the model and action optimizer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

from .cem_models import load_cem_model
from .evaluate import json_safe, sha256, write_json
from .logic import PredicateConfig, latent_distance, score_trajectories
from .planner import CEMConfig, plan_cem


def atomic_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def save_trajectory(directory, frames, actions, diagnostics, *, fps=20, name='trajectory'):
    np.savez_compressed(directory / f'{name}.npz', rgb=np.asarray(frames),
                        action=np.asarray(actions, dtype=np.float32).reshape(-1, 7),
                        **{f'diagnostic_{key}': np.asarray([d[key] for d in diagnostics]) for key in diagnostics[0]})
    temporary = directory / f'{name}.partial.mp4'
    with imageio.get_writer(temporary, fps=fps, codec='libx264', quality=9,
                           macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
        for frame in frames:
            writer.append_data(frame)
    temporary.replace(directory / f'{name}.mp4')


def physical_summary(diagnostics):
    safe = all(bool(d['safe']) for d in diagnostics)
    reach = any(bool(d['reach']) for d in diagnostics)
    return {'physical_safe': safe, 'physical_reached': reach, 'physical_success': safe and reach,
            'final_goal_error_m': float(diagnostics[-1]['goal_error']),
            'minimum_goal_error_m': min(float(d['goal_error']) for d in diagnostics),
            'maximum_mug_displacement_m': max(float(d['object_displacement']) for d in diagnostics),
            'maximum_mug_rotation_deg': max(float(d['object_rotation_deg']) for d in diagnostics),
            'robot_mug_contact_frames': sum(int(d['robot_mug_contact_count']) > 0 for d in diagnostics)}


@torch.no_grad()
def run(args):
    if not args.exploratory:
        raise ValueError('This diagnostic run requires explicit --exploratory; use mpc.py for gated runs')
    if args.action_repeat != 5 or args.steps % args.action_repeat:
        raise ValueError('Matched comparison uses five-step controls and a whole number of controls')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    status = {'status': 'initializing', 'pid': os.getpid(), 'kind': args.kind,
              'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'started_at': datetime.now(timezone.utc).isoformat(),
              'exploratory': True, 'closed_loop_executed': False, 'completed_episodes': []}

    def update(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.time()-started)
        atomic_json(output / 'status.json', status)

    update()
    env = None
    try:
        model, model_metadata = load_cem_model(args.kind, args.checkpoint, device=args.device, action_repeat=args.action_repeat)
        torch.cuda.reset_peak_memory_stats() if str(args.device).startswith('cuda') else None
        with h5py.File(args.dataset, 'r') as data:
            metadata = json.loads(data.attrs['metadata_json'])
            refs = np.stack([data['references/goal_rgb'][:], data['references/safe_rgb'][:]])
            train_actions = np.concatenate([ep['action'][:] for ep in data['trajectories'].values() if ep.attrs['split'] == 'train'])
            starts = {}
            for bank in args.banks:
                group = data['start_snapshots'][bank]
                starts[bank] = {key: value[()] for key, value in group.items()}
                matching = [ep for ep in data['trajectories'].values() if ep.attrs['bank_id'] == bank]
                if not matching or any(ep.attrs['split'] != 'test' for ep in matching):
                    raise ValueError('CEM episode banks must be held-out test starts')
        low, high = np.maximum(train_actions.min(0), -1), np.minimum(train_actions.max(0), 1)
        # This reaching task needs translation only; training already fixes the gripper open.
        low[3:6] = high[3:6] = 0
        low[6] = high[6] = -1
        predicate = PredicateConfig()
        cem = CEMConfig(horizon=args.horizon, population=args.population, iterations=args.iterations,
                        elites=args.elites, chunk_size=args.chunk_size, seed=args.seed)

        def encode(frames):
            x = torch.as_tensor(np.asarray(frames), device=args.device).permute(0, 3, 1, 2)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                return torch.cat([model.encode(chunk) for chunk in x.split(16)]).float()

        goal, safe = encode(refs)
        Image.fromarray(refs[0]).save(output/'goal.png')
        Image.fromarray(refs[1]).save(output/'safe.png')
        distance = float(latent_distance(goal, safe))
        compatibility_limit = float(1 - np.cos(np.arccos(1-predicate.goal_threshold) + np.arccos(1-predicate.preserve_threshold)))
        evidence = json.loads(Path(args.evidence).read_text()) if args.evidence else None
        if evidence is not None and evidence['checkpoint_sha256'] != model_metadata['checkpoint_sha256']:
            raise ValueError('Local evidence does not match the selected checkpoint')
        config = {'args': vars(args), 'model': model_metadata, 'predicate': predicate.to_dict(),
                  'formula': 'Always(preserve) AND Eventually(reach), exact, including observed history',
                  'dataset_sha256': sha256(args.dataset), 'cem': vars(cem),
                  'action_low': low.tolist(), 'action_high': high.tolist(),
                  'action_bounds_source': 'Training action extrema intersected with environment bounds; rotations fixed and gripper open',
                  'goal_safe_cosine_distance': distance, 'maximum_distance_for_predicate_overlap': compatibility_limit,
                  'predicate_reference_regions_overlap': distance <= compatibility_limit,
                  'validation_gate': None if evidence is None else evidence.get('mpc_gate'),
                  'gate_override_reason': 'User explicitly requested actual simulator CEM experiments after reviewing failed validation',
                  'goal_only_control': 'terminal = negative goal cosine distance, without preservation constraint',
                  'planning_horizon_control_steps': args.horizon * args.action_repeat,
                  'predicted_trace_control_step_spacing': model.native_repeat,
                  'execute_control_steps_per_cycle': args.action_repeat,
                  'sampling': 'Gaussian CEM over held translation controls; no scripted trajectory bank or state-based controller',
                  'warm_start': 'Shift the previous CEM solution by one control block; initial cycle starts from action-bound midpoint',
                  'physical_state_role': 'Reset and diagnostics only; never candidate scoring, action selection, or early success stopping',
                  'predicted_future_limitations': 'Horizon exceeds the previous validated short prediction horizons; this is an exploratory rollout',
                  'selection_note': args.selection_note,
                  'source_hashes': {name: sha256(Path(__file__).parent / name) for name in
                                    ('exploratory_cem.py','cem_models.py','planner.py','logic.py','world_model.py','simulator.py')}}
        write_json(output/'config.json', config)
        update(status='preflight', predicate_reference_regions_overlap=config['predicate_reference_regions_overlap'])
        from .simulator import MugPreservationEnv
        sim_config = metadata['simulator_config'].copy()
        sim_config['render_gpu'] = args.render_gpu
        env = MugPreservationEnv(sim_config)
        first_snapshot = starts[args.banks[0]]
        env.reset(int(first_snapshot['scene_seed']))
        first_rgb = env.restore(first_snapshot)
        first_z = encode([first_rgb])
        model.set_history(first_z, torch.empty((0,7), device=args.device))
        probe = np.zeros((3, 1, 7), dtype=np.float32); probe[:, :, 6] = -1
        probe[1, :, 1] = low[1]; probe[2, :, 1] = high[1]
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
            prediction = model.rollout(first_z.expand(3,-1), torch.as_tensor(probe, device=args.device))
        if not torch.isfinite(prediction).all():
            raise RuntimeError('Nonfinite model output during actual-action preflight')
        preflight = {'finite': True, 'latent_shape': list(first_z.shape),
                     'prediction_shape': list(prediction.shape),
                     'opposite_action_prediction_cosine_distance': float(latent_distance(prediction[1,-1],prediction[2,-1])),
                     'goal_safe_cosine_distance': distance, 'predicate_reference_regions_overlap': distance <= compatibility_limit,
                     'no_physical_diagnostics_used_in_model_probe': True}
        write_json(output/'preflight.json', preflight)
        print(json.dumps({'phase':'preflight', **preflight}), flush=True)

        for bank_index, bank in enumerate(args.banks):
            for objective in args.objectives:
                directory = output / f'{bank}_{objective}'
                directory.mkdir()
                snapshot = starts[bank]
                env.reset(int(snapshot['scene_seed']))
                frames = [env.restore(snapshot)]
                diagnostics, actions, cycle_summaries = [env.diagnostics()], [], []
                history = encode(frames)
                warm = None
                update(status='running', bank=bank, objective=objective, cycle=0, environment_steps=0)
                for cycle in range(args.steps // args.action_repeat):
                    cycle_started = time.time()
                    model.set_history(history, torch.as_tensor(np.asarray(actions,dtype=np.float32).reshape(-1,7), device=args.device))
                    cem.seed = args.seed + 10000 * bank_index + cycle
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                        plan = plan_cem(model, history[-1:], goal, safe, low, high,
                                        objective=objective, config=cem, predicate_config=predicate,
                                        history_latents=history[None,:-1], initial_mean=warm)
                    controls = plan.pop('actions')
                    warm = torch.cat([controls[1:], controls[-1:]], dim=0)
                    sampled = np.asarray([iteration.pop('candidate_actions') for iteration in plan['iterations']], dtype=np.float32)
                    candidate_scores = np.asarray([iteration.pop('candidate_scores') for iteration in plan['iterations']], dtype=np.float32)
                    np.savez_compressed(directory/f'candidates_{cycle:03d}.npz', actions=sampled, scores=candidate_scores)
                    new_frames = []
                    for action in controls[:1].cpu().numpy().repeat(args.action_repeat, axis=0):
                        rgb, done, _ = env.step(action)
                        frames.append(rgb); new_frames.append(rgb); actions.append(action.copy())
                        diagnostics.append(env.diagnostics())
                        if done:
                            break
                    history = torch.cat([history, encode(new_frames)])
                    actual = score_trajectories(history[None], goal, safe, predicate)
                    row = {'cycle': cycle, 'environment_steps': len(actions), **plan,
                           'optimized_control_sequence': controls.cpu().tolist(),
                           'executed_actions': np.asarray(actions[-len(new_frames):]).tolist(),
                           'observed_history_actual_scores': {key: value.cpu().tolist() for key,value in actual.items()},
                           'physical_diagnostics': physical_summary(diagnostics),
                           'elapsed_seconds': time.time()-cycle_started}
                    write_json(directory/f'cycle_{cycle:03d}.json', row)
                    cycle_summaries.append({k:v for k,v in row.items() if k in ('cycle','environment_steps','score','elapsed_seconds','physical_diagnostics')})
                    Image.fromarray(frames[-1]).save(directory/'latest_frame.png')
                    update(status='running', closed_loop_executed=True, bank=bank, objective=objective,
                           cycle=cycle+1, environment_steps=len(actions), last_cycle_seconds=row['elapsed_seconds'],
                           last_predicted_score=plan['score'], last_candidate_score_std=plan['iterations'][-1]['score_std'],
                           physical_diagnostics=row['physical_diagnostics'],
                           peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if str(args.device).startswith('cuda') else 0)
                    print(json.dumps(json_safe({'bank':bank,'objective':objective, **cycle_summaries[-1]})), flush=True)
                    if cycle == 0 or (cycle+1) % 6 == 0:
                        save_trajectory(directory, frames, actions, diagnostics, fps=metadata['frame_freq_hz'], name='latest')
                    if done:
                        break
                save_trajectory(directory, frames, actions, diagnostics, fps=metadata['frame_freq_hz'])
                actual = score_trajectories(history[None], goal, safe, predicate)
                summary = {'bank':bank, 'objective':objective, 'environment_steps':len(actions),
                           'cycles':len(cycle_summaries), 'candidate_evaluations':len(cycle_summaries)*args.population*args.iterations,
                           'actual_stl_robustness':float(actual['stl'][0]), 'actual_terminal_score':float(actual['terminal'][0]),
                           **physical_summary(diagnostics), 'video':str((directory/'trajectory.mp4').resolve())}
                write_json(directory/'summary.json', summary)
                write_json(directory/'cycles.json', cycle_summaries)
                status['completed_episodes'].append(summary)
                update()
        update(status='completed', total_candidate_evaluations=sum(x['candidate_evaluations'] for x in status['completed_episodes']))
    except BaseException as exc:
        update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('local','official_cube_transfer'), required=True)
    for name in ('checkpoint','dataset','output'):
        p.add_argument(f'--{name}', required=True)
    p.add_argument('--evidence')
    p.add_argument('--exploratory', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--render-gpu', type=int, default=0)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261005)
    p.add_argument('--banks', nargs='+', default=['test_0003','test_0000','test_0001','test_0002'])
    p.add_argument('--objectives', nargs='+', choices=('stl','terminal','eventual'), default=['stl','terminal'])
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--action-repeat', type=int, default=5)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--population', type=int, default=128)
    p.add_argument('--iterations', type=int, default=4)
    p.add_argument('--elites', type=int, default=16)
    p.add_argument('--chunk-size', type=int, default=32)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--selection-note', default='')
    args = p.parse_args()
    print(json.dumps(json_safe(run(args)),indent=2))


if __name__ == '__main__':
    main()
