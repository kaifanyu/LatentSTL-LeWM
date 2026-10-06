"""Image-conditioned demonstration baselines and bounded residual latent CEM.

Only successful, preserved-mug training trajectories enter the demonstration
library. Physical state is never used to choose online actions or stop a trial.
This is explicitly demonstration-assisted control, not training-free planning.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .action_prior import VisualActionPrior, visual_state_hash
from .cem_models import CEMModel
from .evaluate import sha256, write_json
from .exploratory_cem import atomic_json, physical_summary, save_trajectory
from .reference_cem import ReferenceCEMConfig, plan_reference_cem
from .reference_predicates import ReferencePredicates, encoder_feature_hash
from .world_model import load_checkpoint


class VisualDemonstrationPolicy:
    """Select a train demonstration visually, then follow its local visual phase."""

    def __init__(self, artifact, allowed_ids, device="cpu", phase_mode="visual", phase_radius=3,
                 phase_penalty=.002):
        self.artifact = artifact
        self.actions = artifact["actions"].to(device)
        self.allowed = [i for i, key in enumerate(artifact["trajectory_ids"]) if key in allowed_ids]
        if not self.allowed:
            raise ValueError("No eligible train demonstrations")
        self.latents = []
        for i in range(len(artifact["trajectory_ids"])):
            rows = torch.where(artifact["frame_episode"] == i)[0]
            rows = rows[artifact["frame_index"][rows].argsort()]
            self.latents.append(F.normalize(artifact["latents"][rows].to(device).float(), dim=-1))
        self.phase_mode, self.phase_radius, self.phase_penalty = phase_mode, phase_radius, phase_penalty
        self.episode = None

    def reset(self, start_latent):
        z = F.normalize(start_latent.reshape(-1).float(), dim=-1)
        distances = torch.stack([1 - self.latents[i][0] @ z for i in self.allowed])
        self.episode = self.allowed[int(distances.argmin())]
        self.start_distance = float(distances.min())
        return {"train_trajectory": self.artifact["trajectory_ids"][self.episode],
                "start_cosine_distance": self.start_distance, "phase_mode": self.phase_mode}

    def propose(self, current_latent, elapsed, horizon):
        if self.episode is None:
            raise RuntimeError("Call reset with a start image first")
        count = len(self.actions[self.episode])
        phase = min(elapsed, count)
        distance = None
        if self.phase_mode == "visual" and 0 < elapsed < count:
            low, high = max(0, phase-self.phase_radius), min(count, phase+self.phase_radius)
            frames = torch.arange(low, high+1, device=current_latent.device)
            distance = 1 - self.latents[self.episode][low:high+1] @ F.normalize(current_latent.reshape(-1).float(), dim=-1)
            cost = distance + self.phase_penalty * (frames-phase).float().square()
            chosen = int(cost.argmin())
            phase = int(frames[chosen])
            distance = float(distance[chosen])
        nominal = self.actions[self.episode, phase:phase+horizon].clone()
        if len(nominal) < horizon:
            hold = nominal.new_zeros(horizon-len(nominal), self.actions.shape[-1])
            hold[:, -1] = -1
            nominal = torch.cat([nominal, hold])
        return nominal, {"train_trajectory": self.artifact["trajectory_ids"][self.episode],
                         "train_phase": phase, "clock_phase": min(elapsed,count),
                         "local_cosine_distance": distance, "split": "train"}


class ResidualDynamics:
    def __init__(self, model, nominal):
        self.model, self.nominal = model, nominal

    def rollout(self, z0, residuals):
        return self.model.rollout(z0, (self.nominal[None] + residuals).clamp(-1, 1))


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    status = {"status": "initializing", "pid": os.getpid(), "variant": args.variant,
              "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "completed_episodes": []}
    def update(**values):
        status.update(values, elapsed_seconds=time.time()-started)
        atomic_json(output/"status.json", status)
    update()
    env = None
    try:
        base, checkpoint = load_checkpoint(args.checkpoint, args.device)
        model = CEMModel(base, kind="local", action_repeat=1)
        prior = VisualActionPrior.load(args.action_prior, args.device)
        refs = ReferencePredicates.from_artifact(args.references, args.device)
        dataset_hash = sha256(args.dataset)
        if prior.artifact["dataset_sha256"] != dataset_hash or refs.metadata["dataset_sha256"] != dataset_hash:
            raise ValueError("Reference/prior dataset provenance mismatch")
        if prior.artifact["encoder_projector_sha256"] != visual_state_hash(base):
            raise ValueError("Prior representation differs")
        if refs.metadata["encoder_feature_sha256"] != encoder_feature_hash(base):
            raise ValueError("Predicate representation differs")
        with h5py.File(args.dataset) as data:
            metadata = json.loads(data.attrs["metadata_json"])
            goal_rgb = data["references/goal_rgb"][:]
            allowed = []
            used_seeds = set()
            for key, ep in data["trajectories"].items():
                used_seeds.add(int(ep.attrs["scene_seed"]))
                if (ep.attrs["split"] == "train" and ep.attrs["policy"] in args.policies
                        and ep["diagnostics/safe"][:].all() and ep["diagnostics/reach"][:].any()):
                    allowed.append(key)
        if used_seeds.intersection(args.scene_seeds):
            raise ValueError("Evaluation seeds overlap original dataset")
        policy = VisualDemonstrationPolicy(prior.artifact, allowed, args.device, args.phase_mode,
                                           args.phase_radius, args.phase_penalty)
        def encode(frames):
            rgb = torch.as_tensor(np.asarray(frames), device=args.device).permute(0,3,1,2)
            with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.float16,
                                enabled=str(args.device).startswith("cuda")):
                return model.encode(rgb).float()
        goal = encode([goal_rgb])[0]
        config = vars(args).copy()
        config.update(checkpoint_sha256=sha256(args.checkpoint), dataset_sha256=dataset_hash,
                      prior_sha256=sha256(args.action_prior), references_sha256=sha256(args.references),
                      allowed_train_trajectories=allowed,
                      online_inputs=["current/initial RGB", "elapsed action count", "training RGB/action library", "goal RGB", "past executed actions"],
                      offline_supervision="Only train-split physical success/safety selects demonstration library; reference thresholds were calibrated on validation.",
                      evaluation_role=args.evaluation_role,
                      stop_policy="Fixed environment-step budget, no physical or latent early success stopping",
                      interpretation="Demonstration-assisted baseline; residual variants run actual generated CEM actions around training demonstrations",
                      source_sha256=sha256(Path(__file__)))
        write_json(output/"config.json", config)
        (output/"source_visual_policy.py").write_text(Path(__file__).read_text())
        Image.fromarray(goal_rgb).save(output/"goal.png")
        sim_cfg = metadata["simulator_config"].copy()
        sim_cfg.update(render_gpu=args.render_gpu, start_jitter=args.start_jitter)
        from .research_scenarios import make_scenario_env
        env = make_scenario_env(sim_cfg,args.scenario)
        write_json(output/"scenario.json",env.metadata())
        cem_config = ReferenceCEMConfig(horizon=args.horizon, population=args.population,
            iterations=args.iterations, elites=min(12,args.population), chunk_size=32,
            min_std=args.residual_bound/8, initial_std_scale=.5, action_cost_weight=args.residual_cost,
            smoothness_weight=0., noise_correlation=.8,
            objective="goal_only" if args.variant == "residual_goal" else "safe_goal")
        for ep_index, scene_seed in enumerate(args.scene_seeds):
            directory = output/f"scene_{scene_seed}"
            directory.mkdir()
            frames = [env.reset(scene_seed)]
            np.savez_compressed(directory/"start_snapshot.npz", **env.snapshot())
            Image.fromarray(frames[0]).save(directory/"start.png")
            diagnostics, actions, cycles = [env.diagnostics()], [], []
            history = encode(frames)
            choice = policy.reset(history[-1])
            write_json(directory/"demonstration.json", choice)
            update(status="running", scene_seed=scene_seed, environment_steps=0, demonstration=choice)
            while len(actions) < args.steps:
                nominal, origin = policy.propose(history[-1], len(actions), args.horizon)
                plan_log = None
                if args.variant == "baseline":
                    selected = nominal
                    origin["action_source"] = "training_visual_demonstration"
                else:
                    previous = torch.as_tensor(np.asarray(actions,dtype=np.float32).reshape(-1,7),device=args.device)
                    model.set_history(history, previous)
                    bound = torch.tensor([args.residual_bound]*3 + [0.]*4,device=args.device)
                    cem_config.seed = args.seed + ep_index*10000 + len(cycles)
                    plan = plan_reference_cem(ResidualDynamics(model,nominal), history[-1:], goal, refs,
                        -bound, bound, config=cem_config, history_latents=history[None,:-1])
                    selected = (nominal + plan["actions"]).clamp(-1,1)
                    origin.update(action_source="residual_cem", residual_source=plan["selected_source"],
                                  max_action_residual=float(plan["actions"].abs().max()))
                    plan_log = {key: value for key,value in plan.items()
                                if key not in {"actions","control_actions","predicted_latents","iterations"}}
                    plan_log["iterations"] = [{key:value for key,value in row.items()
                        if key not in {"candidate_actions","candidate_scores","candidate_goal_utility","candidate_feasible","candidate_selection_keys"}}
                        for row in plan["iterations"]]
                    # Counterfactual ranking uses exactly the same sampled
                    # candidates, not another simulator query or search budget.
                    choices=[]
                    for iteration,row in enumerate(plan["iterations"]):
                        for index,(utility,safety) in enumerate(zip(row["candidate_goal_utility"],
                                                                  row["candidate_scores"]["future_safe_min"])):
                            feasible=safety>=0
                            choices.append({"location":(iteration,index),"utility":utility,"safety":safety,
                                            "goal_key":(utility,safety),
                                            "safe_key":(feasible,utility if feasible else safety,safety if feasible else utility)})
                    goal_choice=max(choices,key=lambda c:c["goal_key"])
                    safe_choice=max(choices,key=lambda c:c["safe_key"])
                    plan_log["counterfactual_goal_winner_future_safe_min"]=goal_choice["safety"]
                    plan_log["counterfactual_safe_winner_future_safe_min"]=safe_choice["safety"]
                    plan_log["safety_veto_changes_winner"]=goal_choice["location"]!=safe_choice["location"]
                    plan_log["safety_veto_changes_actions"]=not np.array_equal(
                        plan["iterations"][goal_choice["location"][0]]["candidate_actions"][goal_choice["location"][1]],
                        plan["iterations"][safe_choice["location"][0]]["candidate_actions"][safe_choice["location"][1]])
                    np.savez_compressed(directory/f"cem_{len(cycles):03d}.npz",
                        nominal=nominal.cpu().numpy(), residual=plan["actions"].cpu().numpy(),
                        candidates=np.asarray([r["candidate_actions"] for r in plan["iterations"]],dtype=np.float32),
                        candidate_keys=np.asarray([r["candidate_selection_keys"] for r in plan["iterations"]],dtype=np.float32))
                take = min(args.execute_steps, args.steps-len(actions))
                executed_residual = selected[:take]-nominal[:take]
                origin["executed_residual_mean_abs_xyz"] = float(executed_residual[:,:3].abs().mean())
                origin["executed_residual_max_abs"] = float(executed_residual.abs().max())
                selected = selected[:take].cpu().numpy()
                new_frames = []
                for action in selected:
                    rgb, done, _ = env.step(action)
                    actions.append(action.copy()); frames.append(rgb); new_frames.append(rgb)
                    diagnostics.append(env.diagnostics())
                history = torch.cat([history, encode(new_frames)])
                record = {"cycle":len(cycles),"environment_steps":len(actions),"origin":origin,
                          "executed_actions":selected.tolist(),"plan":plan_log,
                          "physical_diagnostics":physical_summary(diagnostics)}
                cycles.append(record)
                write_json(directory/f"cycle_{len(cycles)-1:03d}.json",record)
                if len(cycles)%10==0:
                    update(status="running",scene_seed=scene_seed,environment_steps=len(actions),last_cycle=record)
            save_trajectory(directory,frames,actions,diagnostics,fps=metadata["frame_freq_hz"])
            actual = refs.score(history[None],goal)
            np.savez_compressed(directory/"actual_latents.npz",latents=history.cpu().numpy(),goal=goal.cpu().numpy(),
                                safe_trace=actual["safe_trace"][0].cpu().numpy(),
                                reach_trace=actual["reach_trace"][0].cpu().numpy())
            result={"scene_seed":scene_seed,"environment_steps":len(actions),**physical_summary(diagnostics),
                    "actual_stl":float(actual["stl"][0]),
                    "actual_safe_min":float(actual["full_safe_min"][0]),
                    "actual_reach_max":float(actual["full_reach_max"][0]),
                    "demonstration":choice,"variant":args.variant,
                    "video":str((directory/"trajectory.mp4").resolve()),
                    "residual_nonzero_cycles":sum(c["origin"].get("max_action_residual",0)>1e-6 for c in cycles),
                    "safety_veto_changes_actions_cycles":sum(bool((c["plan"] or {}).get("safety_veto_changes_actions")) for c in cycles),
                    "executed_residual_mean_abs_xyz":float(np.mean([c["origin"]["executed_residual_mean_abs_xyz"] for c in cycles])),
                    "scenario":args.scenario}
            write_json(directory/"summary.json",result)
            status["completed_episodes"].append(result)
            update()
            print(json.dumps(result),flush=True)
        update(status="completed",physical_successes=sum(ep["physical_success"] for ep in status["completed_episodes"]))
    except BaseException as error:
        update(status="failed",error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",default="runs/campaign_20261004T073606Z/seed_2/train/best.pt")
    p.add_argument("--dataset",default="datasets/libero_mug_v1/trajectories.h5")
    p.add_argument("--action-prior",default="runs/cem_improved_20261005T064943Z/action_prior.pt")
    p.add_argument("--references",default="runs/cem_improved_20261005T064943Z/references/references.pt")
    p.add_argument("--output",required=True)
    p.add_argument("--variant",choices=["baseline","residual_safe","residual_goal"],default="baseline")
    p.add_argument("--phase-mode",choices=["clock","visual"],default="visual")
    p.add_argument("--phase-radius",type=int,default=3);p.add_argument("--phase-penalty",type=float,default=.002)
    p.add_argument("--scene-seeds",nargs="+",type=int,default=[901100,901101,901102])
    p.add_argument("--policies",nargs="+",default=["safe_arc"])
    p.add_argument("--steps",type=int,default=120);p.add_argument("--execute-steps",type=int,default=2)
    p.add_argument("--horizon",type=int,default=8);p.add_argument("--population",type=int,default=64)
    p.add_argument("--iterations",type=int,default=3);p.add_argument("--residual-bound",type=float,default=.01)
    p.add_argument("--residual-cost",type=float,default=.001)
    p.add_argument("--start-jitter",type=float,default=.012)
    p.add_argument("--device",default="cuda");p.add_argument("--render-gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=20261005)
    p.add_argument("--scenario",default="baseline")
    p.add_argument("--evaluation-role",choices=["development","untouched_final"],default="development")
    args=p.parse_args()
    if not 1<=args.execute_steps<=args.horizon or args.steps<1 or args.residual_bound<0:
        p.error("Require positive steps, execute <= horizon, and nonnegative residual bound")
    run(args)


if __name__=="__main__":main()
