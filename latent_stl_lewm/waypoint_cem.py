"""Demonstration-assisted visual waypoint CEM research.

Only training images/actions define intermediate waypoints. Runtime phase
estimation uses RGB latents (optionally a clock prior), never physical state.
Final-goal STL and unchanged physical criteria are evaluation metrics. The
planner's explicit surrogate is future visual safety plus nearby-waypoint
progress and, optionally, a penalty for departing from demonstration actions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .action_prior import visual_state_hash
from .cem_models import CEMModel
from .evaluate import sha256, write_json, json_safe
from .exploratory_cem import atomic_json, physical_summary, save_trajectory
from .improved_cem import save_population
from .reference_cem import ReferenceCEMConfig, plan_reference_cem
from .reference_predicates import ReferencePredicates, encoder_feature_hash
from .world_model import load_checkpoint


class WaypointGuide:
    """Monotone visual alignment to one initial-image-selected training path."""
    def __init__(self, artifact, *, device="cpu", episode=None, lookahead=8,
                 maximum_advance=8, backward_window=4, tracking="visual",
                 clock_weight=0.0):
        self.artifact = artifact
        self.device = torch.device(device)
        self.lookahead = lookahead
        self.maximum_advance = maximum_advance
        self.backward_window = backward_window
        self.tracking = tracking
        self.clock_weight = clock_weight
        if tracking not in {"visual", "clock", "hybrid"}:
            raise ValueError("Unknown tracking rule")
        if lookahead < 1 or maximum_advance < 1 or backward_window < 0 or clock_weight < 0:
            raise ValueError("Invalid waypoint tracking configuration")
        self.episode = episode
        self.progress = 0

    def start(self, latent):
        starts = self.artifact["latents"][self.artifact["frame_index"] == 0].to(self.device)
        distance = 1 - F.cosine_similarity(starts.float(), latent.reshape(1, -1).float())
        if self.episode is None:
            self.episode = int(distance.argmin())
        selected = self.artifact["frame_episode"] == self.episode
        self.path = self.artifact["latents"][selected].to(self.device).float()
        self.actions = self.artifact["actions"][self.episode].to(self.device).float()
        self.progress = 0
        return {"trajectory": self.artifact["trajectory_ids"][self.episode],
                "episode_index": self.episode, "initial_cosine_distance": float(distance[self.episode]),
                "split": "train"}

    def update(self, latent, elapsed_steps):
        end = len(self.path) - 1
        if self.tracking == "clock":
            matched = min(elapsed_steps, end)
            distance = 1 - F.cosine_similarity(self.path[matched], latent.reshape(-1), dim=0)
        else:
            lo = max(0, self.progress - self.backward_window)
            hi = min(end + 1, self.progress + self.maximum_advance + 1)
            distance = 1 - F.cosine_similarity(self.path[lo:hi], latent.reshape(1, -1), dim=-1)
            rank = distance.clone()
            if self.tracking == "hybrid":
                # A known control clock regularizes ambiguous visual matches;
                # no simulator position, contact, or goal distance is used.
                rank += self.clock_weight * (torch.arange(lo, hi, device=self.device) - min(elapsed_steps, end)).abs()
            offset = int(rank.argmin())
            matched, distance = lo + offset, distance[offset]
        self.progress = max(self.progress, matched)
        self.target_index = min(end, self.progress + self.lookahead)
        return {"matched_frame": matched, "progress_frame": self.progress,
                "waypoint_frame": self.target_index, "visual_distance": float(distance),
                "tracking": self.tracking}

    @property
    def target(self):
        return self.path[self.target_index]

    def action_sequence(self, horizon):
        part = self.actions[self.progress:self.progress + horizon]
        if len(part) < horizon:
            hold = torch.zeros((horizon - len(part), self.actions.shape[-1]), device=self.device)
            hold[:, -1] = -1
            part = torch.cat([part, hold])
        return part


class ActionTraceModel:
    def __init__(self, model):
        self.model = model

    def rollout(self, z0, actions):
        self.last_actions = actions
        return self.model.rollout(z0, actions)


class WaypointPredicates:
    """Keep exact original STL, change only the documented CEM surrogate."""
    def __init__(self, predicates, waypoint, *, dynamics=None, nominal=None,
                 deviation_weight=0.0):
        self.predicates, self.waypoint = predicates, waypoint
        self.dynamics, self.nominal = dynamics, nominal
        self.deviation_weight = deviation_weight

    def score(self, latents, goal, history=None):
        scores = self.predicates.score(latents, goal, history=history)
        waypoint_distance = 1 - F.cosine_similarity(latents.float(), self.waypoint.float(), dim=-1)
        progress = waypoint_distance[:, 0] - waypoint_distance[:, -1]
        scores["final_goal_progress"] = scores["goal_progress"]
        scores["waypoint_distance_trace"] = waypoint_distance
        scores["waypoint_progress"] = progress
        scores["goal_progress"] = progress
        if self.deviation_weight:
            deviation = (self.dynamics.last_actions[..., :3] - self.nominal[None, :, :3]).square().mean((1, 2))
            scores["action_prior_deviation"] = deviation
            scores["goal_progress"] = progress - self.deviation_weight * deviation
        return scores


@torch.no_grad()
def run(args, model_override=None):
    if args.steps < 1 or not 1 <= args.execute_steps <= args.horizon:
        raise ValueError("Require steps > 0 and 1 <= execute_steps <= horizon")
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source_snapshot"
    source_dir.mkdir()
    source_hashes = {}
    for name in ("waypoint_cem.py", "reference_cem.py", "reference_predicates.py",
                 "cem_models.py", "world_model.py", "action_prior.py", "research_scenarios.py"):
        source = Path(__file__).parent / name
        if source.exists():
            shutil.copy2(source, source_dir / name)
            source_hashes[name] = sha256(source_dir / name)
    write_json(source_dir / "manifest.json", source_hashes)
    started = time.time()
    base, checkpoint = load_checkpoint(args.checkpoint, args.device)
    model = model_override if model_override is not None else CEMModel(base, kind="local", action_repeat=1)
    if model_override is not None:
        override_encoder = getattr(model_override, "encoder_model", getattr(model_override, "model", None))
        if override_encoder is None or encoder_feature_hash(override_encoder) != encoder_feature_hash(base):
            raise ValueError("Model override must expose its unchanged LeWM encoder through encoder_model or model")
    traced = ActionTraceModel(model)
    predicates = ReferencePredicates.from_artifact(args.references, args.device)
    artifact = torch.load(args.action_prior, weights_only=True, map_location="cpu")
    if artifact["encoder_projector_sha256"] != visual_state_hash(base):
        raise ValueError("Waypoint feature space differs from planning encoder")
    if predicates.metadata["encoder_feature_sha256"] != encoder_feature_hash(base):
        raise ValueError("Safety feature space differs from planning encoder")
    dataset_hash = sha256(args.dataset)
    if artifact["dataset_sha256"] != dataset_hash or predicates.metadata["dataset_sha256"] != dataset_hash:
        raise ValueError("Training reference dataset provenance does not match supplied dataset")
    with h5py.File(args.dataset, "r") as data:
        metadata = json.loads(data.attrs["metadata_json"])
        goal_rgb = data["references/goal_rgb"][:]
        training_actions = np.concatenate([ep["action"][:] for ep in data["trajectories"].values() if ep.attrs["split"] == "train"])
        old_seeds = {int(ep.attrs["scene_seed"]) for ep in data["trajectories"].values()}
    if old_seeds.intersection(args.scene_seeds):
        raise ValueError("Development scene overlaps original dataset")
    low = np.maximum(training_actions.min(0), -args.translation_bound)
    high = np.minimum(training_actions.max(0), args.translation_bound)
    low[3:6] = high[3:6] = 0
    low[6] = high[6] = -1
    def encode(images):
        x = torch.as_tensor(np.asarray(images), device=args.device).permute(0, 3, 1, 2)
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(args.device).startswith("cuda")):
            return model.encode(x).float()
    goal = encode([goal_rgb])[0]
    config = ReferenceCEMConfig(horizon=args.horizon, population=args.population,
                                iterations=args.iterations, elites=args.elites, chunk_size=args.chunk_size,
                                initial_std_scale=args.std, min_std=args.min_std,
                                smoothness_weight=args.smoothness, noise_correlation=.8,
                                objective=args.objective)
    write_json(output / "config.json", {"args": vars(args), "cem": asdict(config),
                "evaluation_role": args.evaluation_role,
                "source_hashes": source_hashes,
                "algorithm": "Demonstration-assisted waypoint CEM" if args.mode == "cem" else "Training demonstration replay control",
                "runtime_policy_inputs": ["current/historical RGB latents", "goal RGB latent", "fixed safe/unsafe reference banks",
                                          "previous actions", "elapsed control count", "training visual waypoints/actions"],
                "physical_state_for_action_selection": False, "stop_on_physical_success": False,
                "checkpoint_sha256": sha256(args.checkpoint), "prior_sha256": sha256(args.action_prior),
                "reference_sha256": sha256(args.references), "action_low": low.tolist(), "action_high": high.tolist()})
    Image.fromarray(goal_rgb).save(output / "goal.png")
    simconfig = metadata["simulator_config"].copy()
    simconfig["render_gpu"] = args.render_gpu
    if args.scenario != "baseline":
        from .research_scenarios import make_scenario_env
        env = make_scenario_env(simconfig, args.scenario)
    else:
        from .simulator import MugPreservationEnv
        env = MugPreservationEnv(simconfig)
    status = {"status": "running", "completed_episodes": []}
    try:
        for scene_seed in args.scene_seeds:
            folder = output / f"scene_{scene_seed}"
            folder.mkdir()
            frames = [env.reset(scene_seed)]
            np.savez_compressed(folder / "start_snapshot.npz", **env.snapshot())
            diagnostics = [env.diagnostics()]
            actions, cycles = [], []
            history = encode(frames)
            guide = WaypointGuide(artifact, device=args.device, episode=args.episode,
                                  lookahead=args.lookahead, maximum_advance=args.maximum_advance,
                                  tracking=args.tracking, clock_weight=args.clock_weight)
            chosen = guide.start(history[-1])
            warm = None
            while len(actions) < args.steps:
                cycle = len(cycles)
                alignment = guide.update(history[-1], len(actions))
                nominal = guide.action_sequence(args.horizon).clamp(torch.as_tensor(low,device=args.device),torch.as_tensor(high,device=args.device))
                previous = torch.as_tensor(np.asarray(actions, dtype=np.float32).reshape(-1, 7), device=args.device)
                model.set_history(history, previous)
                if args.mode == "cem":
                    target = guide.target if args.surrogate == "waypoint" else goal
                    scorer = WaypointPredicates(predicates, target, dynamics=traced,
                                                nominal=nominal, deviation_weight=args.deviation_weight)
                    seeds = nominal[None] if warm is None else torch.stack([nominal, warm])
                    config.seed = args.seed + scene_seed + cycle
                    plan = plan_reference_cem(traced, history[-1:], goal, scorer, low, high,
                                              config=config, history_latents=history[None, :-1],
                                              initial_mean=nominal, initial_candidates=seeds,
                                              previous_action=previous[-1] if len(previous) else None)
                    plan["algorithm_selected_source"] = plan["selected_source"]
                    for iteration in plan["iterations"]:
                        origins = iteration["candidate_sources"]
                        origins[0] = "demonstration_initial_mean" if iteration["iteration"] == 0 else "cem_refit_mean"
                        origins[2] = "training_demonstration"
                        if warm is not None:
                            origins[3] = "shifted_previous_plan"
                    plan["selected_source"] = plan["iterations"][plan["selected_iteration"]]["candidate_sources"][plan["selected_index"]]
                    proposed = plan.pop("actions").cpu().numpy()
                    controls = plan.pop("control_actions")
                    predicted = plan.pop("predicted_latents").numpy()
                    plan["iterations"] = save_population(folder, cycle, plan["iterations"])
                else:
                    proposed = nominal.cpu().numpy()
                    controls = nominal
                    with torch.autocast("cuda", dtype=torch.float16, enabled=str(args.device).startswith("cuda")):
                        predicted = model.rollout(history[-1:], nominal[None]).float().cpu().numpy()
                    plan = {"selected_source": "training_demonstration_control", "candidate_evaluations": 0}
                take = min(args.execute_steps, args.steps-len(actions))
                warm = torch.cat([controls[take:], controls[-1:].expand(take, -1)])
                segment = []
                for action in proposed[:take]:
                    rgb, done, _ = env.step(action)
                    frames.append(rgb); segment.append(rgb); actions.append(action.copy())
                    diagnostics.append(env.diagnostics())
                    if done: break
                actual_segment = encode(segment)
                history = torch.cat([history, actual_segment])
                scores = predicates.score(history[None], goal)
                np.savez_compressed(folder / f"prediction_{cycle:03d}.npz", latent=predicted,
                                    executed_prefix_actual_latent=actual_segment.cpu().numpy(), actions=proposed)
                row = {"cycle": cycle, "environment_steps": len(actions), "reference": chosen,
                       "alignment": alignment, "physical": physical_summary(diagnostics),
                       "actual_stl": float(scores["stl"][0]), "actual_goal_distance": -float(scores["terminal"][0]),
                       "execution_prediction_cosine_error": float(1-F.cosine_similarity(torch.as_tensor(predicted[0,len(segment)],device=args.device),actual_segment[-1],dim=0)),
                       "selected_source": plan["selected_source"]}
                write_json(folder / f"cycle_{cycle:03d}.json", {**plan, **row,
                           "optimized_action_sequence": proposed.tolist(), "executed_actions": np.asarray(segment and actions[-len(segment):]).tolist()})
                cycles.append(row)
                status.update(scene_seed=scene_seed, cycle=cycle+1, last_cycle=row, elapsed_seconds=time.time()-started)
                atomic_json(output / "status.json", status)
                if (cycle+1) % 10 == 0:
                    print(json.dumps(json_safe(row)), flush=True)
                    save_trajectory(folder,frames,actions,diagnostics,fps=metadata["frame_freq_hz"],name="latest")
                if done:break
            save_trajectory(folder, frames, actions, diagnostics, fps=metadata["frame_freq_hz"])
            np.savez_compressed(folder / "actual_latent_trace.npz", latent=history.cpu().numpy(),
                                **{name: value.cpu().numpy() for name, value in scores.items()})
            summary = {"scene_seed": scene_seed, "reference": chosen,
                       "environment_steps": len(actions), "cycles": len(cycles),
                       "actual_stl": float(scores["stl"][0]), **physical_summary(diagnostics),
                       "final_progress": guide.progress,
                       "selected_sources": {s:sum(r["selected_source"]==s for r in cycles) for s in {r["selected_source"] for r in cycles}},
                       "video": str((folder / "trajectory.mp4").resolve())}
            write_json(folder / "summary.json", summary)
            status["completed_episodes"].append(summary)
            atomic_json(output / "status.json", status)
            print(json.dumps(json_safe(summary)),flush=True)
        status.update(status="completed", elapsed_seconds=time.time()-started)
        atomic_json(output / "status.json", status)
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        atomic_json(output / "status.json", status)
        raise
    finally:
        env.close()
    return status


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "references", "action-prior", "output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--mode",choices=["cem","prior"],default="cem")
    p.add_argument("--tracking",choices=["visual","clock","hybrid"],default="visual")
    p.add_argument("--objective",choices=["safe_goal","goal_only"],default="safe_goal")
    p.add_argument("--surrogate",choices=["waypoint","final_goal"],default="waypoint")
    p.add_argument("--device",default="cuda");p.add_argument("--render-gpu",type=int,default=0)
    p.add_argument("--scene-seeds",nargs="+",type=int,default=[902100,902101,902102])
    p.add_argument("--seed",type=int,default=624);p.add_argument("--episode",type=int)
    p.add_argument("--steps",type=int,default=120);p.add_argument("--execute-steps",type=int,default=2)
    p.add_argument("--horizon",type=int,default=8);p.add_argument("--lookahead",type=int,default=8)
    p.add_argument("--population",type=int,default=64);p.add_argument("--iterations",type=int,default=3)
    p.add_argument("--elites",type=int,default=8);p.add_argument("--chunk-size",type=int,default=32)
    p.add_argument("--std",type=float,default=.03);p.add_argument("--min-std",type=float,default=.005)
    p.add_argument("--translation-bound",type=float,default=1.0)
    p.add_argument("--smoothness",type=float,default=.001)
    p.add_argument("--deviation-weight",type=float,default=1.0)
    p.add_argument("--maximum-advance",type=int,default=8);p.add_argument("--clock-weight",type=float,default=.003)
    p.add_argument("--scenario",default="baseline")
    p.add_argument("--evaluation-role",choices=["development","heldout"],default="development")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
