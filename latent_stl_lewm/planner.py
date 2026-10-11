"""Bounded CEM with identical budgets for terminal, eventual, and STL scores."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from .logic import PredicateConfig, score_trajectories


@dataclass
class CEMConfig:
    horizon: int = 8
    population: int = 128
    iterations: int = 4
    elites: int = 16
    seed: int = 0
    chunk_size: int = 32
    min_std: float = 0.05
    momentum: float = 0.1


@torch.no_grad()
def plan_cem(model, z0, goal, safe, action_low, action_high, *, objective="stl", config=None, predicate_config=None, history_latents=None, initial_mean=None,
             trajectory_scorer=None):
    config = config or CEMConfig()
    predicate_config = predicate_config or PredicateConfig()
    if objective not in ("terminal", "eventual", "stl"):
        raise ValueError("objective must be terminal, eventual, or stl")
    if trajectory_scorer is not None and (objective != "stl" or history_latents is not None):
        raise ValueError("Custom trajectory_scorer requires objective='stl' and owns its observed history")
    if z0.shape[0] != 1:
        raise ValueError("CEM plans from exactly one initial observation")
    if not 1 <= config.elites <= config.population or min(config.horizon, config.iterations, config.chunk_size) < 1:
        raise ValueError("Invalid CEM search budget")
    if history_latents is not None:
        if history_latents.shape[0] != 1 or history_latents.shape[2:] != z0.shape[1:]:
            raise ValueError("History must have shape 1,T,... matching the latent features")
        if predicate_config.end is not None:
            raise ValueError("History-aware MPC requires end=None for an expanding observed-prefix window")
    def with_history(latent):
        # History excludes the current observation, which rollout already contains.
        if history_latents is None or history_latents.shape[1] == 0:
            return latent
        prefix = history_latents.expand(latent.shape[0], *history_latents.shape[1:])
        return torch.cat([prefix, latent], dim=1)
    def score(latent):
        return (trajectory_scorer(latent) if trajectory_scorer is not None else
                score_trajectories(with_history(latent), goal, safe, predicate_config))
    device = z0.device
    low = torch.as_tensor(action_low, device=device, dtype=torch.float32)
    high = torch.as_tensor(action_high, device=device, dtype=torch.float32)
    if low.ndim != 1 or low.shape != high.shape or (low > high).any():
        raise ValueError("Action bounds must be matching A-vectors with low <= high (equal bounds fix an action coordinate)")
    generator = torch.Generator(device=device).manual_seed(config.seed)
    mean = ((low + high) / 2).expand(config.horizon, -1).clone()
    if initial_mean is not None:
        initial_mean = torch.as_tensor(initial_mean, device=device, dtype=torch.float32)
        if initial_mean.shape != mean.shape or not torch.isfinite(initial_mean).all():
            raise ValueError("Warm-start mean must be finite and match horizon/action dimensions")
        mean = initial_mean.clamp(low, high).clone()
    std = ((high - low) / 2).expand_as(mean).clone()
    best_score = -float("inf")
    best_actions = None
    progress = []
    for iteration in range(config.iterations):
        noise = torch.randn(config.population, *mean.shape, generator=generator, device=device)
        actions = (mean[None] + std[None] * noise).clamp(low, high)
        if initial_mean is not None:
            # Keep the shifted previous solution among the genuinely sampled candidates.
            actions[0] = mean
        values = []
        for chunk in actions.split(config.chunk_size):
            initial = z0.expand(chunk.shape[0], *z0.shape[1:])
            latent = model.rollout(initial, chunk)
            values.append(score(latent)[objective])
        scores = torch.cat(values)
        if scores.shape != (config.population,) or not torch.isfinite(scores).all():
            raise RuntimeError("Planner received invalid/nonfinite latent scores")
        elite_scores, indices = torch.topk(scores, config.elites)
        elite = actions[indices]
        mean = config.momentum * mean + (1 - config.momentum) * elite.mean(0)
        std = config.momentum * std + (1 - config.momentum) * elite.std(0, unbiased=False)
        std = std.clamp_min(config.min_std)
        if elite_scores[0].item() > best_score:
            best_score = elite_scores[0].item()
            best_actions = actions[indices[0]].clone()
        progress.append({"iteration": iteration, "best_score": elite_scores[0].item(), "mean_score": scores.mean().item(),
                         "score_std": scores.std(unbiased=False).item(),
                         "unique_scores": int(torch.unique(scores).numel()),
                         "candidate_actions": actions.cpu().tolist(), "candidate_scores": scores.cpu().tolist()})
    latent = model.rollout(z0, best_actions[None])
    scores = score(latent)
    return {"actions": best_actions, "score": best_score,
            "scores": {key: val.detach().cpu().tolist() for key, val in scores.items()},
            "iterations": progress, "config": asdict(config), "objective": objective,
            "predicate_config": predicate_config.to_dict() if trajectory_scorer is None else None,
            "scoring": "legacy_visual" if trajectory_scorer is None else "explicit_trajectory_scorer",
            "candidate_evaluations": config.population * config.iterations}
