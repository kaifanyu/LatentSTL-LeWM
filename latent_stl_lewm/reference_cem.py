"""CEM with calibrated reference predicates and explicit future-action selection.

This is a safety/goal surrogate for action selection, not exact full-history STL
maximization. A historical violation remains in the reported exact STL score.
Candidate selection first requires predicted future safety, then rewards goal
progress. When no candidate is future-safe, it minimizes the worst violation.
No simulator state, hand-written target controller, or physical labels are read.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


@dataclass
class ReferenceCEMConfig:
    horizon: int = 8
    population: int = 128
    iterations: int = 4
    elites: int = 16
    seed: int = 0
    chunk_size: int = 32
    min_std: float = 0.05
    momentum: float = 0.1
    initial_std_scale: float = 0.5
    noise_correlation: float = 0.75
    smoothness_weight: float = 0.005
    action_cost_weight: float = 0.0
    feasible_threshold: float = 0.0
    action_repeat: int = 1
    amp: bool = True
    objective: str = "safe_goal"

    def validate(self):
        if min(self.horizon, self.iterations, self.chunk_size, self.action_repeat) < 1:
            raise ValueError("CEM horizon, iterations, chunk size, and repeat must be positive")
        if self.population < 3 or not 1 <= self.elites <= self.population:
            raise ValueError("CEM requires population >= 3 and 1 <= elites <= population")
        if not 0 <= self.momentum < 1 or not 0 <= self.noise_correlation < 1:
            raise ValueError("Momentum and noise correlation must be in [0, 1)")
        if min(self.min_std, self.initial_std_scale, self.smoothness_weight,
               self.action_cost_weight) < 0:
            raise ValueError("CEM standard deviations and cost weights must be nonnegative")
        if self.objective not in {"safe_goal", "goal_only"}:
            raise ValueError("Objective must be safe_goal or goal_only")
        for value in asdict(self).values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("CEM configuration values must be finite")


def _lexicographic_order(keys):
    """Descending stable lexicographic order, without arbitrary scale mixing."""
    order = torch.arange(keys.shape[0], device=keys.device)
    for column in reversed(range(keys.shape[1])):
        order = order[torch.argsort(keys[order, column], descending=True, stable=True)]
    return order


@torch.no_grad()
def plan_reference_cem(model, z0, goal, predicates, action_low, action_high, *,
                       config=None, history_latents=None, initial_mean=None,
                       initial_candidates=None, previous_action=None):
    """Select a genuinely evaluated candidate using image-only latent dynamics.

    ``predicates.score(latents, goal, history=history_latents)`` returns per-step
    preserve/reach traces, exact ``stl``, ``future_safe_min`` excluding t=0,
    ``goal_progress``, and ``terminal``. History excludes current z0. Reported
    exact STL includes the whole observed history, current, and predicted future.

    Action bounds must come from the training set; equal bounds fix coordinates.
    Optional ``initial_candidates[N,H,A]`` are an explicitly logged data prior.
    They are evaluated in every CEM iteration, leaving at least one Gaussian
    sample. ``horizon`` counts control blocks; returned actions repeat each block
    ``action_repeat`` times. Warm starts and priors have the control-block shape.

    The selection tuple is [future feasible, progress utility, safety] for safe
    candidates and [not feasible, worst safety, progress utility] otherwise.
    Consequently progress breaks safety ties even after historical violations.
    This is a predicted safety preference, not a hard simulator safety guarantee.
    """
    config = config or ReferenceCEMConfig()
    config.validate()
    if z0.ndim != 2 or z0.shape[0] != 1 or not torch.isfinite(z0).all():
        raise ValueError("z0 must have finite shape [1,D]")
    if history_latents is not None:
        if (history_latents.ndim != 3 or history_latents.shape[0] != 1
                or history_latents.shape[-1] != z0.shape[-1]
                or not torch.isfinite(history_latents).all()):
            raise ValueError("History must be finite [1,T,D] and exclude current z0")
    device = z0.device
    low = torch.as_tensor(action_low, device=device, dtype=torch.float32)
    high = torch.as_tensor(action_high, device=device, dtype=torch.float32)
    if (low.ndim != 1 or low.numel() == 0 or low.shape != high.shape
            or not torch.isfinite(low).all() or not torch.isfinite(high).all()
            or (low > high).any()):
        raise ValueError("Action bounds must be finite vectors with low <= high")
    shape = (config.horizon, low.numel())
    mean = ((low + high) / 2).expand(shape).clone()
    if initial_mean is not None:
        mean = torch.as_tensor(initial_mean, device=device, dtype=torch.float32)
        if mean.shape != shape or not torch.isfinite(mean).all():
            raise ValueError("Initial mean must have finite shape [H,A]")
        mean = mean.clamp(low, high).clone()
    priors = torch.empty((0, *shape), device=device)
    if initial_candidates is not None:
        priors = torch.as_tensor(initial_candidates, device=device, dtype=torch.float32)
        if priors.ndim != 3 or priors.shape[1:] != shape or not torch.isfinite(priors).all():
            raise ValueError("Initial candidates must have finite shape [N,H,A]")
        if priors.shape[0] > config.population - 3:
            raise ValueError("Too many prior candidates; reserve mean, no-op, and a Gaussian sample")
        priors = priors.clamp(low, high)
    if previous_action is not None:
        previous_action = torch.as_tensor(previous_action, device=device, dtype=torch.float32)
        if previous_action.shape != low.shape or not torch.isfinite(previous_action).all():
            raise ValueError("Previous action must have finite shape [A]")
    half_range = (high - low) / 2
    std = (half_range * config.initial_std_scale).expand_as(mean).clone()
    std_floor = torch.minimum(half_range, torch.full_like(half_range, config.min_std))
    movable = high > low
    scale = half_range.clamp_min(1e-8)
    no_op = torch.zeros_like(mean).clamp(low, high)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    progress = []
    best_key = None
    best_actions = None
    best_location = None
    selected_metrics = None
    required = ("stl", "future_safe_min", "terminal", "goal_progress")
    empty_history = history_latents is None or history_latents.shape[1] == 0

    def evaluate(actions):
        collected = {}
        for chunk in actions.split(config.chunk_size):
            physical_actions = chunk.repeat_interleave(config.action_repeat, dim=1)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=config.amp and device.type == "cuda"):
                latent = model.rollout(z0.expand(chunk.shape[0], -1), physical_actions)
            values = predicates.score(latent.float(), goal, history=history_latents)
            for name in required:
                if name not in values or values[name].shape != (len(chunk),):
                    raise RuntimeError(f"Predicate scorer must return {name}[B]")
            for name, value in values.items():
                if not isinstance(value, torch.Tensor) or value.shape[0] != len(chunk):
                    raise RuntimeError(f"Predicate score {name} must be a tensor with batch axis")
                valid = torch.isfinite(value)
                # Empty reductions use the exact STL neutral elements. These
                # diagnostics never enter candidate selection. No other field,
                # sign of infinity, or NaN is permitted through validation.
                if empty_history and name == "history_safe_min":
                    valid = valid | torch.isposinf(value)
                elif empty_history and name == "history_reach_max":
                    valid = valid | torch.isneginf(value)
                if not valid.all():
                    raise RuntimeError(f"Nonfinite predicate score: {name}")
                collected.setdefault(name, []).append(value.float())
        return {name: torch.cat(values) for name, values in collected.items()}

    for iteration in range(config.iterations):
        noise = torch.randn(config.population, *shape, device=device, generator=generator)
        if config.noise_correlation:
            innovation_scale = math.sqrt(1 - config.noise_correlation ** 2)
            for t in range(1, config.horizon):
                noise[:, t] = config.noise_correlation * noise[:, t - 1] + innovation_scale * noise[:, t]
        actions = (mean[None] + std[None] * noise).clamp(low, high)
        actions[0], actions[1] = mean, no_op
        actions[2:2 + len(priors)] = priors
        sources = ["cem_mean", "no_op"] + ["retrieved_prior"] * len(priors)
        sources += ["gaussian_sample"] * (config.population - len(sources))
        values = evaluate(actions)
        normalized = torch.where(movable, actions / scale, 0.)
        cost = normalized.square().mean((1, 2))
        delta = normalized[:, 1:] - normalized[:, :-1]
        if previous_action is not None:
            previous = torch.where(movable, previous_action / scale, 0.)
            delta = torch.cat([normalized[:, :1] - previous[None, None], delta], dim=1)
        smoothness = delta.square().mean((1, 2)) if delta.shape[1] else torch.zeros_like(cost)
        utility = values["goal_progress"] - config.smoothness_weight * smoothness - config.action_cost_weight * cost
        feasible = values["future_safe_min"] >= config.feasible_threshold
        if config.objective == "safe_goal":
            primary = torch.where(feasible, utility, values["future_safe_min"])
            secondary = torch.where(feasible, values["future_safe_min"], utility)
            keys = torch.stack([feasible.float(), primary, secondary], dim=-1)
        else:
            keys = torch.stack([torch.ones_like(utility), utility, values["future_safe_min"]], dim=-1)
        order = _lexicographic_order(keys)
        elite = actions[order[:config.elites]]
        winner = int(order[0])
        winner_key = tuple(keys[winner].cpu().tolist())
        if best_key is None or winner_key > best_key:
            best_key = winner_key
            best_actions = actions[winner].clone()
            best_location = (iteration, winner)
            selected_metrics = {name: value[winner:winner + 1].cpu().tolist() for name, value in values.items()}
            selected_metrics.update({"goal_utility": [float(utility[winner])],
                                     "future_feasible": [bool(feasible[winner])]})
        mean = config.momentum * mean + (1 - config.momentum) * elite.mean(0)
        std = config.momentum * std + (1 - config.momentum) * elite.std(0, unbiased=False)
        std = torch.maximum(std, std_floor).clamp_max(half_range)
        progress.append({
            "iteration": iteration, "selected_index": winner,
            "feasible_count": int(feasible.sum()),
            "unique_exact_stl": int(torch.unique(values["stl"]).numel()),
            "unique_goal_utility": int(torch.unique(utility).numel()),
            "selection_key": list(winner_key),
            "candidate_actions": actions.cpu().tolist(),
            "candidate_sources": sources,
            "candidate_scores": {name: value.cpu().tolist() for name, value in values.items()},
            "candidate_goal_utility": utility.cpu().tolist(),
            "candidate_feasible": feasible.cpu().tolist(),
            "candidate_selection_keys": keys.cpu().tolist(),
        })
    physical_actions = best_actions.repeat_interleave(config.action_repeat, dim=0)
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=config.amp and device.type == "cuda"):
        chosen_latents = model.rollout(z0, physical_actions[None])
    return {
        "actions": physical_actions, "control_actions": best_actions,
        "score": selected_metrics["goal_utility"][0], "scores": selected_metrics,
        "predicted_latents": chosen_latents.float().cpu(),
        "selected_iteration": best_location[0], "selected_index": best_location[1],
        "selected_source": progress[best_location[0]]["candidate_sources"][best_location[1]],
        "selection_key": list(best_key), "iterations": progress, "config": asdict(config),
        "candidate_evaluations": config.population * config.iterations,
        "selection_semantics": ("lexicographic future feasibility, goal progress minus control costs; "
                                "infeasible candidates ranked by worst future safety then goal progress; "
                                "exact full-history STL reported unchanged"
                                if config.objective == "safe_goal" else
                                "goal progress minus control costs; safety and exact full-history STL reported only"),
    }
