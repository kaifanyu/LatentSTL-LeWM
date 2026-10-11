"""Check action-conditioned LeWM forecasts with an immutable STL task clock.

The forecast uses only observations/actions through the forecast origin plus
the proposed future actions. Predicted satisfaction concerns the model's trace,
not a certificate about physical execution. Only global LeWM dynamics exists.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import json
import math

import h5py
import numpy as np
import torch

from .check_stl import (default_formula, episode_timestamps, monitor_record,
                        physical_predicates, _metadata, _validate_artifact)
from .evaluate import sha256, write_json
from .timed_stl import evaluate_timed_formula, predicate_names
from .visual_stl import VisualSTLChecker, canonical_reach_preserve_specs


def rollout_from_history(model, observed_latents, observed_actions, candidate_actions):
    """Return B,H+1,D predictions from raw observed LeWM latents.

    Inputs: observed_latents[T,D], observed_actions[T-1,A], and candidate
    actions[B,H,A]. Includes the actual current state at index zero. Unlike
    normalizing visual descriptors, normalization here would change dynamics.
    """
    if getattr(model, "latent_layout", "global") != "global":
        raise ValueError("Only global latent dynamics is supported")
    if (observed_latents.ndim != 2 or len(observed_latents) < 1 or observed_actions.ndim != 2
            or candidate_actions.ndim != 3 or observed_actions.shape[0] != len(observed_latents) - 1
            or observed_actions.shape[-1] != candidate_actions.shape[-1] or candidate_actions.shape[0] < 1):
        raise ValueError("Expected observed z[T,D], actions[T-1,A], candidates[B,H,A]")
    values = (observed_latents, observed_actions, candidate_actions)
    if any(not v.is_floating_point() or not torch.isfinite(v).all() for v in values):
        raise ValueError("Latents and actions must be finite floating point tensors")
    if any(v.device != observed_latents.device for v in values):
        raise ValueError("Latents and actions must share a device")
    batch = candidate_actions.shape[0]
    history_size = int(model.config.history_size)
    if history_size < 1:
        raise ValueError("Model history_size must be positive")
    states = [z.expand(batch, -1) for z in observed_latents[-history_size:]]
    controls = [u.expand(batch, -1) for u in observed_actions[-(len(states)-1):]] if len(states) > 1 else []
    predicted = [states[-1]]
    for action in candidate_actions.unbind(1):
        controls.append(action)
        length = min(len(states), history_size)
        successor = model.predict_next(torch.stack(states[-length:], 1), torch.stack(controls[-length:], 1))
        states.append(successor)
        predicted.append(successor)
    return torch.stack(predicted, dim=1)


class HistoryConditionedRollout:
    """Adapt observed LeWM state/action history to the CEM rollout interface.

    Construct alongside TimedLatentScorer at each replan. The scorer remembers
    specification history; this adapter separately preserves dynamics context.
    """

    def __init__(self, model, observed_latents, observed_actions):
        self.model = model
        self.observed_latents = observed_latents
        self.observed_actions = observed_actions

    def rollout(self, z0, actions):
        if z0.shape != (actions.shape[0], self.observed_latents.shape[-1]):
            raise ValueError("Initial candidate states must match batch and feature dimensions")
        if not torch.allclose(z0, self.observed_latents[-1].expand_as(z0)):
            raise ValueError("Initial states must match the observed dynamics context")
        return rollout_from_history(self.model, self.observed_latents, self.observed_actions, actions)


class TimedLatentScorer:
    """Reusable exact CEM scorer retaining the observed prefix and fixed refs.

    ``evaluate`` exposes partial-prefix bounds. ``__call__`` is the strict
    planner adapter: unresolved robustness raises instead of inventing a
    terminal value. Pass it as ``trajectory_scorer`` to ``plan_cem``. Clock
    durations describe individual predicted transitions, not action blocks.
    """

    def __init__(self, checker, formula, observed_latents, timestamps, references, *, step_seconds):
        if checker.method != "global":
            raise ValueError("Current LeWM forecasts contain global features only")
        if not math.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("step_seconds must be finite and positive")
        self.checker, self.formula, self.references = checker, formula, references
        self.observed = torch.as_tensor(observed_latents)
        self.timestamps = torch.as_tensor(timestamps, dtype=torch.float64, device="cpu")
        if self.observed.ndim != 2 or len(self.observed) != len(self.timestamps):
            raise ValueError("Observed latents[T,D] must align with timestamps[T]")
        # Validate the entire observed prefix before accepting candidate plans.
        checker.check_features(self.observed, self.timestamps, formula, references=references,
                               source="observed_global_latents")
        self.step_seconds = float(step_seconds)

    def evaluate(self, rollout):
        if rollout.ndim != 3 or rollout.shape[1] < 1 or rollout.shape[-1] != self.observed.shape[-1]:
            raise ValueError("Expected rollout[B,H+1,D] including the current observation")
        observed = self.observed.to(device=rollout.device, dtype=rollout.dtype)
        if not torch.allclose(rollout[:, 0], observed[-1].expand(rollout.shape[0], -1)):
            raise ValueError("Rollout must begin at the actual current latent")
        prefix = observed[:-1].unsqueeze(0).expand(rollout.shape[0], -1, -1)
        joined = torch.cat((prefix, rollout), dim=1)
        future = self.timestamps[-1] + self.step_seconds * torch.arange(1, rollout.shape[1], dtype=torch.float64)
        times = torch.cat((self.timestamps, future))
        return self.checker.check_features(joined, times, self.formula, references=self.references,
                                           source="observed_prefix_and_predicted_global_future")

    def __call__(self, rollout):
        checked = self.evaluate(rollout)
        result = checked["result"]
        score = result.robustness  # Missing task horizon is not a finite CEM objective.
        return {"stl": score, **{name + "_trace": value for name, value in checked["predicate_traces"].items()}}


@torch.inference_mode()
def run(args):
    from .world_model import load_checkpoint
    if args.start < 0 or args.horizon < 1:
        raise ValueError("start must be nonnegative and horizon positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    artifact = json.loads(Path(args.calibration_artifact).read_text())
    dataset_hash, checkpoint_hash = sha256(args.dataset), sha256(args.checkpoint)
    _validate_artifact(artifact, dataset_hash, checkpoint_hash, ["global"])
    formula = json.loads(Path(args.formula).read_text()) if args.formula else default_formula(args.deadline, args.hold)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    thresholds = artifact["methods"]["global"]["thresholds"]
    specs = canonical_reach_preserve_specs(thresholds, method="global")
    if "safe" in predicate_names(formula):
        specs.append(replace(specs[0], name="safe"))
    checker = VisualSTLChecker("global", specs)
    with h5py.File(args.dataset, "r") as dataset:
        frequency, config = _metadata(dataset)
        if args.episode not in dataset["trajectories"]:
            raise ValueError(f"Unknown episode {args.episode}")
        episode = dataset["trajectories"][args.episode]
        split = episode.attrs["split"]
        if split not in ("test", b"test"):
            raise ValueError("Forecast audit requires a held-out test episode")
        if args.episode in artifact.get("validation_episode_ids", []):
            raise ValueError("Forecast episode overlaps calibration")
        timestamps = episode_timestamps(episode, frequency)
        if not np.allclose(np.diff(timestamps), 1 / frequency, rtol=0, atol=1e-10):
            raise ValueError("LeWM audit requires one observation per raw control transition")
        end = args.start + args.horizon
        if end >= len(timestamps):
            raise ValueError("Requested forecast exceeds recorded action/observation coverage")

        def encode(rgb):
            chunks = []
            for begin in range(0, len(rgb), 32):
                batch = torch.as_tensor(np.asarray(rgb[begin:begin+32]), device=args.device).permute(0, 3, 1, 2)
                with torch.autocast(torch.device(args.device).type, dtype=torch.float16,
                                    enabled=torch.device(args.device).type == "cuda"):
                    chunks.append(model.encode(batch).float())
            return torch.cat(chunks)

        # Only the observed prefix is encoded before producing the forecast.
        observed = encode(episode["rgb"][:args.start+1])
        goal = encode(dataset["references/goal_rgb"][:][None])[0]
        actions = torch.as_tensor(episode["action"][:end], device=args.device)
        with torch.autocast(torch.device(args.device).type, dtype=torch.float16,
                            enabled=torch.device(args.device).type == "cuda"):
            predicted = rollout_from_history(model, observed, actions[:args.start],
                                              actions[None, args.start:end]).float()
        references = {"initial": observed[0], "goal": goal}
        scorer = TimedLatentScorer(checker, formula, observed, timestamps[:args.start+1],
                                   references, step_seconds=1 / frequency)
        forecast = scorer.evaluate(predicted)
        # Actual future images enter only the independent comparison below.
        actual_future = encode(episode["rgb"][args.start+1:end+1])
        actual = torch.cat((observed, actual_future))
        checked = checker.check_features(actual, timestamps[:end+1], formula, references=references,
                                          source="observed_global_latents")
        physical, physical_kind = physical_predicates(episode, config)
        physical = {name: value[:end+1] for name, value in physical.items()}
        oracle = evaluate_timed_formula(formula, physical, timestamps[:end+1])
        def unbatch(result):
            from .timed_stl import MonitorResult
            return MonitorResult(result.lower[0], result.upper[0], result.complete)
        report = {"scope": "Recorded-action forecast audit; predicted success is not physical certification",
                  "dataset_sha256": dataset_hash, "checkpoint_sha256": checkpoint_hash,
                  "calibration_sha256": sha256(args.calibration_artifact), "episode": args.episode,
                  "task_start_seconds": float(timestamps[0]), "forecast_start_seconds": float(timestamps[args.start]),
                  "forecast_horizon_steps": args.horizon, "forecast_horizon_seconds": args.horizon / frequency,
                  "formula": formula, "timestamps_seconds": timestamps[:end+1].tolist(),
                  "predicted": monitor_record(unbatch(forecast["result"])),
                  "observed": monitor_record(checked["result"]),
                  "physical": monitor_record(oracle), "physical_score_kind": physical_kind,
                  "predicted_predicates": {key: value[0].cpu().tolist() for key, value in forecast["predicate_traces"].items()},
                  "observed_predicates": {key: value.cpu().tolist() for key, value in checked["predicate_traces"].items()},
                  "physical_predicates": {key: value.tolist() for key, value in physical.items()}}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("predicted", "observed", "physical")}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "checkpoint", "calibration-artifact", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--formula", type=Path)
    parser.add_argument("--deadline", type=float, default=5.)
    parser.add_argument("--hold", type=float, default=0.)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
