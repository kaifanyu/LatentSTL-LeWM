from types import SimpleNamespace

import pytest
import torch

from latent_stl_lewm.check_stl import default_formula
from latent_stl_lewm.forecast_stl import HistoryConditionedRollout, TimedLatentScorer, rollout_from_history
from latent_stl_lewm.planner import CEMConfig, plan_cem
from latent_stl_lewm.visual_stl import VisualSTLChecker, canonical_reach_preserve_specs


class Integrator:
    latent_layout = "global"
    config = SimpleNamespace(history_size=3)

    def __init__(self):
        self.contexts = []

    def predict_next(self, states, controls):
        self.contexts.append((states.clone(), controls.clone()))
        return states[:, -1] + controls[:, -1]

    def rollout(self, z0, actions):
        return torch.cat((z0[:, None], z0[:, None] + actions.cumsum(1)), dim=1)


def checker(safety_threshold=1.):
    calibration = dict(safe_threshold=safety_threshold, goal_threshold=.03, safe_scale=1., goal_scale=1.)
    return VisualSTLChecker("global", canonical_reach_preserve_specs(calibration, method="global"))


def test_forecast_uses_raw_observed_context_and_action_alignment():
    model = Integrator()
    observed = torch.tensor([[2., 0.], [2., 1.], [2., 2.], [2., 3.]])
    previous_actions = torch.tensor([[0., 1.]]).expand(3, -1)
    candidates = torch.tensor([[[0., 2.], [0., -1.]], [[0., -2.], [0., 1.]]])
    predicted = rollout_from_history(model, observed, previous_actions, candidates)
    torch.testing.assert_close(predicted, torch.tensor([
        [[2., 3.], [2., 5.], [2., 4.]], [[2., 3.], [2., 1.], [2., 2.]]]))
    states, controls = model.contexts[0]
    torch.testing.assert_close(states[0], observed[-3:])
    torch.testing.assert_close(controls[0], torch.tensor([[0., 1.], [0., 1.], [0., 2.]]))
    adapter = HistoryConditionedRollout(model, observed, previous_actions)
    torch.testing.assert_close(adapter.rollout(observed[-1].expand(2, -1), candidates), predicted)
    with pytest.raises(ValueError, match="observed dynamics context"):
        adapter.rollout(torch.zeros(2, 2), candidates)


def test_deadline_is_not_reset_for_late_forecast_goal():
    observed = torch.tensor([[1., 0.], [1., 0.], [1., 0.]])
    formula = {"op": "eventually", "interval": [0., .1], "arg": {"op": "predicate", "name": "reach"}}
    scorer = TimedLatentScorer(checker(), formula, observed, [0., .05, .1],
                              {"initial": observed[0], "goal": torch.tensor([0., 1.])}, step_seconds=.05)
    candidate = torch.tensor([[[1., 0.], [0., 1.]]])
    result = scorer.evaluate(candidate)["result"]
    assert result.complete and result.status == ["violated"]
    assert scorer(candidate)["stl"].item() < 0


def test_short_prediction_exposes_unknown_instead_of_inventing_cem_value():
    observed = torch.tensor([[1., 0.]])
    scorer = TimedLatentScorer(checker(), default_formula(5.), observed, [0.],
                              {"initial": observed[0], "goal": observed[0]}, step_seconds=.05)
    candidate = observed[None].expand(1, 9, 2)
    result = scorer.evaluate(candidate)["result"]
    assert result.status == ["pending"] and not result.complete
    with pytest.raises(ValueError, match="unresolved"):
        scorer(candidate)


def test_history_violation_survives_future_recovery():
    observed = torch.tensor([[1., 1.], [1., 0.]])
    scorer = TimedLatentScorer(checker(.1), default_formula(.1), observed, [0., .05],
                              {"initial": torch.tensor([1., 0.]), "goal": observed[-1]}, step_seconds=.05)
    result = scorer(torch.tensor([[[1., 0.], [1., 0.]]]))
    assert result["stl"].item() < 0
    assert result["preserve_trace"].shape == (1, 3)
    with pytest.raises(ValueError, match="actual current"):
        scorer(torch.tensor([[[0., 1.], [1., 0.]]]))


def test_cem_actually_ranks_timestamped_formula_scores():
    current = torch.tensor([[1., 0.]])
    goal = torch.tensor([1., 1.])
    scorer = TimedLatentScorer(checker(), default_formula(.1), current, [0.],
                              {"initial": current[0], "goal": goal}, step_seconds=.05)
    result = plan_cem(Integrator(), current, goal, current[0], [0., -1.], [0., 1.],
                      config=CEMConfig(horizon=2, population=64, iterations=3, elites=8, seed=10),
                      trajectory_scorer=scorer)
    assert result["score"] > 0
    assert result["scoring"] == "explicit_trajectory_scorer"
    assert result["predicate_config"] is None
    assert result["candidate_evaluations"] == 192


def test_custom_scorer_rejects_duplicate_history_and_unrelated_objectives():
    with pytest.raises(ValueError, match="owns its observed history"):
        plan_cem(Integrator(), torch.tensor([[1., 0.]]), None, None, [-1., -1.], [1., 1.],
                 history_latents=torch.ones(1, 1, 2), trajectory_scorer=lambda x: x)
    with pytest.raises(ValueError, match="objective"):
        plan_cem(Integrator(), torch.tensor([[1., 0.]]), None, None, [-1., -1.], [1., 1.],
                 objective="terminal", trajectory_scorer=lambda x: x)
