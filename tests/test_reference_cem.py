import pytest
import torch

from latent_stl_lewm.reference_cem import ReferenceCEMConfig, plan_reference_cem


class Integrator:
    def rollout(self, z0, actions):
        return torch.cat([z0[:, None], z0[:, None] + actions.cumsum(1)], dim=1)


class Predicates:
    def __init__(self, upper=2.0):
        self.upper = upper

    def score(self, latent, goal, history=None):
        safe = self.upper - latent[..., 0].abs()
        reach = .05 - (latent[..., 0] - goal[0]).abs()
        whole_safe, whole_reach = safe, reach
        if history is not None:
            h = history.expand(len(latent), -1, -1)
            whole_safe = torch.cat([self.upper - h[..., 0].abs(), safe], dim=1)
            whole_reach = torch.cat([.05 - (h[..., 0] - goal[0]).abs(), reach], dim=1)
        terminal = -(latent[:, -1, 0] - goal[0]).abs()
        return {"preserve_trace": safe, "reach_trace": reach,
                "stl": torch.minimum(whole_safe.amin(1), whole_reach.amax(1)),
                "future_safe_min": safe[:, 1:].amin(1),
                "terminal": terminal,
                "goal_progress": terminal + (latent[:, 0, 0] - goal[0]).abs()}


def run(predicates=None, **kwargs):
    return plan_reference_cem(Integrator(), torch.tensor([[0.]]), torch.tensor([1.]),
                              predicates or Predicates(), [-1.], [1.], **kwargs)


def test_historical_violation_stays_negative_but_does_not_flatten_selection():
    result = run(history_latents=torch.tensor([[[20.]]]),
                 config=ReferenceCEMConfig(horizon=2, population=64, iterations=4,
                                           elites=8, smoothness_weight=0, seed=3))
    assert result["scores"]["stl"] == [-18.0]
    assert all(it["unique_exact_stl"] == 1 for it in result["iterations"])
    assert all(it["unique_goal_utility"] > 1 for it in result["iterations"])
    assert abs(result["actions"].sum().item() - 1) < .05


def test_future_feasibility_dominates_better_goal_and_selected_actions_are_sampled():
    result = run(Predicates(upper=.5), initial_candidates=torch.tensor([[[1.]], [[.4]]]),
                 config=ReferenceCEMConfig(horizon=1, population=64, iterations=3,
                                           elites=8, smoothness_weight=0))
    assert result["scores"]["future_feasible"] == [True]
    assert .39 < result["actions"].item() <= .5
    selected = result["iterations"][result["selected_iteration"]]
    assert torch.equal(result["actions"], torch.tensor(selected["candidate_actions"][result["selected_index"]]))
    for iteration in result["iterations"]:
        assert iteration["candidate_sources"][2:4] == ["retrieved_prior"] * 2
        assert iteration["candidate_actions"][2:4] == [[[1.]], [[pytest.approx(.4)]]]


def test_all_future_infeasible_minimizes_violation_and_does_not_erase_current():
    result = plan_reference_cem(Integrator(), torch.tensor([[1.]]), torch.tensor([2.]),
                               Predicates(upper=.1), [-.2], [.2],
                               config=ReferenceCEMConfig(horizon=1, population=48,
                                                         iterations=3, elites=6))
    assert result["scores"]["future_feasible"] == [False]
    assert result["actions"].item() < -.15
    # Current state is unsafe even when the selected future improves safety.
    assert result["scores"]["stl"][0] <= -.9 + 1e-6


def test_bounds_repeat_determinism_and_candidate_accounting():
    kwargs = dict(config=ReferenceCEMConfig(horizon=2, population=16, iterations=2,
                                            elites=4, action_repeat=3),
                  initial_mean=torch.tensor([[4.], [-4.]]))
    first, second = run(**kwargs), run(**kwargs)
    assert torch.equal(first["actions"], second["actions"])
    assert first["actions"].shape == (6, 1)
    assert first["predicted_latents"].shape == (1, 7, 1)
    assert first["candidate_evaluations"] == 32
    assert (first["actions"].abs() <= 1).all()
    assert first["iterations"][0]["candidate_actions"][0] == [[1.], [-1.]]
    assert first["iterations"][0]["candidate_actions"][1] == [[0.], [0.]]
    fixed = plan_reference_cem(Integrator(), torch.tensor([[0.]]), torch.tensor([1.]),
                              Predicates(), [0.], [0.], config=kwargs["config"])
    assert torch.equal(fixed["actions"], torch.zeros(6, 1))


def test_goal_only_control_does_not_accidentally_apply_safety_constraint():
    result = run(Predicates(upper=.1), initial_candidates=torch.tensor([[[1.]]]),
                 config=ReferenceCEMConfig(horizon=1, population=16, iterations=1,
                                           elites=4, objective="goal_only"))
    assert result["actions"].item() == 1
    assert result["scores"]["future_feasible"] == [False]


def test_invalid_prior_or_nonfinite_scores_are_rejected():
    with pytest.raises(ValueError, match="Too many prior"):
        run(initial_candidates=torch.zeros(14, 2, 1),
            config=ReferenceCEMConfig(horizon=2, population=16))
    with pytest.raises(ValueError, match="finite"):
        run(initial_mean=torch.tensor([[float("nan")]]),
            config=ReferenceCEMConfig(horizon=1))
    class Invalid(Predicates):
        def score(self, *args, **kwargs):
            result = super().score(*args, **kwargs)
            result["stl"][:] = float("nan")
            return result
    with pytest.raises(RuntimeError, match="Nonfinite"):
        run(Invalid(), config=ReferenceCEMConfig(horizon=1, population=8, elites=2))


@pytest.mark.parametrize("history", [None, torch.empty(1, 0, 2)])
def test_reference_predicate_empty_history_neutral_values_are_preserved(history):
    from latent_stl_lewm.reference_predicates import ReferencePredicates
    predicates = ReferencePredicates(torch.tensor([[1., 0.], [0., 1.]]),
                                     torch.tensor([[-1., 0.], [0., -1.]]), .1, .2)
    result = plan_reference_cem(Integrator(), torch.tensor([[1., 0.]]),
                               torch.tensor([0., 1.]), predicates, [-1., -1.], [1., 1.],
                               history_latents=history,
                               config=ReferenceCEMConfig(horizon=2, population=8, elites=2))
    assert result["scores"]["history_safe_min"] == [float("inf")]
    assert result["scores"]["history_reach_max"] == [-float("inf")]
    assert torch.isfinite(torch.tensor(result["scores"]["stl"])).all()


@pytest.mark.parametrize("field,value,history", [
    ("history_safe_min", -float("inf"), None),
    ("history_reach_max", float("inf"), None),
    ("history_safe_min", float("nan"), None),
    ("history_safe_min", float("inf"), torch.zeros(1, 1, 1)),
    ("history_reach_max", -float("inf"), torch.zeros(1, 1, 1)),
    ("terminal", float("inf"), None),
])
def test_only_neutral_infinities_for_empty_history_are_allowed(field, value, history):
    class InvalidHistory(Predicates):
        def score(self, *args, **kwargs):
            result = super().score(*args, **kwargs)
            result[field] = torch.full_like(result["stl"], value)
            return result
    with pytest.raises(RuntimeError, match="Nonfinite"):
        run(InvalidHistory(), history_latents=history,
            config=ReferenceCEMConfig(horizon=1, population=8, elites=2))
