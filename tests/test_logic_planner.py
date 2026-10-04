import math
import pytest
import torch
from latent_stl_lewm.logic import (always, eventually, conjunction, disjunction, evaluate_formula,
    predicate_reach, predicate_preserve, predicate_avoid, latent_distance, smooth_min, smooth_max, PredicateConfig)
from latent_stl_lewm.planner import CEMConfig, plan_cem
from latent_stl_lewm.evaluate import classification, agreement


def test_exact_inclusive_windows_and_boolean():
    x = torch.tensor([[3., -2., 1., 4.], [0., 4., -1., 2.]])
    assert torch.equal(always(x, 1, 2), torch.tensor([-2., -1.]))
    assert torch.equal(eventually(x, 1, 2), torch.tensor([1., 4.]))
    assert torch.equal(always(x, 2, 2), x[:, 2])
    assert torch.equal(conjunction(x[:, 0], x[:, 1]), torch.tensor([-2., 0.]))
    assert torch.equal(disjunction(x[:, 0], x[:, 1]), torch.tensor([3., 4.]))
    for window in ((-1, 2), (2, 1), (0, 4)):
        with pytest.raises(ValueError):
            always(x, *window)


def test_smooth_extrema_bounds_limit_and_gradients():
    x = torch.tensor([0.3, -0.5, 0.8], requires_grad=True)
    for tau in (.1, .01, .001):
        lo, hi = smooth_min(x, tau), smooth_max(x, tau)
        assert x.min() - tau*math.log(len(x)) - 1e-6 <= lo <= x.min() + 1e-6
        assert x.max() - 1e-6 <= hi <= x.max() + tau*math.log(len(x)) + 1e-6
    assert torch.allclose(smooth_min(x, .001), x.min(), atol=.001)
    smooth_min(x, .1).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):
        always(x, semantics="automatic")


def test_predicate_signs_and_token_mask():
    goal = torch.tensor([1., 0.])
    assert predicate_reach(goal, goal, .1) > 0
    assert predicate_preserve(-goal, goal, .1) < 0
    assert predicate_avoid(-goal, goal, .1) > 0
    assert predicate_avoid(goal, goal, .1) < 0
    a, b = torch.tensor([[1., 0.], [1., 0.]]), torch.tensor([[1., 0.], [-1., 0.]])
    assert latent_distance(a, b, "mean_token_cosine", feature_ndim=2) == 1
    assert latent_distance(a, b, "mean_token_cosine", feature_ndim=2, token_mask=[1, 0]) == 0
    assert latent_distance(a, b, "cosine", feature_ndim=2, token_mask=[1, 0]) == 0
    with pytest.raises(ValueError):
        latent_distance(a, b, feature_ndim=2, token_mask=[0, 0])


def test_formula_composition_and_nested_temporal_rejection():
    pred = {"safe": torch.tensor([[1., .2, .5]]), "reach": torch.tensor([[-1., -.2, .3]])}
    formula = {"op": "and", "args": [{"op": "always", "arg": {"op": "predicate", "name": "safe"}}, {"op": "eventually", "arg": {"op": "predicate", "name": "reach"}}]}
    assert torch.allclose(evaluate_formula(formula, pred), torch.tensor([.2]))
    with pytest.raises(ValueError):
        evaluate_formula({"op": "always", "arg": formula}, pred)


class Integrator:
    def rollout(self, z0, actions):
        return torch.cat([z0[:, None], z0[:, None] + actions.cumsum(1)], dim=1)


def test_cem_direction_bounds_reproducibility_and_matched_budget():
    model, initial, goal, safe = Integrator(), torch.tensor([[0.]]), torch.tensor([1.]), torch.tensor([0.])
    config = CEMConfig(horizon=2, population=96, iterations=4, elites=12, seed=7)
    predicate = PredicateConfig(metric="l2", goal_threshold=.1, preserve_threshold=2.)
    runs = [plan_cem(model, initial, goal, safe, [-1.], [1.], objective=objective, config=config, predicate_config=predicate) for objective in ("terminal", "eventual", "stl")]
    for run in runs:
        assert run["candidate_evaluations"] == 384
        assert (run["actions"].abs() <= 1).all()
        positions = run["actions"].cumsum(0)
        assert (positions - goal).abs().min() < .1
    assert (runs[0]["actions"].sum() - 1.).abs() < .1
    assert runs[0]["iterations"][0]["candidate_actions"] == runs[1]["iterations"][0]["candidate_actions"]
    repeated = plan_cem(model, initial, goal, safe, [-1.], [1.], objective="terminal", config=config, predicate_config=predicate)
    assert torch.equal(repeated["actions"], runs[0]["actions"])


def test_metrics_handle_single_class_and_collapsed_predictions():
    result = classification([1., 1.], [True, True])
    assert result["balanced_accuracy"] is None and result["false_safe_rate"] is None
    result = classification([1., 1.], [True, False])
    assert result["auc"] == .5 and result["false_safe_rate"] == 1
    result = agreement([1., 1., 1.], [-1., 0., 1.])
    assert result["spearman"] is None


def test_mpc_gate_checks_hashes_horizon_and_explicit_ready(tmp_path):
    from latent_stl_lewm.evaluate import sha256
    from latent_stl_lewm.mpc import check_evidence
    checkpoint, dataset = tmp_path / "checkpoint", tmp_path / "dataset"
    checkpoint.write_bytes(b"model"); dataset.write_bytes(b"rgb/actions")
    evidence = {"checkpoint_sha256": sha256(checkpoint), "dataset_sha256": sha256(dataset),
                "mpc_gate": {"ready": True, "reasons": [], "credible_horizons": [4]}, "level2": {"horizon": 4}}
    assert check_evidence(evidence, checkpoint, dataset, 4, 1) == []
    checkpoint.write_bytes(b"different model")
    assert any("checkpoint hash" in reason for reason in check_evidence(evidence, checkpoint, dataset, 4, 1))
    dataset.write_bytes(b"different observations")
    assert any("dataset hash" in reason for reason in check_evidence(evidence, checkpoint, dataset, 4, 1))
    assert any("planning horizon" in reason for reason in check_evidence(evidence, checkpoint, dataset, 8, 1))
    assert any("Execute horizon" in reason for reason in check_evidence(evidence, checkpoint, dataset, 4, 5))
    evidence["mpc_gate"]["ready"] = False
    assert any("not certified" in reason for reason in check_evidence(evidence, checkpoint, dataset, 4, 1))


def test_past_visual_violation_cannot_be_erased_by_replanning():
    initial, goal, safe = torch.tensor([[0.]]), torch.tensor([1.]), torch.tensor([0.])
    result = plan_cem(Integrator(), initial, goal, safe, [-1.], [1.], objective="stl",
                      config=CEMConfig(horizon=2, population=24, iterations=2, elites=4),
                      predicate_config=PredicateConfig(metric="l2", goal_threshold=.1, preserve_threshold=.5),
                      history_latents=torch.tensor([[[2.]]]))
    assert result["score"] <= -1.5


def test_fixed_action_coordinate_respects_training_coverage():
    result = plan_cem(Integrator(), torch.tensor([[0.]]), torch.tensor([1.]), torch.tensor([0.]), [0.], [0.],
                      config=CEMConfig(horizon=2, population=8, iterations=1, elites=2),
                      predicate_config=PredicateConfig(metric="l2"))
    assert torch.equal(result["actions"], torch.zeros(2, 1))
