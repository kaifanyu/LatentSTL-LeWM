import pytest
import torch

from latent_stl_lewm.logic import evaluate_formula


def test_legacy_entrypoint_routes_explicit_seconds_to_compositional_monitor():
    formula = {"op": "eventually", "interval": [0, 1], "arg": {
        "op": "always", "interval": [0, 1], "arg": {"op": "predicate", "name": "goal"}}}
    result = evaluate_formula(formula, {"goal": torch.tensor([-1., 1., 1.])}, timestamps=[0., 1., 2.])
    assert result.complete and result.status == "satisfied"
    assert result.robustness.item() == 1
    with pytest.raises(ValueError, match="exact"):
        evaluate_formula(formula, {"goal": torch.ones(3)}, timestamps=[0., 1., 2.], semantics="smooth")
