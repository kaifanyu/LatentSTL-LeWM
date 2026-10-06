import numpy as np
import pytest
import torch

from latent_stl_lewm.reference_predicates import (
    ReferencePredicates, calibrate_threshold, diverse_reference_indices, encoder_feature_hash,
)


def scorer():
    return ReferencePredicates(torch.tensor([[1., 0.], [0., 1.]]),
                               torch.tensor([[-1., 0.], [0., -1.]]), .1, .2)


def test_two_distinct_safe_arm_poses_and_disturbed_references():
    predicate = scorer()
    # Distinct safe endpoints need not resemble one whole-image safe reference.
    z = torch.tensor([[[1., 0.], [0., 1.]], [[1., 0.], [-1., 0.]]])
    result = predicate.score(z, torch.tensor([0., 1.]))
    assert result["safe_trace"][0].min() > 0
    assert result["safe_trace"][1, -1] < 0
    assert result["stl"][0] > 0 and result["stl"][1] < 0


def test_past_and_current_violation_preserved_without_flat_future_scores():
    predicate = scorer()
    z = torch.tensor([[[-1., 0.], [1., 0.]], [[-1., 0.], [-1., 0.]]])
    history = torch.tensor([[0., -1.]])
    result = predicate.score(z, torch.tensor([1., 0.]), history=history)
    assert (result["stl"] < 0).all()
    assert result["future_safe_min"][0] > 0 > result["future_safe_min"][1]
    assert torch.allclose(result["future_safe_min"], result["safe_trace"][:, 1])
    assert result["goal_progress"][0] > result["goal_progress"][1]
    assert result["safe_trace"].shape == (2, 2)


def test_serialization_preserves_scores(tmp_path):
    predicate = scorer()
    predicate.metadata = {"reference_split": "train", "calibration_split": "val"}
    path = tmp_path / "references.pt"
    predicate.save(path)
    loaded = ReferencePredicates.from_artifact(path)
    z = torch.tensor([[[1., 0.], [0., 1.]]])
    for name, value in predicate.score(z, torch.tensor([0., 1.])).items():
        assert torch.equal(value, loaded.score(z, torch.tensor([0., 1.]))[name])
    assert loaded.metadata == predicate.metadata


def test_calibration_balances_classes_and_goal_constraint_is_explicit():
    scores = np.array([-.9, -.8, -.4, .1, .2, .3])
    labels = np.array([False, False, False, True, True, True])
    margin = calibrate_threshold(scores, labels)
    assert -.4 < margin <= .1
    assert np.array_equal(scores >= margin, labels)
    constrained = calibrate_threshold(scores, labels, maximum=-.5)
    assert constrained <= -.5
    with pytest.raises(ValueError, match="both physical classes"):
        calibrate_threshold(scores, np.ones(6, dtype=bool))


def test_prototypes_only_select_eligible_training_members_and_cover_diversity():
    z = torch.tensor([[1., 0.], [.99, .01], [0., 1.], [-1., 0.]])
    eligible = np.array([True, True, True, False])
    indices = diverse_reference_indices(z, eligible, 2)
    assert eligible[indices].all()
    assert 2 in indices and (0 in indices or 1 in indices)
    assert len(set(indices)) == 2


def test_invalid_shapes_and_empty_reference_banks_rejected():
    with pytest.raises(ValueError):
        ReferencePredicates(torch.zeros(0, 2), torch.ones(1, 2), 0., .1)
    with pytest.raises(ValueError):
        scorer().score(torch.ones(1, 1, 2), torch.ones(2))
    with pytest.raises(ValueError):
        scorer().score(torch.ones(2, 2, 2), torch.ones(2), history=torch.ones(3, 2, 2))


def test_feature_hash_accepts_predictor_update_but_rejects_encoder_change():
    module = torch.nn.Module()
    module.model = torch.nn.Module()
    module.model.encoder = torch.nn.Linear(2, 2)
    module.model.projector = torch.nn.Linear(2, 2)
    module.model.predictor = torch.nn.Linear(2, 2)
    module.register_buffer("image_mean", torch.tensor([.5]))
    before = encoder_feature_hash(module)
    with torch.no_grad():
        module.model.predictor.weight.add_(1)
    assert encoder_feature_hash(module) == before
    with torch.no_grad():
        module.model.encoder.weight.add_(1)
    assert encoder_feature_hash(module) != before
