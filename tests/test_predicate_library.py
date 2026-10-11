import math

import pytest
import torch

from latent_stl_lewm.predicate_library import (
    AtomicPredicate, backward_velocity, near, orientation, physical_mug_predicates,
    polytope, quaternion_angle, vector_range,
)


def test_geometry_margins_keep_physical_boundaries_and_batch_shape():
    positions = torch.tensor([[[0., 0.], [.02, 0.], [.05, 0.]]])
    margin = near(positions, [0., 0.], .03, scale=.03)
    assert margin.shape == (1, 3)
    torch.testing.assert_close(margin, torch.tensor([[1., 1/3, -2/3]]))
    torch.testing.assert_close(vector_range(positions, [-.03, -.03], [.03, .03]),
                               torch.tensor([[.03, .01, -.02]]))
    normals = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
    torch.testing.assert_close(polytope(positions, normals, [.03]*4),
                               torch.tensor([[.03, .01, -.02]]))


def test_orientation_is_sign_invariant_and_has_finite_identity_gradient():
    angles = torch.tensor([0., math.pi/2, math.pi])
    quaternions = torch.stack([torch.cos(angles/2), angles*0, angles*0,
                               torch.sin(angles/2)], -1)
    reference = torch.tensor([1., 0., 0., 0.])
    torch.testing.assert_close(quaternion_angle(quaternions, reference), angles)
    torch.testing.assert_close(quaternion_angle(-quaternions, reference), angles)
    q = reference.clone().requires_grad_()
    orientation(q, reference, .1).backward()
    assert torch.isfinite(q.grad).all()
    with pytest.raises(ValueError, match="Zero quaternion"):
        quaternion_angle(torch.zeros(4), reference)


def test_velocity_uses_elapsed_seconds_and_does_not_invent_initial_sample():
    velocity, timestamps = backward_velocity(torch.tensor([[0.], [.1], [.5]]), [0., .1, .3])
    torch.testing.assert_close(velocity, torch.tensor([[1.], [2.]]))
    torch.testing.assert_close(timestamps, torch.tensor([.1, .3], dtype=torch.float64))
    with pytest.raises(ValueError, match="strictly increasing"):
        backward_velocity(torch.zeros(3, 2), [0., 0., 1.])
    velocity, _ = backward_velocity(torch.tensor([[0.], [.1]]), [1700000000., 1700000000.05])
    torch.testing.assert_close(velocity, torch.tensor([[2.]]), atol=1e-5, rtol=1e-5)


def test_oracle_separates_translation_rotation_and_goal():
    traces = physical_mug_predicates([.02, .04, .01], [0., .004, 0.], [0., 0., 4.])
    assert traces["reach"].tolist()[0] > 0
    assert traces["reach"].tolist()[1] < 0
    torch.testing.assert_close(traces["preserve"], torch.tensor([1., -1., -1.]))
    assert traces["translation"][2] > 0 and traces["rotation"][2] < 0


def test_registry_and_invalid_inputs_fail_explicitly():
    atom = AtomicPredicate("stopped", "norm_limit", "velocity", {"limit": .1})
    assert atom.evaluate({"velocity": torch.tensor([[.05, 0.]])}).item() > 0
    with pytest.raises(ValueError, match="Unknown predicate"):
        AtomicPredicate("test", "magic_grasp", "x").evaluate({"x": torch.ones(2)})
    with pytest.raises(ValueError, match="finite"):
        near(torch.tensor([float("nan")]), [0.], 1.)
    with pytest.raises(ValueError, match="positive"):
        near(torch.zeros(2), [0., 0.], 1., scale=0.)
    with pytest.raises(ValueError, match="lower"):
        vector_range(torch.zeros(2), [1., 0.], [0., 1.])
