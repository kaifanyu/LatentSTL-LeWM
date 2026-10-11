"""Signed atomic margins, independent of perception and temporal operators.

Every function returns a positive margin inside the specified condition. Physical
units belong to the supplied signal; a visual similarity is not a metric pose.
Positive scales normalize margins without moving their zero boundary. Nothing
in this module accesses a simulator, image, or checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping
import math

import torch


def _tensor(value, *, like=None):
    kwargs = {} if like is None else {"device": like.device, "dtype": like.dtype}
    value = torch.as_tensor(value, **kwargs)
    if not value.is_floating_point():
        value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError("Predicate inputs must be finite")
    return value


def _positive(value, name="scale", *, allow_zero=False):
    if not math.isfinite(float(value)) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def above(value, threshold, *, scale=1.0):
    value = _tensor(value)
    return (value - _tensor(threshold, like=value)) / _positive(scale)


def below(value, threshold, *, scale=1.0):
    return -above(value, threshold, scale=scale)


def within(value, lower, upper, *, scale=1.0):
    """Elementwise scalar interval; use vector_range to combine coordinates."""
    value = _tensor(value)
    lower, upper = _tensor(lower, like=value), _tensor(upper, like=value)
    if (lower > upper).any():
        raise ValueError("lower must not exceed upper")
    return torch.minimum(value - lower, upper - value) / _positive(scale)


def vector_range(value, lower, upper, *, scale=1.0):
    return within(value, lower, upper, scale=scale).amin(dim=-1)


def near(value, reference, tolerance, *, scale=1.0):
    """Position, alignment, or initial-position preservation; last axis is vector."""
    value = _tensor(value)
    reference = _tensor(reference, like=value)
    tolerance = _positive(tolerance, "tolerance", allow_zero=True)
    return (tolerance - torch.linalg.vector_norm(value - reference, dim=-1)) / _positive(scale)


def far(value, reference, clearance, *, scale=1.0):
    """Point separation, not whole-body collision avoidance."""
    return -near(value, reference, clearance, scale=scale)


def norm_limit(value, limit, *, scale=1.0):
    value = _tensor(value)
    return below(torch.linalg.vector_norm(value, dim=-1),
                 _positive(limit, "limit", allow_zero=True), scale=scale)


def polytope(value, normals, offsets, *, scale=1.0):
    """Signed normalized halfspace margins for A x <= b, combined by min."""
    value = _tensor(value)
    normals, offsets = _tensor(normals, like=value), _tensor(offsets, like=value)
    if normals.ndim != 2 or offsets.shape != normals.shape[:1] or value.shape[-1] != normals.shape[-1]:
        raise ValueError("Expected value[...,D], normals[M,D], offsets[M]")
    lengths = torch.linalg.vector_norm(normals, dim=-1)
    if normals.shape[0] == 0 or (lengths == 0).any():
        raise ValueError("Polytope needs nonzero halfspace normals")
    return ((offsets - value @ normals.T) / lengths).amin(-1) / _positive(scale)


def quaternion_angle(value, reference):
    """SO(3) distance in radians; q and -q denote the same orientation.

    Either common quaternion component order works if both inputs use the same
    convention. The atan2 form avoids acos's singular derivative at identity.
    """
    value = _tensor(value)
    reference = _tensor(reference, like=value)
    if value.shape[-1] != 4 or reference.shape[-1] != 4:
        raise ValueError("Quaternions need four components")
    value_norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    reference_norm = torch.linalg.vector_norm(reference, dim=-1, keepdim=True)
    if (value_norm == 0).any() or (reference_norm == 0).any():
        raise ValueError("Zero quaternion is not an orientation")
    value, reference = value / value_norm, reference / reference_norm
    reference = torch.where((value * reference).sum(-1, keepdim=True) < 0, -reference, reference)
    return 4 * torch.atan2(torch.linalg.vector_norm(value - reference, dim=-1),
                           torch.linalg.vector_norm(value + reference, dim=-1))


def orientation(value, reference, tolerance_rad, *, scale=1.0):
    return below(quaternion_angle(value, reference),
                 _positive(tolerance_rad, "tolerance_rad", allow_zero=True), scale=scale)


def clearance(value, minimum, *, scale=1.0):
    """Threshold an externally measured/estimated signed geometry clearance."""
    return above(value, _positive(minimum, "minimum", allow_zero=True), scale=scale)


def backward_velocity(positions, timestamps):
    """Return (velocity[...,T-1,D], timestamps[1:]); no invented initial velocity."""
    positions = _tensor(positions)
    times = torch.as_tensor(timestamps, device=positions.device, dtype=torch.float64)
    if not torch.isfinite(times).all():
        raise ValueError("Timestamps must be finite")
    if positions.ndim < 2 or times.ndim != 1 or positions.shape[-2] != len(times) or len(times) < 2:
        raise ValueError("Expected positions[...,T,D] and at least two timestamps[T]")
    delta = times.diff()
    if (delta <= 0).any():
        raise ValueError("Timestamps must be strictly increasing")
    velocity = positions.diff(dim=-2) / delta.to(positions.dtype).unsqueeze(-1)
    return velocity, times[1:]


PREDICATE_FUNCTIONS: dict[str, Callable] = {
    "above": above, "below": below, "within": within, "vector_range": vector_range,
    "near": near, "far": far, "norm_limit": norm_limit, "polytope": polytope,
    "orientation": orientation, "clearance": clearance,
}


@dataclass(frozen=True)
class AtomicPredicate:
    """Serializable recipe for an atomic margin over an explicitly named signal.

    For learned semantic scores use above(score, threshold). Grasp/contact
    semantics must be supplied by that estimator, not inferred from its name.
    Additional grounded families can be registered explicitly by Python callers.
    """

    name: str
    kind: str
    signal: str
    parameters: Mapping = field(default_factory=dict)

    def evaluate(self, signals: Mapping[str, torch.Tensor]):
        if not self.name or not self.signal:
            raise ValueError("Predicate name and signal must be nonempty")
        if self.kind not in PREDICATE_FUNCTIONS:
            raise ValueError(f"Unknown predicate kind {self.kind!r}")
        if self.signal not in signals:
            raise ValueError(f"Missing signal {self.signal!r}")
        return PREDICATE_FUNCTIONS[self.kind](signals[self.signal], **self.parameters)


def physical_mug_predicates(goal_error, object_displacement, object_rotation_deg, *,
                            goal_tolerance=0.03, translation_tolerance=0.002,
                            rotation_tolerance_deg=2.0):
    """Diagnostic oracle only: explicit physical margins, not visual inference.

    Inputs are scalar traces whose object values already represent the worst
    object, as in this repository's diagnostics. No input labels are consumed.
    """
    goal = _tensor(goal_error)
    position = _tensor(object_displacement, like=goal)
    rotation = _tensor(object_rotation_deg, like=goal)
    if goal.shape != position.shape or goal.shape != rotation.shape:
        raise ValueError("Physical diagnostic traces must have identical shapes")
    if (goal < 0).any() or (position < 0).any() or (rotation < 0).any():
        raise ValueError("Physical error magnitudes must be nonnegative")
    reach = below(goal, goal_tolerance, scale=_positive(goal_tolerance, "goal_tolerance"))
    translation = below(position, translation_tolerance,
                        scale=_positive(translation_tolerance, "translation_tolerance"))
    angular = below(rotation, rotation_tolerance_deg,
                    scale=_positive(rotation_tolerance_deg, "rotation_tolerance_deg"))
    return {"reach": reach, "translation": translation, "rotation": angular,
            "preserve": torch.minimum(translation, angular)}
