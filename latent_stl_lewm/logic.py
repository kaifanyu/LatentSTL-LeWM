"""Exact and explicitly smooth finite-trace STL over visual latents only.

Temporal windows are inclusive integer observation indices [start, end].  A
rollout with H actions has H+1 observations, so [0,H] includes the start.
Smooth extrema use unnormalised log-sum-exp: smooth_min <= min and
smooth_max >= max, with error bounded by tau*log(number of terms).
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Mapping
import torch
import torch.nn.functional as F


@dataclass
class PredicateConfig:
    metric: str = "cosine"
    goal_threshold: float = 0.02
    preserve_threshold: float = 0.01
    feature_ndim: int = 1
    semantics: str = "exact"
    tau: float = 0.01
    start: int = 0
    end: int | None = None

    def to_dict(self):
        return asdict(self)


def latent_distance(a, b, metric="cosine", *, feature_ndim=1, token_mask=None):
    """Distance retaining all batch/time axes; references broadcast naturally.

    cosine/l2/normalized_l2 flatten the explicitly declared feature axes.
    normalized_l2 is Euclidean distance between unit-normalized vectors.
    mean_token_cosine averages per-token cosine distances.  An optional mask
    weights tokens (zero-weight tokens are excluded), never simulator state.
    """
    if feature_ndim not in (1, 2):
        raise ValueError("feature_ndim must be 1 (vector) or 2 (tokens,features)")
    a, b = torch.broadcast_tensors(a, b)
    if token_mask is not None:
        if feature_ndim != 2:
            raise ValueError("token_mask requires feature_ndim=2")
        mask = torch.as_tensor(token_mask, device=a.device, dtype=a.dtype)
        if mask.ndim != 1 or mask.numel() != a.shape[-2] or (mask < 0).any() or mask.sum() <= 0:
            raise ValueError("token_mask must be a nonnegative N-vector with positive sum")
    else:
        mask = None
    if metric == "mean_token_cosine":
        if feature_ndim != 2:
            raise ValueError("mean_token_cosine requires spatial tokens, feature_ndim=2")
        d = 1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1)
        return d.mean(-1) if mask is None else (d * mask).sum(-1) / mask.sum()
    if mask is not None:
        # sqrt weight makes L2 energy, and cosine dot products, mask-weighted.
        a, b = a * mask.sqrt().unsqueeze(-1), b * mask.sqrt().unsqueeze(-1)
    if feature_ndim == 2:
        a, b = a.flatten(-2), b.flatten(-2)
    a, b = a.float(), b.float()
    if metric == "cosine":
        return 1.0 - F.cosine_similarity(a, b, dim=-1)
    if metric == "normalized_l2":
        return torch.linalg.vector_norm(F.normalize(a, dim=-1) - F.normalize(b, dim=-1), dim=-1)
    if metric == "l2":
        return torch.linalg.vector_norm(a - b, dim=-1)
    raise ValueError(f"Unknown latent distance {metric!r}")


def predicate_reach(z, reference, threshold, **distance_kwargs):
    return threshold - latent_distance(z, reference, **distance_kwargs)


def predicate_preserve(z, reference, threshold, **distance_kwargs):
    return threshold - latent_distance(z, reference, **distance_kwargs)


def predicate_avoid(z, reference, threshold, **distance_kwargs):
    return latent_distance(z, reference, **distance_kwargs) - threshold


def smooth_min(values, tau=0.01, dim=-1):
    if tau <= 0:
        raise ValueError("tau must be positive")
    return -tau * torch.logsumexp(-values / tau, dim=dim)


def smooth_max(values, tau=0.01, dim=-1):
    if tau <= 0:
        raise ValueError("tau must be positive")
    return tau * torch.logsumexp(values / tau, dim=dim)


def _reduce(values, minimum, semantics="exact", tau=0.01):
    if semantics == "exact":
        return values.amin(-1) if minimum else values.amax(-1)
    if semantics == "smooth":
        return smooth_min(values, tau) if minimum else smooth_max(values, tau)
    raise ValueError("semantics must explicitly be exact or smooth")


def _window(values, start, end):
    end = values.shape[-1] - 1 if end is None else end
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("STL windows require integer observation indices")
    if not 0 <= start <= end < values.shape[-1]:
        raise ValueError(f"Invalid inclusive window [{start},{end}] for length {values.shape[-1]}")
    return values[..., start:end + 1]


def always(values, start=0, end=None, *, semantics="exact", tau=0.01):
    return _reduce(_window(values, start, end), True, semantics, tau)


def eventually(values, start=0, end=None, *, semantics="exact", tau=0.01):
    return _reduce(_window(values, start, end), False, semantics, tau)


def conjunction(*values, semantics="exact", tau=0.01):
    if not values:
        raise ValueError("conjunction needs at least one argument")
    return _reduce(torch.stack(torch.broadcast_tensors(*values), -1), True, semantics, tau)


def disjunction(*values, semantics="exact", tau=0.01):
    if not values:
        raise ValueError("disjunction needs at least one argument")
    return _reduce(torch.stack(torch.broadcast_tensors(*values), -1), False, semantics, tau)


def evaluate_formula(formula: Mapping, predicates: Mapping[str, torch.Tensor], *, semantics="exact", tau=0.01):
    """Evaluate a small declarative AST; never evaluates code or language.

    Temporal operators take predicate traces and reduce their final time axis.
    This initial fragment permits Boolean composition of finite-window
    temporal clauses, not nested temporal operators with relative-time shifts.
    """
    def visit(node):
        op = node["op"]
        if op == "predicate":
            return predicates[node["name"]], True
        if op in ("always", "eventually"):
            value, is_trace = visit(node["arg"])
            if not is_trace:
                raise ValueError("Nested temporal operators are outside this finite-window fragment")
            fn = always if op == "always" else eventually
            return fn(value, node.get("start", 0), node.get("end"), semantics=semantics, tau=tau), False
        if op in ("and", "or"):
            children = [visit(child) for child in node["args"]]
            if not children or len(set(trace for _, trace in children)) != 1:
                raise ValueError("Boolean arguments must all be traces or all be scalar clauses")
            fn = conjunction if op == "and" else disjunction
            return fn(*(value for value, _ in children), semantics=semantics, tau=tau), children[0][1]
        if op == "not":
            value, is_trace = visit(node["arg"])
            return -value, is_trace
        raise ValueError(f"Unknown STL operator {op!r}")
    return visit(formula)[0]


def score_trajectories(z, goal, safe, config: PredicateConfig, *, goal_mask=None, preserve_mask=None):
    """Return B-shaped candidate scores and B,T predicate traces.

    Inputs have shape (...,T,D), or (...,T,N,D) for token features.
    All maximization objectives depend solely on visual latents.
    """
    kwargs = dict(metric=config.metric, feature_ndim=config.feature_ndim)
    reach = predicate_reach(z, goal, config.goal_threshold, token_mask=goal_mask, **kwargs)
    preserve = predicate_preserve(z, safe, config.preserve_threshold, token_mask=preserve_mask, **kwargs)
    temporal = dict(start=config.start, end=config.end, semantics=config.semantics, tau=config.tau)
    reach_score = eventually(reach, **temporal)
    preserve_score = always(preserve, **temporal)
    return {
        "reach_trace": reach, "preserve_trace": preserve,
        "terminal": reach[..., -1] - config.goal_threshold,
        "eventual": reach_score,
        "stl": conjunction(preserve_score, reach_score, semantics=config.semantics, tau=config.tau),
    }
