"""Exact bounded future STL over timestamped, sampled predicate signals.

Intervals are closed and measured in seconds relative to each evaluation
sample. No interpolation or continuous-time safety claim is made. Until uses
the exclusive witness convention: its left operand holds on [t, witness).
The sampled trace is a prefix: missing future samples have unbounded predicate
values, giving conservative robustness intervals rather than clipped windows.

``complete`` concerns the full syntactic horizon; a partial trace can already
establish satisfaction/violation without determining the final robustness.
This module deliberately leaves past-time, next-step and smooth semantics out
of its language. Existing index-based scoring lives separately in logic.py.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
from typing import Mapping

import torch


@dataclass(frozen=True)
class MonitorResult:
    """Conservative robustness bounds with differentiable tensor endpoints."""

    lower: torch.Tensor
    upper: torch.Tensor
    complete: bool

    @property
    def robustness(self) -> torch.Tensor:
        """Return the exact score; raise if any batch member is unresolved.

        Decisive Boolean status alone is not enough: an early successful
        eventually may still improve its numerical robustness in the future.
        """
        if not torch.equal(self.lower, self.upper):
            raise ValueError("Robustness is unresolved; inspect lower and upper bounds")
        return self.lower

    @property
    def status(self):
        """Return a scalar string, or lists preserving the batch dimensions.

        Zero robustness is called boundary, not robust satisfaction. Bounds
        touching zero without equality remain pending.
        """
        codes = torch.full_like(self.lower, 2, dtype=torch.int64)
        codes = torch.where(self.lower > 0, 0, codes)
        codes = torch.where(self.upper < 0, 1, codes)
        codes = torch.where((self.lower == 0) & (self.upper == 0), 3, codes)
        labels = ("satisfied", "violated", "pending", "boundary")

        def decode(value):
            return [decode(item) for item in value] if isinstance(value, list) else labels[value]

        return decode(codes.detach().cpu().tolist())


@dataclass(frozen=True)
class _Node:
    op: str
    children: tuple = ()
    name: str | None = None
    interval: tuple[float, float] | None = None
    horizon: float = 0.0


def _compile(formula: Mapping) -> _Node:
    active: set[int] = set()

    def visit(node):
        if not isinstance(node, Mapping):
            raise ValueError("Every formula node must be a mapping")
        if id(node) in active:
            raise ValueError("Formula must not contain cycles")
        active.add(id(node))
        op = node.get("op")
        if not isinstance(op, str):
            raise ValueError("Formula node requires a string op")
        interval = None
        name = None
        if op == "predicate":
            name = node.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Predicate name must be a nonempty string")
            children = ()
            allowed = {"op", "name"}
        elif op in {"true", "false"}:
            children = ()
            allowed = {"op"}
        elif op in {"not", "always", "eventually"}:
            if "arg" not in node:
                raise ValueError(f"{op} requires arg")
            children = (visit(node["arg"]),)
            allowed = {"op", "arg"}
        elif op in {"and", "or"}:
            args = node.get("args")
            if not isinstance(args, (list, tuple)):
                raise ValueError(f"{op} requires an args list")
            children = tuple(visit(arg) for arg in args)
            allowed = {"op", "args"}
        elif op in {"implies", "iff", "until", "release", "weak_until"}:
            if "left" not in node or "right" not in node:
                raise ValueError(f"{op} requires left and right")
            children = (visit(node["left"]), visit(node["right"]))
            allowed = {"op", "left", "right"}
        else:
            raise ValueError(f"Unsupported STL operator {op!r}")
        horizon = max((child.horizon for child in children), default=0.0)
        if op in {"always", "eventually", "until", "release", "weak_until"}:
            raw = node.get("interval")
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError(f"{op} requires interval [a, b] in seconds")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw):
                raise ValueError("Interval endpoints must be finite real numbers")
            a, b = map(float, raw)
            if not math.isfinite(a) or not math.isfinite(b) or not 0 <= a <= b:
                raise ValueError("Intervals require finite 0 <= a <= b")
            interval = a, b
            horizon += b
            if not math.isfinite(horizon):
                raise ValueError("Formula horizon must be finite")
            allowed.add("interval")
        extra = set(node) - allowed
        if extra:
            raise ValueError(f"Unexpected fields for {op}: {sorted(extra)}")
        active.remove(id(node))
        return _Node(op, children, name, interval, horizon)

    return visit(formula)


def formula_horizon(formula: Mapping) -> float:
    """Return a sufficient syntactic lookahead in seconds, validating the AST."""
    return _compile(formula).horizon


def predicate_names(formula: Mapping) -> set[str]:
    """Return all referenced predicate names, validating the AST."""
    def names(node):
        result = {node.name} if node.op == "predicate" else set()
        for child in node.children:
            result.update(names(child))
        return result
    return names(_compile(formula))


def _clock_epsilon(value, precision):
    """Local rounding allowance; unrelated samples must not change a window."""
    return 8 * precision * abs(value)


def _timestamps(timestamps):
    raw = torch.as_tensor(timestamps)
    if raw.ndim != 1 or raw.numel() == 0:
        raise ValueError("timestamps must be a nonempty one-dimensional sequence")
    if raw.is_complex() or raw.dtype == torch.bool:
        raise ValueError("timestamps must contain finite real seconds")
    if isinstance(timestamps, (list, tuple)):
        # Reconstruct from the original Python doubles, not the temporary
        # dtype probe: float32 can collapse neighboring absolute timestamps.
        raw = torch.as_tensor(timestamps, dtype=torch.float64)
    values = raw.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("timestamps must contain finite real seconds")
    gaps = values[1:] - values[:-1]
    if (gaps <= 0).any():
        raise ValueError("timestamps must be strictly increasing")
    precision = torch.finfo(raw.dtype).eps if raw.is_floating_point() else torch.finfo(torch.float64).eps
    # Reject an ambiguous clock instead of shrinking all comparison tolerances
    # when a new, tightly spaced sample arrives. The latter would change past
    # verdicts. Float64 timestamps are preferable for long absolute clocks.
    allowances = 8 * precision * values.abs()
    if gaps.numel() and (gaps <= 4 * torch.maximum(allowances[:-1], allowances[1:])).any():
        raise ValueError("Timestamp gaps are too small for their dtype precision; use float64 timestamps")
    return values.tolist(), precision


def evaluate_timed_formula(
    formula: Mapping,
    predicates: Mapping[str, torch.Tensor],
    timestamps,
    *,
    evaluation_time: float | None = None,
) -> MonitorResult:
    """Evaluate an STL AST on predicate tensors of identical shape ``(..., T)``.

    ``evaluation_time`` defaults to the first sample and must coincide with an
    observed timestamp. Windows without samples have max=-inf/min=+inf only
    when their end is observed; otherwise unobserved future stays unknown.
    Inputs must be finite; infinities are reserved for logic and unknown bounds.
    Unused supplied predicates are also validated. A predicate-free formula
    with an empty mapping returns a scalar CPU tensor.
    """
    node = _compile(formula)
    times, precision = _timestamps(timestamps)
    if not isinstance(predicates, Mapping):
        raise ValueError("predicates must be a mapping of names to traces")
    traces = {}
    prototype = None
    for name, value in predicates.items():
        value = torch.as_tensor(value)
        if value.ndim < 1 or value.shape[-1] != len(times):
            raise ValueError(f"Predicate {name!r} must have shape (..., {len(times)})")
        if value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"Predicate {name!r} must contain finite real margins")
        if not value.is_floating_point():
            value = value.to(torch.get_default_dtype())
        if not torch.isfinite(value).all():
            raise ValueError(f"Predicate {name!r} contains NaN or infinity")
        if prototype is not None and (value.shape != prototype.shape or value.device != prototype.device or value.dtype != prototype.dtype):
            raise ValueError("All predicate traces must have identical shape, dtype and device")
        prototype = value if prototype is None else prototype
        traces[name] = value
    missing = predicate_names(formula) - traces.keys()
    if missing:
        raise ValueError(f"Missing predicate traces: {sorted(missing)}")
    base = torch.zeros(()) if prototype is None else prototype.new_zeros(prototype.shape[:-1])
    plus = torch.full_like(base, math.inf)
    minus = -plus
    now = times[0] if evaluation_time is None else float(evaluation_time)
    if not math.isfinite(now):
        raise ValueError("evaluation_time must be finite")
    eps = _clock_epsilon(now, precision)
    index = bisect_left(times, now - eps)
    if index == len(times) or abs(times[index] - now) > eps:
        raise ValueError("evaluation_time must match an observed timestamp")
    now = times[index]
    cache = {}
    future_cache = {}

    def negate(pair):
        return -pair[1], -pair[0]

    def reduce(pairs, minimum):
        if not pairs:
            return (plus, plus) if minimum else (minus, minus)
        fn = torch.amin if minimum else torch.amax
        return tuple(fn(torch.stack([p[i] for p in pairs]), dim=0) for i in (0, 1))

    def window(i, interval):
        a, b = interval
        start, end = times[i] + a, times[i] + b
        start_eps = _clock_epsilon(start, precision)
        end_eps = _clock_epsilon(end, precision)
        return range(bisect_left(times, start - start_eps), bisect_right(times, end + end_eps)), end > times[-1] + end_eps

    def future_bounds(current):
        """Signal-independent bounds at an arbitrary future sample.

        Logical constants remain known in the future. Positive-offset windows
        can contain no samples because the future sample schedule is unknown;
        windows starting at zero always contain their evaluation sample.
        """
        key = id(current)
        if key in future_cache:
            return future_cache[key]
        op, children = current.op, current.children
        if op == "predicate":
            result = minus, plus
        elif op in {"true", "false"}:
            value = plus if op == "true" else minus
            result = value, value
        elif op == "not":
            result = negate(future_bounds(children[0]))
        elif op in {"and", "or"}:
            result = reduce([future_bounds(child) for child in children], op == "and")
        elif op in {"implies", "iff"}:
            left, right = (future_bounds(child) for child in children)
            forward = reduce([negate(left), right], False)
            result = forward if op == "implies" else reduce([forward, reduce([negate(right), left], False)], True)
        elif op in {"always", "eventually"}:
            lower, upper = future_bounds(children[0])
            if current.interval[0] == 0:
                result = lower, upper
            else:
                result = (lower, plus) if op == "always" else (minus, upper)
        else:
            left, right = (future_bounds(child) for child in children)
            if op == "release":
                left, right = negate(left), negate(right)
            # An immediate witness is permitted when a=0. Otherwise the
            # window might contain no samples, so strong until can be false.
            lower = right[0] if current.interval[0] == 0 else minus
            result = lower, right[1]
            if op == "release":
                result = negate(result)
            elif op == "weak_until":
                result = reduce([result, left], False)
        future_cache[key] = result
        return result

    def temporal(child, i, interval, minimum):
        indices, unknown = window(i, interval)
        pairs = [visit(child, j) for j in indices]
        if unknown:
            lower, upper = future_bounds(child)
            # The unseen part might have no samples; include the appropriate
            # empty-window identity, not a fabricated future observation.
            pairs.append((lower, plus) if minimum else (minus, upper))
        return reduce(pairs, minimum)

    def until(left, right, i, interval, dual=False):
        indices, unknown = window(i, interval)
        candidates = []
        prefix = (plus, plus)
        witnesses = set(indices)
        last = len(times) if unknown else (max(indices) + 1 if indices else i)
        for j in range(i, last):
            if j in witnesses:
                right_pair = visit(right, j)
                candidates.append(reduce([prefix, negate(right_pair) if dual else right_pair], True))
            left_pair = visit(left, j)
            prefix = reduce([prefix, negate(left_pair) if dual else left_pair], True)
        if unknown:
            # Any future witness must preserve left on every observed sample.
            right_pair = future_bounds(right)
            if dual:
                right_pair = negate(right_pair)
            candidates.append((minus, torch.minimum(prefix[1], right_pair[1])))
        result = reduce(candidates, False)
        return negate(result) if dual else result

    def visit(current, i):
        key = id(current), i
        if key in cache:
            return cache[key]
        op, children = current.op, current.children
        if op == "predicate":
            value = traces[current.name][..., i]
            result = value, value
        elif op in {"true", "false"}:
            value = plus if op == "true" else minus
            result = value, value
        elif op == "not":
            result = negate(visit(children[0], i))
        elif op in {"and", "or"}:
            result = reduce([visit(child, i) for child in children], op == "and")
        elif op in {"implies", "iff"}:
            left, right = (visit(child, i) for child in children)
            forward = reduce([negate(left), right], False)
            result = forward if op == "implies" else reduce([forward, reduce([negate(right), left], False)], True)
        elif op in {"always", "eventually"}:
            result = temporal(children[0], i, current.interval, op == "always")
        else:
            result = until(*children, i, current.interval, dual=op == "release")
            if op == "weak_until":
                result = reduce([result, temporal(children[0], i, (0.0, current.interval[1]), True)], False)
        cache[key] = result
        return result

    lower, upper = visit(node, index)
    endpoint = now + node.horizon
    return MonitorResult(lower, upper, endpoint <= times[-1] + _clock_epsilon(endpoint, precision))


class STLMonitor:
    """Append timestamped trace chunks while retaining one original task start.

    Chunks must have strictly increasing, nonoverlapping timestamps. The first
    chunk must include ``task_start`` (or it defaults to its first timestamp).
    All predicate names, batch dimensions, dtypes and devices stay consistent.
    Timestamp dtype also stays fixed: Python lists use float64, so convert
    float32 tensor chunks explicitly before mixing them with Python lists.
    Stored tensors retain autograd history; detach externally for long-running
    observation-only monitoring.
    """

    def __init__(self, formula: Mapping, *, task_start: float | None = None):
        from copy import deepcopy
        _compile(formula)
        if task_start is not None and not math.isfinite(float(task_start)):
            raise ValueError("task_start must be finite")
        self.formula = deepcopy(formula)
        self.task_start = None if task_start is None else float(task_start)
        self._timestamps = None
        self._predicates = None

    def append(self, predicates: Mapping[str, torch.Tensor], timestamps) -> MonitorResult:
        # Validate the incoming chunk independently before changing state.
        evaluate_timed_formula(self.formula, predicates, timestamps)
        new_times = (torch.as_tensor(timestamps, dtype=torch.float64) if isinstance(timestamps, (list, tuple)) else torch.as_tensor(timestamps)).detach().cpu()
        new_predicates = {name: torch.as_tensor(value) for name, value in predicates.items()}
        start = float(new_times[0]) if self.task_start is None else self.task_start
        if self._timestamps is not None:
            if new_times.dtype != self._timestamps.dtype:
                raise ValueError("Appended timestamp dtype must match the original clock precision")
            if set(new_predicates) != set(self._predicates):
                raise ValueError("Appended chunks must have the same predicate names")
            if new_times[0] <= self._timestamps[-1]:
                raise ValueError("Appended timestamps must be strictly after the existing trace")
            for name, value in new_predicates.items():
                old = self._predicates[name]
                if value.shape[:-1] != old.shape[:-1] or value.dtype != old.dtype or value.device != old.device:
                    raise ValueError("Appended predicate batch shape, dtype and device must match")
            new_times = torch.cat((self._timestamps, new_times))
            new_predicates = {name: torch.cat((self._predicates[name], value), dim=-1) for name, value in new_predicates.items()}
        result = evaluate_timed_formula(self.formula, new_predicates, new_times, evaluation_time=start)
        self._timestamps, self._predicates, self.task_start = new_times, new_predicates, start
        return result

    def evaluate(self) -> MonitorResult:
        if self._timestamps is None:
            raise ValueError("Append a trace before evaluating the monitor")
        return evaluate_timed_formula(self.formula, self._predicates, self._timestamps, evaluation_time=self.task_start)
