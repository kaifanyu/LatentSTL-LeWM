import math

import pytest
import torch

from latent_stl_lewm.timed_stl import (
    STLMonitor,
    evaluate_timed_formula,
    formula_horizon,
    predicate_names,
)


P = {"op": "predicate", "name": "p"}
Q = {"op": "predicate", "name": "q"}


def temporal(op, interval, arg=P):
    return {"op": op, "interval": interval, "arg": arg}


def binary(op, interval, left=P, right=Q):
    return {"op": op, "interval": interval, "left": left, "right": right}


def evaluate(formula, values, times, **kwargs):
    predicates = {name: torch.as_tensor(value, dtype=torch.float64) for name, value in values.items()}
    return evaluate_timed_formula(formula, predicates, times, **kwargs)


def test_five_seconds_is_100_transitions_and_101_observations():
    times = torch.arange(101, dtype=torch.float64) * 0.05
    trace = -torch.ones(101, dtype=torch.float64)
    trace[-1] = 0.75
    formula = temporal("eventually", [0, 5])
    result = evaluate_timed_formula(formula, {"p": trace}, times)
    assert result.complete
    assert result.robustness == 0.75
    assert result.status == "satisfied"
    prefix = evaluate_timed_formula(formula, {"p": trace[:-1]}, times[:-1])
    assert not prefix.complete
    assert prefix.status == "pending"
    assert prefix.lower == -1
    assert prefix.upper == math.inf
    with pytest.raises(ValueError, match="unresolved"):
        _ = prefix.robustness


def test_prefix_can_prove_boolean_result_before_final_robustness():
    eventually = evaluate(temporal("eventually", [0, 5]), {"p": [-1, 2]}, [0, 1])
    always = evaluate(temporal("always", [0, 5]), {"p": [1, -2]}, [0, 1])
    assert not eventually.complete and eventually.status == "satisfied"
    assert eventually.lower == 2 and eventually.upper == math.inf
    assert not always.complete and always.status == "violated"
    assert always.lower == -math.inf and always.upper == -2


def test_nested_relative_dwell_and_horizon():
    # A two-second dwell must finish by t=5: start window is [0,3].
    formula = temporal("eventually", [0, 3], temporal("always", [0, 2]))
    values = {"p": [-2, -2, -2, 1, 2, 3]}
    result = evaluate(formula, values, [0, 1, 2, 3, 4, 5])
    assert formula_horizon(formula) == 5
    assert result.complete and result.robustness == 1
    partial = evaluate(formula, {"p": values["p"][:-1]}, [0, 1, 2, 3, 4])
    assert not partial.complete and partial.status == "pending"
    # Starting dwell by 5 is a different formula, with horizon 7.
    assert formula_horizon(temporal("eventually", [0, 5], temporal("always", [0, 2]))) == 7


def test_until_immediate_witness_does_not_require_left():
    result = evaluate(binary("until", [0, 2]), {"p": [-9, -9, -9], "q": [3, -4, -4]}, [0, 1, 2])
    assert result.robustness == 3


def test_until_left_holds_before_lower_bound_and_excludes_witness():
    formula = binary("until", [2, 2])
    failed = evaluate(formula, {"p": [-3, 4, -8], "q": [-1, -1, 5]}, [0, 1, 2])
    passed = evaluate(formula, {"p": [3, 4, -8], "q": [-1, -1, 5]}, [0, 1, 2])
    assert failed.robustness == -3
    assert passed.robustness == 3
    partial = evaluate(formula, {"p": [-3, 4], "q": [-1, -1]}, [0, 1])
    assert not partial.complete and partial.status == "violated"
    assert partial.upper == -3


@pytest.mark.parametrize("prefix_length", [2, 4])
def test_release_and_weak_until_agree_with_definitions(prefix_length):
    values = {"p": [1, -2, 3, 4][:prefix_length], "q": [-1, 2, -3, 1][:prefix_length]}
    times = [0, 1, 2, 3][:prefix_length]
    release = binary("release", [1, 3])
    dual = {"op": "not", "arg": binary("until", [1, 3], {"op": "not", "arg": P}, {"op": "not", "arg": Q})}
    weak = binary("weak_until", [1, 3])
    expanded = {"op": "or", "args": [binary("until", [1, 3]), temporal("always", [0, 3])]}
    for actual, expected in [(release, dual), (weak, expanded)]:
        a, b = evaluate(actual, values, times), evaluate(expected, values, times)
        assert torch.equal(a.lower, b.lower)
        assert torch.equal(a.upper, b.upper)


def test_zero_is_boundary_and_boolean_composition():
    assert evaluate(P, {"p": [0]}, [0]).status == "boundary"
    values = {"p": [2], "q": [-1]}
    implication = {"op": "implies", "left": P, "right": Q}
    equivalent = {"op": "iff", "left": P, "right": Q}
    assert evaluate(implication, values, [0]).robustness == -1
    assert evaluate(equivalent, values, [0]).robustness == -1
    assert evaluate(temporal("always", [0, 5]), {"p": [0]}, [0]).status == "pending"


def test_closed_float32_endpoints_and_explicit_evaluation_sample():
    times = torch.arange(11, dtype=torch.float32) * 0.1
    values = torch.arange(11, dtype=torch.float64)
    result = evaluate_timed_formula(temporal("eventually", [0.3, 0.3]), {"p": values}, times, evaluation_time=0.2)
    assert result.robustness == 5
    with pytest.raises(ValueError, match="observed timestamp"):
        evaluate_timed_formula(P, {"p": values}, times, evaluation_time=0.25)


def test_irregular_timestamps_use_seconds_instead_of_sample_indices():
    result = evaluate(temporal("eventually", [0, 1]), {"p": [-1, -2, 3]}, [0, 0.1, 2])
    assert result.complete and result.robustness == -1
    # An observed time after the window closes establishes an empty window.
    empty_f = evaluate(temporal("eventually", [0.5, 0.6]), {"p": [1, 2]}, [0, 1])
    empty_g = evaluate(temporal("always", [0.5, 0.6]), {"p": [1, 2]}, [0, 1])
    assert empty_f.robustness == -math.inf
    assert empty_g.robustness == math.inf
    unknown = evaluate(temporal("eventually", [0.5, 0.6]), {"p": [1]}, [0])
    assert unknown.status == "pending"


def test_constants_empty_boolean_identities_and_names():
    assert evaluate_timed_formula({"op": "true"}, {}, [0]).robustness == math.inf
    assert evaluate_timed_formula({"op": "false"}, {}, [0]).robustness == -math.inf
    assert evaluate_timed_formula({"op": "and", "args": []}, {}, [0]).robustness == math.inf
    assert evaluate_timed_formula({"op": "or", "args": []}, {}, [0]).robustness == -math.inf
    assert predicate_names(binary("until", [0, 2])) == {"p", "q"}


@pytest.mark.parametrize("formula,expected", [
    (temporal("always", [0, 5], {"op": "true"}), math.inf),
    (temporal("eventually", [0, 5], {"op": "false"}), -math.inf),
    (temporal("always", [2, 5], {"op": "true"}), math.inf),
    (temporal("eventually", [2, 5], {"op": "false"}), -math.inf),
    (binary("until", [2, 5], P, {"op": "false"}), -math.inf),
    (binary("release", [2, 5], P, {"op": "true"}), math.inf),
    (temporal("always", [0, 2], temporal("always", [0, 3], {"op": "true"})), math.inf),
    (temporal("always", [0, 2], temporal("eventually", [0, 3], {"op": "true"})), math.inf),
])
def test_constant_future_formulas_remain_exact_on_partial_traces(formula, expected):
    result = evaluate(formula, {"p": [1]}, [0])
    assert not result.complete
    assert result.robustness == expected


def test_unknown_future_sample_schedule_does_not_invent_witnesses():
    # A positive-offset window may have no future samples at all. A next
    # observed timestamp at t=6 closes [2,5] without adding any witness.
    formula = temporal("eventually", [2, 5], {"op": "true"})
    assert evaluate_timed_formula(formula, {}, [0]).status == "pending"
    assert evaluate_timed_formula(formula, {}, [0, 6]).robustness == -math.inf
    nested = temporal("always", [0, 2], temporal("eventually", [1, 3], {"op": "true"}))
    assert evaluate_timed_formula(nested, {}, [0]).status == "pending"


def test_multiple_batch_axes_preserve_gradients_and_status_shape():
    values = torch.tensor([[[1.0, 3.0, 2.0], [-1.0, -3.0, -2.0]], [[0.0, -1.0, -2.0], [2.0, 1.0, 3.0]]], requires_grad=True)
    result = evaluate_timed_formula(temporal("eventually", [0, 2]), {"p": values}, [0, 1, 2])
    assert result.lower.shape == (2, 2)
    assert result.status == [["satisfied", "violated"], ["boundary", "satisfied"]]
    result.robustness.sum().backward()
    assert torch.equal(values.grad, torch.tensor([[[0., 1., 0.], [1., 0., 0.]], [[1., 0., 0.], [0., 0., 1.]]]))


def test_persistent_monitor_does_not_reset_deadline():
    monitor = STLMonitor(temporal("eventually", [0, 5]), task_start=10)
    assert monitor.append({"p": torch.tensor([-1., -2.])}, [10, 12]).status == "pending"
    expired = monitor.append({"p": torch.tensor([-3., 8.])}, [15, 16])
    assert expired.complete and expired.status == "violated"
    assert expired.robustness == -1  # Success at t=16 misses original t=15 deadline.
    assert monitor.task_start == 10
    assert monitor.evaluate().robustness == -1
    with pytest.raises(ValueError, match="strictly after"):
        monitor.append({"p": torch.tensor([1.])}, [16])
    assert monitor.evaluate().robustness == -1


def test_appended_timestamp_dtype_cannot_change_a_completed_verdict():
    monitor = STLMonitor(temporal("eventually", [0, .3]))
    first = monitor.append({"p": torch.tensor([-1., -1., -1., 1.])},
                           torch.tensor([0., .1, .2, .3], dtype=torch.float32))
    assert first.complete and first.status == "satisfied"
    with pytest.raises(ValueError, match="timestamp dtype"):
        monitor.append({"p": torch.tensor([-1.])}, [.4])
    assert monitor.evaluate().status == "satisfied"
    assert monitor.append({"p": torch.tensor([-1.])}, torch.tensor([.4], dtype=torch.float32)).status == "satisfied"


def test_unrelated_later_timestamps_cannot_widen_an_expired_window():
    monitor = STLMonitor(temporal("eventually", [0, .3]))
    first = monitor.append({"p": torch.tensor([-1., 1.])}, torch.tensor([0., .301], dtype=torch.float32))
    assert first.complete and first.status == "violated"
    extended = monitor.append({"p": torch.tensor([-1.])}, torch.tensor([10000.], dtype=torch.float32))
    assert extended.complete and extended.status == "violated"
    torch.testing.assert_close(first.robustness, extended.robustness)


def test_unrelated_small_gap_cannot_shrink_a_past_endpoint_tolerance():
    monitor = STLMonitor(temporal("eventually", [0, .3]))
    monitor.append({"p": torch.tensor([-1., 1.])}, torch.tensor([0., .3], dtype=torch.float32))
    # Reject timestamps whose declared precision cannot resolve their gap;
    # leave all previous samples and their interpretation intact.
    with pytest.raises(ValueError, match="dtype precision"):
        monitor.append({"p": torch.tensor([-1.])}, torch.tensor([.30000004], dtype=torch.float32))
    assert monitor.evaluate().status == "satisfied"


def test_absolute_epoch_clock_preserves_closed_endpoint_after_extension():
    origin = 1_700_000_000.
    monitor = STLMonitor(temporal("eventually", [0, .15]), task_start=origin)
    first = monitor.append({"p": torch.tensor([-1., -1., -1., 1.])},
                           [origin, origin + .05, origin + .1, origin + .15])
    assert first.complete and first.status == "satisfied"
    later = monitor.append({"p": torch.tensor([-1.])}, [origin + 10000])
    assert later.status == "satisfied"
    torch.testing.assert_close(first.robustness, later.robustness)


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [0, math.nan], [0, math.inf], [], [[0, 1]]])
def test_invalid_timestamps(times):
    with pytest.raises(ValueError, match="timestamps"):
        evaluate_timed_formula(P, {"p": torch.tensor([1., 2.])}, times)


@pytest.mark.parametrize("formula", [
    temporal("eventually", [2, 1]), temporal("always", [-1, 2]),
    temporal("eventually", [0, math.inf]), temporal("always", [0, math.nan]),
    {"op": "eventually", "start": 0, "end": 2, "arg": P},
    {"op": "next", "arg": P}, {"op": "predicate"},
])
def test_invalid_formula(formula):
    with pytest.raises(ValueError):
        formula_horizon(formula)


def test_invalid_predicate_values_shapes_and_missing_names():
    for values in [torch.tensor([math.nan]), torch.tensor([math.inf]), torch.ones(2)]:
        with pytest.raises(ValueError):
            evaluate_timed_formula(P, {"p": values}, [0])
    with pytest.raises(ValueError, match="Missing"):
        evaluate_timed_formula(P, {}, [0])
    with pytest.raises(ValueError, match="identical"):
        evaluate_timed_formula(P, {"p": torch.ones(1), "q": torch.ones(2, 1)}, [0])


@pytest.mark.parametrize("formula", [
    temporal("eventually", [1, 3]),
    temporal("always", [1, 3]),
    temporal("eventually", [0, 1], temporal("always", [0, 2])),
    binary("until", [1, 3]), binary("release", [1, 3]), binary("weak_until", [1, 3]),
    {"op": "not", "arg": temporal("always", [0, 1], temporal("eventually", [0, 2]))},
])
def test_prefix_bounds_contain_all_tested_completions(formula):
    # Independent completion samples verify conservative bounds across nested
    # negation and until, including decisive and undecidable partial prefixes.
    generator = torch.Generator().manual_seed(19)
    full = {name: torch.randn(32, 4, generator=generator) for name in ("p", "q")}
    expected = evaluate_timed_formula(formula, full, [0, 1, 2, 3]).robustness
    for length in (1, 2, 3):
        prefix = evaluate_timed_formula(formula, {name: value[..., :length] for name, value in full.items()}, list(range(length)))
        assert torch.all(prefix.lower <= expected)
        assert torch.all(expected <= prefix.upper)


def test_absolute_python_timestamps_preserve_subsecond_resolution_across_chunks():
    start = 1700000000.0
    formula = temporal("eventually", [0, .1])
    monitor = STLMonitor(formula, task_start=start)
    result = monitor.append({"p": torch.tensor([-1., -1.])}, [start, start + .05])
    assert result.status == "pending" and not result.complete
    result = monitor.append({"p": torch.tensor([1.])}, [start + .1])
    assert result.complete and result.status == "satisfied"
    assert result.robustness.item() == 1.


@pytest.mark.parametrize("times", [[False, True], [0j, 1j]])
def test_python_boolean_and_complex_timestamps_are_rejected(times):
    with pytest.raises(ValueError, match="real seconds"):
        evaluate_timed_formula(P, {"p": torch.ones(2)}, times)
