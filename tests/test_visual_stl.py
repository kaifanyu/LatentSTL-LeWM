import numpy as np
import pytest
import torch

from latent_stl_lewm.visual_predicate_benchmark import VisualFeatures
from latent_stl_lewm.visual_stl import (
    VisualPredicateSpec, VisualSTLChecker, canonical_reach_preserve_specs,
)


def _images():
    initial = np.zeros((128, 128, 3), dtype=np.uint8)
    initial[60:80, 55:65] = [120, 50, 50]
    goal = initial.copy()
    goal[40:60, 85:100] = [30, 170, 50]
    disturbed = goal.copy()
    disturbed[60:80, 55:65] = [0, 0, 0]
    return np.stack([initial, goal, disturbed]), goal


@pytest.mark.parametrize("method", ["pixels", "red_pixels"])
def test_calibration_matches_existing_visual_benchmark(method):
    rgb, goal = _images()
    cfg = {"safe_threshold": .1, "goal_threshold": .2,
           "safe_scale": .3, "goal_scale": .4}
    specs = canonical_reach_preserve_specs(cfg, method=method, initial_rgb=rgb[0])
    checker = VisualSTLChecker(method, specs, {"initial": rgb[0], "goal": goal})
    traces = checker.predicates_from_rgb(rgb)
    safe, reach, _, _ = VisualFeatures(method, None, "cpu").traces(rgb, goal)
    np.testing.assert_allclose(traces["preserve"], (.1 - safe) / .3, atol=1e-6)
    np.testing.assert_allclose(traces["reach"], (.2 - reach) / .4, atol=1e-6)
    assert not checker.metadata()["runtime_physical_labels"]


def test_image_features_and_timestamped_formula_have_identical_scores():
    rgb, goal = _images()
    checker = VisualSTLChecker("pixels", [VisualPredicateSpec("reach", threshold=.01)],
                               {"goal": goal})
    formula = {"op": "eventually", "interval": [0, .1],
               "arg": {"op": "predicate", "name": "reach"}}
    direct = checker.check_rgb(rgb, [0, .05, .1], formula)
    encoded = checker.check_features(checker.encode_rgb(rgb), [0, .05, .1], formula)
    torch.testing.assert_close(direct["result"].robustness, encoded["result"].robustness)
    assert direct["result"].status == "satisfied"
    # A truncated recording does not silently shorten a five-second deadline.
    formula["interval"] = [0, 5]
    partial = checker.check_rgb(rgb[:1], [0], formula)["result"]
    assert not partial.complete
    assert partial.status == "pending"


def test_reference_remains_fixed_across_chunks_and_float_rgb_is_supported():
    rgb, _ = _images()
    checker = VisualSTLChecker("pixels", [VisualPredicateSpec("preserve", "preserve", "initial", .01)],
                               {"initial": rgb[0]})
    whole = checker.predicates_from_rgb(rgb)["preserve"]
    chunk = checker.predicates_from_rgb(rgb[1:].astype(np.float32) / 255)["preserve"]
    torch.testing.assert_close(whole[1:], chunk)
    assert chunk[-1] < 0
    missing = VisualSTLChecker("pixels", checker.predicates)
    with pytest.raises(ValueError, match="Missing explicit reference"):
        missing.predicates_from_rgb(rgb)


def test_predicted_global_latents_support_batches_and_gradients_without_checkpoint():
    checker = VisualSTLChecker("global", [VisualPredicateSpec("reach", threshold=.2)])
    predicted = torch.tensor([[[1., 0.], [.8, .6]], [[0., 1.], [.6, .8]]], requires_grad=True)
    margin = checker.predicates_from_features(predicted, {"goal": torch.tensor([1., 0.])})["reach"]
    torch.testing.assert_close(margin, torch.tensor([[.2, 0.], [-.8, -.2]]), atol=1e-6, rtol=1e-5)
    margin.sum().backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad.abs().sum() > 0


def test_rgb_global_encoding_retains_dynamics_scale_and_float_input_convention():
    from types import SimpleNamespace

    class Model:
        config = SimpleNamespace(image_size=2, patch_size=1)

        def encode(self, rgb):
            value = rgb.float() / 255 if rgb.dtype == torch.uint8 else rgb
            average = value.mean((1, 2, 3))
            return torch.stack([average * 3, average * 4], dim=-1)

    extractor = SimpleNamespace(model=Model())
    checker = VisualSTLChecker("global", [VisualPredicateSpec("reach")],
                               image_size=2, patch_size=1, feature_extractor=extractor)
    integer = checker.encode_rgb(np.full((1, 2, 2, 3), 255, np.uint8))
    floating = checker.encode_rgb(np.ones((1, 2, 2, 3), np.float32))
    torch.testing.assert_close(integer, torch.tensor([[3., 4.]]))
    torch.testing.assert_close(integer, floating)
    assert not integer.is_inference()


def test_spatial_and_pixel_masks_use_corresponding_object_region():
    spec = VisualPredicateSpec("object", threshold=.1, roi=(0, 0, 16, 16))
    checker = VisualSTLChecker("spatial", [spec])
    reference = torch.tensor([1., 0.]).repeat(64, 1)
    features = reference.repeat(2, 1, 1)
    features[0, 1] = torch.tensor([0., 1.])  # Outside the selected token.
    features[1, 0] = torch.tensor([0., 1.])
    scores = checker.predicates_from_features(features, {"goal": reference})["object"]
    torch.testing.assert_close(scores, torch.tensor([.1, -.9]))
    with pytest.raises(ValueError, match="dimensions"):
        checker.predicates_from_features(torch.ones(2, 192), {"goal": reference})
    mask = np.zeros((128, 128)); mask[:16, :16] = 1
    masked = VisualSTLChecker("pixels", [VisualPredicateSpec("object", mask=mask, threshold=.1)])
    pixels = torch.zeros(2, 128, 128, 3)
    pixels[0, 50:70] = 1
    pixels[1, :16, :16] = 1
    values = masked.predicates_from_features(pixels, {"goal": torch.zeros(128, 128, 3)})["object"]
    torch.testing.assert_close(values, torch.tensor([.1, -.9]))


def test_pixel_distance_gradient_is_finite_at_an_exact_reference_match():
    checker = VisualSTLChecker("pixels", [VisualPredicateSpec("reach", threshold=.1)], image_size=2)
    features = torch.zeros(1, 2, 2, 3, requires_grad=True)
    margin = checker.predicates_from_features(features, {"goal": torch.zeros(2, 2, 3)})["reach"]
    margin.sum().backward()
    assert torch.isfinite(features.grad).all()


def test_named_reference_far_relation_and_roi_fallback_provenance():
    checker = VisualSTLChecker("global", [VisualPredicateSpec("avoid_visual", "far", "hazard", .2)])
    values = checker.predicates_from_features(torch.eye(2), {"hazard": torch.tensor([1., 0.])})
    torch.testing.assert_close(values["avoid_visual"], torch.tensor([-.2, .8]))
    cfg = {"safe_threshold": .1, "goal_threshold": .2, "safe_scale": 1, "goal_scale": 1}
    specs = canonical_reach_preserve_specs(cfg, initial_rgb=np.zeros((128, 128, 3), np.uint8))
    metadata = VisualSTLChecker("pixels", specs).metadata()
    assert metadata["roi_fallback_predicates"] == ["preserve"]
    assert metadata["physical_units"] is None


@pytest.mark.parametrize("kwargs", [{"scale": 0}, {"scale": -1}, {"threshold": float("nan")},
                                    {"relation": "collision"}, {"roi": (2, 2, 1, 1)},
                                    {"mask": [0, 0]}])
def test_invalid_specs_fail(kwargs):
    with pytest.raises(ValueError):
        VisualPredicateSpec("test", **kwargs)


def test_invalid_shapes_ranges_and_global_regions_fail():
    with pytest.raises(ValueError, match="Global latents"):
        VisualSTLChecker("global", [VisualPredicateSpec("test", roi=(0, 0, 10, 10))])
    with pytest.raises(ValueError, match="beyond"):
        VisualSTLChecker("pixels", [VisualPredicateSpec("test", roi=(0, 0, 129, 10))])
    checker = VisualSTLChecker("global", [VisualPredicateSpec("test")])
    with pytest.raises(ValueError, match="nonzero"):
        checker.predicates_from_features(torch.zeros(3, 192), {"goal": torch.ones(192)})
    with pytest.raises(ValueError, match="finite"):
        checker.predicates_from_features(torch.full((3, 192), float("nan")), {"goal": torch.ones(192)})
    pixels = VisualSTLChecker("pixels", [VisualPredicateSpec("test")])
    with pytest.raises(ValueError, match="RGB must"):
        pixels.predicates_from_rgb(np.ones((1, 128, 128, 3), np.float32) * 255)
