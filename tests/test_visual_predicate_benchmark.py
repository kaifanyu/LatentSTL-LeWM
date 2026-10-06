import numpy as np

from latent_stl_lewm.visual_predicate_benchmark import (
    calibration, detection, mug_box, temporal_scores, token_weights,
)


def test_stl_remembers_violation_even_after_recovery():
    safe = np.ones(30); safe[3] = -.2
    reach = -np.ones(30); reach[-12:] = .4
    scores = temporal_scores(safe, reach)
    assert scores["stl"] == -.2
    assert scores["reach_and_hold"] == -.2


def test_eventual_reach_is_distinct_from_sustained_reach():
    reach = -np.ones(30); reach[12] = .5
    scores = temporal_scores(np.ones(30), reach)
    assert scores["stl"] == .5
    assert scores["reach_and_hold"] == -1


def test_alarm_before_violation_does_not_hide_later_miss():
    r = detection(np.array([-1, 1, 1, 1]), [True, True, False, False])
    assert r["pre_violation_false_alarm"]
    assert r["missed_violation"]
    assert r["detection_delay_frames"] is None
    assert detection(np.array([1, 1, 1, -1]), [True, True, False, False])["detection_delay_frames"] == 1


def test_token_mask_keeps_pixel_overlap_and_location():
    weights = token_weights((16, 16, 32, 32))
    assert weights.sum() == 1
    assert weights[9] == 1
    assert np.count_nonzero(weights) == 1


def test_roi_uses_initial_image_and_tracks_initial_scene_variation():
    image = np.zeros((128, 128, 3), np.uint8)
    image[60:80, 55:65] = [120, 50, 50]
    box, fallback = mug_box(image)
    assert box == (50, 55, 70, 85)
    assert not fallback
    changed = np.roll(image, 4, axis=1)
    assert mug_box(changed)[0][0] == box[0] + 4


def test_false_safe_calibration_constraint_handles_ties():
    record = {"safe_distance": np.array([0, .1, .2, .2, .3, .4]),
              "goal_distance": np.array([1, .9, .8, .2, .1, 0]),
              "safe_label": np.array([True, True, True, False, False, False]),
              "reach_label": np.array([False, False, False, True, True, True])}
    cfg = calibration([record])["false_safe_05"]
    assert (record["safe_distance"][~record["safe_label"]] <= cfg["safe_threshold"]).mean() <= .05
    assert cfg["safe_scale"] > 0
