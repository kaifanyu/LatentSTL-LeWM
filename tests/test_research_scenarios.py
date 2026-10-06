import numpy as np

from latent_stl_lewm.research_scenarios import SCENARIOS, image_transform, scenario_config
from latent_stl_lewm.simulator import SimulatorConfig


def test_scenarios_preserve_task_and_do_not_mutate_config():
    original = dict(goal_tolerance=.03, displacement_tolerance=.002,
                    rotation_tolerance_deg=2., start_eef=(-.05, .22, .54))
    for scenario in SCENARIOS.values():
        cfg = scenario_config(original, scenario)
        assert cfg['goal_tolerance'] == .03
        assert cfg['displacement_tolerance'] == .002
        assert cfg['rotation_tolerance_deg'] == 2.
    assert original['start_eef'] == (-.05, .22, .54)
    shifted = scenario_config(original, SCENARIOS['start_x_plus3cm'])
    np.testing.assert_allclose(shifted['start_eef'], (-.02, .22, .54))
    assert scenario_config(SimulatorConfig(), SCENARIOS['start_jitter_3cm'])['start_jitter'] == .03


def test_brightness_does_not_wrap_uint8():
    rgb = np.asarray([0, 100, 200, 255], dtype=np.uint8)
    np.testing.assert_array_equal(image_transform(rgb, .85), [0, 85, 170, 216])
    np.testing.assert_array_equal(image_transform(rgb, 2.), [0, 200, 255, 255])
