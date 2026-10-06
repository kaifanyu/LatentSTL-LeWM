"""Controlled evaluation variations; scenario state never enters the planner.

Offsets change simulator initialization only. Brightness changes the image seen
by the encoder and saved in the replay. Goal and safety tolerances are inherited
unchanged. A single fixed task goal is used across these robustness tests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .simulator import MugPreservationEnv, SimulatorConfig


@dataclass(frozen=True)
class Scenario:
    name: str = "baseline"
    start_jitter: float | None = None
    start_offset: tuple = (0., 0., 0.)
    mug_offset: tuple = (0., 0., 0.)
    camera_offset: tuple = (0., 0., 0.)
    brightness: float = 1.


SCENARIOS = {
    "baseline": Scenario(),
    "start_jitter_3cm": Scenario("start_jitter_3cm", start_jitter=.03),
    "start_x_plus3cm": Scenario("start_x_plus3cm", start_offset=(.03, 0., 0.)),
    "mug_x_plus3cm": Scenario("mug_x_plus3cm", mug_offset=(.03, 0., 0.)),
    "mug_x_minus3cm": Scenario("mug_x_minus3cm", mug_offset=(-.03, 0., 0.)),
    "mug_y_plus3cm": Scenario("mug_y_plus3cm", mug_offset=(0., .03, 0.)),
    "camera_x_plus2cm": Scenario("camera_x_plus2cm", camera_offset=(.02, 0., 0.)),
    "dim_085": Scenario("dim_085", brightness=.85),
}


def scenario_config(config, scenario):
    cfg = asdict(config) if isinstance(config, SimulatorConfig) else dict(config)
    defaults = SimulatorConfig()
    cfg["start_eef"] = tuple(np.asarray(cfg.get("start_eef", defaults.start_eef))
                             + np.asarray(scenario.start_offset))
    if scenario.start_jitter is not None:
        cfg["start_jitter"] = scenario.start_jitter
    return cfg


def image_transform(rgb, brightness):
    if brightness == 1.:
        return rgb
    return np.clip(np.asarray(rgb, dtype=np.float32) * brightness, 0, 255).astype(np.uint8)


class ScenarioEnv(MugPreservationEnv):
    def __init__(self, config, scenario="baseline"):
        self.scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        super().__init__(scenario_config(config, self.scenario))
        self._camera_id = self.env.sim.model.camera_name2id(self.config.camera)
        self._base_camera_position = self.env.sim.model.cam_pos[self._camera_id].copy()

    def render(self):
        return image_transform(super().render(), self.scenario.brightness)

    def _refresh_observations(self):
        self.env.sim.forward()
        self.env.env._obs_cache = {}
        for observable in self.env.env._observables.values():
            observable.reset()
            observable._sampled = False
        self.env._update_observables(force=True)
        self.obs = self.env.env._get_observations()

    def reset(self, seed=0):
        self.env.sim.model.cam_pos[self._camera_id] = self._base_camera_position
        super().reset(seed)
        if any(self.scenario.mug_offset):
            for name in self.env.obj_of_interest:
                obj = self.env.env.objects_dict[name]
                joint = obj.joints[-1]
                qpos = self.env.sim.data.get_joint_qpos(joint).copy()
                qpos[:3] += np.asarray(self.scenario.mug_offset)
                self.env.sim.data.set_joint_qpos(joint, qpos)
                self.env.sim.data.set_joint_qvel(joint, np.zeros(6))
            self._refresh_observations()
            hold = np.zeros(self.action_dim, dtype=np.float32)
            hold[-1] = -1.
            for _ in range(12):
                self.obs, _, _, _ = self.env.step(hold)
            self._initial_object_pos, self._initial_object_quat = self._objects()
        self.env.sim.model.cam_pos[self._camera_id] = (
            self._base_camera_position + np.asarray(self.scenario.camera_offset))
        self._refresh_observations()
        self.control_step = 0
        return self.render()

    def metadata(self):
        return {**super().metadata(), "scenario": asdict(self.scenario),
                "scenario_role": "evaluation initialization/image variation only"}


def make_scenario_env(config, scenario="baseline"):
    return ScenarioEnv(config, scenario)
