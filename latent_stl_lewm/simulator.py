"""Single-camera LIBERO mug scene. Geometry is diagnostic or collection-only.

The learned policy consumes ``render()`` and executes raw OSC actions. Collection
controllers may use simulator state to obtain coverage; this is not a planner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SimulatorConfig:
    camera: str = "agentview"
    image_size: int = 128
    control_freq: int = 20
    render_gpu: int = 2
    stage_steps: int = 48
    settle_steps: int = 12
    controller_gain: float = 15.0
    start_eef: tuple = (-0.05, 0.22, 0.54)
    goal_eef: tuple = (-0.05, -0.22, 0.54)
    goal_tolerance: float = 0.03
    displacement_tolerance: float = 0.002
    rotation_tolerance_deg: float = 2.0
    start_jitter: float = 0.012


def minimum_jerk(u):
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


class MugPreservationEnv:
    def __init__(self, config: SimulatorConfig | dict | None = None):
        self.config = config if isinstance(config, SimulatorConfig) else SimulatorConfig(**(config or {}))
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
        libero_path = ROOT / "third_party" / "LIBERO"
        if str(libero_path) not in sys.path:
            sys.path.insert(0, str(libero_path))
        from libero.libero.envs import OffScreenRenderEnv

        self.bddl_path = Path(__file__).parent / "assets" / "mug_obstacle_table_move_arm.bddl"
        cfg = self.config
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(self.bddl_path), camera_names=[cfg.camera],
            camera_heights=cfg.image_size, camera_widths=cfg.image_size,
            render_gpu_device_id=cfg.render_gpu, control_freq=cfg.control_freq,
            horizon=1000000, ignore_done=True, hard_reset=False,
        )
        self.action_low, self.action_high = (np.asarray(x, dtype=np.float32) for x in self.env.env.action_spec)
        self.action_dim = len(self.action_low)
        if self.action_dim != 7:
            raise RuntimeError(f"Expected inspected Panda OSC_POSE 7D action, received {self.action_dim}")
        self.obs: dict[str, Any] = {}
        self.control_step = 0

    def _objects(self):
        states = [self.env.env.object_states_dict[name].get_geom_state()
                  for name in self.env.obj_of_interest]
        return (np.asarray([s["pos"] for s in states], dtype=np.float64),
                np.asarray([s["quat"] for s in states], dtype=np.float64))

    def reset(self, seed: int = 0):
        self.env.seed(int(seed))
        self.obs = self.env.reset()
        self.control_step = 0
        rng = np.random.default_rng(seed)
        target = np.asarray(self.config.start_eef, dtype=float)
        if self.config.start_jitter:
            target += rng.uniform(-self.config.start_jitter, self.config.start_jitter, 3)
        start = self.obs["robot0_eef_pos"].copy()
        for t in range(self.config.stage_steps + self.config.settle_steps):
            u = min(1.0, (t + 1) / self.config.stage_steps)
            desired = start + minimum_jerk(u) * (target - start)
            action = self.action_to_target(desired)
            self.obs, _, _, _ = self.env.step(action)
        if np.linalg.norm(self.obs["robot0_eef_pos"] - target) > 0.035:
            raise RuntimeError("Unable to stage arm at the mug avoidance start pose")
        for robot in self.env.robots:
            robot.controller.update(force=True)
            robot.controller.update_initial_joints(np.asarray(robot.controller.joint_pos).copy())
            robot.controller.reset_goal()
        self._initial_object_pos, self._initial_object_quat = self._objects()
        self.control_step = 0
        self.scene_seed = int(seed)
        return self.render()

    def action_to_target(self, target):
        """Privileged data-collection controller; never used by latent CEM."""
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[:3] = self.config.controller_gain * (np.asarray(target) - self.obs["robot0_eef_pos"])
        action[-1] = -1.0
        return np.clip(action, self.action_low, self.action_high)

    def render(self):
        # Match existing LIBERO/ImageSTL image convention (180 degree rotation).
        return np.ascontiguousarray(self.obs[f"{self.config.camera}_image"][::-1, ::-1], dtype=np.uint8)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite raw OSC_POSE vector")
        if np.any(action < self.action_low - 1e-6) or np.any(action > self.action_high + 1e-6):
            raise ValueError("Action outside environment bounds")
        self.obs, _, done, info = self.env.step(action)
        self.control_step += 1
        return self.render(), bool(done), info

    def snapshot(self):
        import mujoco
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        model, data = self.env.sim.model._model, self.env.sim.data._data
        integration = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
        mujoco.mj_getState(model, data, integration, spec)
        return {"sim_state": np.asarray(self.env.get_sim_state()).copy(),
                "integration_state": integration,
                "gripper_actions": np.stack([r.gripper.current_action.copy() for r in self.env.robots]),
                "controller_initial_joints": np.stack([r.controller.initial_joint.copy() for r in self.env.robots]),
                "initial_object_pos": self._initial_object_pos.copy(),
                "initial_object_quat": self._initial_object_quat.copy(),
                "control_step": self.control_step, "scene_seed": self.scene_seed}

    def restore(self, snapshot):
        import mujoco
        if "integration_state" in snapshot:
            mujoco.mj_setState(self.env.sim.model._model, self.env.sim.data._data,
                              np.asarray(snapshot["integration_state"]), mujoco.mjtState.mjSTATE_INTEGRATION)
        else:
            self.env.set_state(np.asarray(snapshot["sim_state"]))
        self.env.sim.forward()
        self.env.env._obs_cache = {}
        for observable in self.env.env._observables.values():
            observable.reset()
            observable._sampled = False
        self.env._update_observables(force=True)
        self.obs = self.env.env._get_observations()
        for i, robot in enumerate(self.env.robots):
            robot.gripper.current_action = np.asarray(snapshot["gripper_actions"][i]).copy()
            robot.controller.update(force=True)
            robot.controller.update_initial_joints(snapshot["controller_initial_joints"][i])
            robot.controller.reset_goal()
        self._initial_object_pos = np.asarray(snapshot["initial_object_pos"]).copy()
        self._initial_object_quat = np.asarray(snapshot["initial_object_quat"]).copy()
        self.control_step = int(snapshot["control_step"])
        self.scene_seed = int(snapshot["scene_seed"])
        self.env.env.timestep = self.control_step
        self.env.env.cur_time = float(self.env.sim.data.time)
        self.env.env.done = False
        return self.render()

    def diagnostics(self):
        pos, quat = self._objects()
        displacement = np.linalg.norm(pos - self._initial_object_pos, axis=-1).max()
        normalized = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
        initial = self._initial_object_quat / np.linalg.norm(self._initial_object_quat, axis=-1, keepdims=True)
        rotation = np.rad2deg(2 * np.arccos(np.clip(np.abs((normalized * initial).sum(-1)), 0, 1))).max()
        eef = np.asarray(self.obs["robot0_eef_pos"])
        goal_error = np.linalg.norm(eef - self.config.goal_eef)
        sim = self.env.sim
        mug_contact = 0
        for contact in sim.data.contact[:sim.data.ncon]:
            names = [sim.model.geom_id2name(int(contact.geom1)) or "",
                     sim.model.geom_id2name(int(contact.geom2)) or ""]
            if any("red_coffee_mug" in n for n in names) and any("robot" in n or "gripper" in n for n in names):
                mug_contact += 1
        return {"eef_pos": eef.copy(), "eef_quat": np.asarray(self.obs["robot0_eef_quat"]).copy(),
                "joint_pos": np.asarray(self.obs["robot0_joint_pos"]).copy(),
                "object_pos": pos, "object_quat": quat,
                "contact_count": np.int32(sim.data.ncon), "robot_mug_contact_count": np.int32(mug_contact),
                "object_displacement": np.float32(displacement),
                "object_rotation_deg": np.float32(rotation), "goal_error": np.float32(goal_error),
                "safe": np.uint8(displacement <= self.config.displacement_tolerance and rotation <= self.config.rotation_tolerance_deg),
                "reach": np.uint8(goal_error <= self.config.goal_tolerance)}

    def metadata(self):
        cfg = asdict(self.config)
        camera_id = self.env.sim.model.camera_name2id(self.config.camera)
        return {"simulator_config": cfg, "scene": "living_room_mug_obstacle",
                "bddl_sha256": hashlib.sha256(self.bddl_path.read_bytes()).hexdigest(),
                "controller": "OSC_POSE", "action_dim": self.action_dim,
                "action_low": self.action_low.tolist(), "action_high": self.action_high.tolist(),
                "translation_output_scale_m": 0.05, "rotation_output_scale_rad": 0.5,
                "action_interpretation": "delta_xyz, world-frame delta_axis_angle, gripper; raw normalized OSC_POSE",
                "control_freq_hz": self.config.control_freq, "frame_freq_hz": self.config.control_freq,
                "action_repeat": 1, "camera": self.config.camera, "image_rotation_degrees": 180,
                "camera_position": self.env.sim.model.cam_pos[camera_id].tolist(),
                "camera_quaternion_wxyz": self.env.sim.model.cam_quat[camera_id].tolist(),
                "camera_fovy": float(self.env.sim.model.cam_fovy[camera_id]),
                "object_names": list(self.env.obj_of_interest)}

    def close(self):
        self.env.close()
