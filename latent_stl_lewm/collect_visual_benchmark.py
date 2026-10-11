"""Collect paired, held-out visual-STL stress tests. Scripts are data collection.

This is not an image-only controller and its actions are not claimed as planned.
Existing run artifacts are never changed. A completed shard is published atomically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

from .collect import POLICIES, collect_trajectory, write_trajectory
from .evaluate import write_json
from .research_scenarios import ScenarioEnv, SCENARIOS
from .simulator import minimum_jerk


def reach_leave(env, steps, seed):
    initial = env.obs["robot0_eef_pos"].copy()
    first = collect_trajectory(env, "safe_arc", steps * 2 // 3, seed)
    start = env.obs["robot0_eef_pos"].copy()
    frames, actions, diagnostics = [], [], []
    remaining = steps - len(first["action"])
    for t in range(remaining):
        u = (t + 1) / remaining
        target = start + minimum_jerk(u) * (initial - start)
        target[2] += .16 * 4*u*(1-u)
        action = env.action_to_target(target)
        frame, _, _ = env.step(action)
        frames.append(frame); actions.append(action); diagnostics.append(env.diagnostics())
    first["rgb"] = np.concatenate([first["rgb"], frames])
    first["action"] = np.concatenate([first["action"], actions])
    first["done"] = np.zeros(steps, np.uint8)
    first["control_step"] = np.arange(steps + 1, dtype=np.int32)
    for name in first["diagnostics"]:
        first["diagnostics"][name] = np.concatenate([first["diagnostics"][name], [d[name] for d in diagnostics]])
    return first


def observability_probe(env, snapshot, directory):
    """Static image interventions, explicitly excluded from executable trajectories."""
    images, diagnostics, conditions = [], [], []
    for axis in ("x", "y", "yaw"):
        for amount in ([0, .001, .002, .004, .008, .02] if axis != "yaw" else [0, 1, 2, 4, 8, 20]):
            env.restore(snapshot)
            for name in env.env.obj_of_interest:
                joint = env.env.env.objects_dict[name].joints[-1]
                qpos = env.env.sim.data.get_joint_qpos(joint).copy()
                if axis == "yaw":
                    # Free-joint qpos quaternion is wxyz; SciPy uses xyzw.
                    rot = Rotation.from_quat(qpos[[4, 5, 6, 3]])
                    quat = (Rotation.from_euler("z", amount, degrees=True) * rot).as_quat()
                    qpos[3:7] = quat[[3, 0, 1, 2]]
                else:
                    qpos[0 if axis == "x" else 1] += amount
                env.env.sim.data.set_joint_qpos(joint, qpos)
                env.env.sim.data.set_joint_qvel(joint, np.zeros(6))
            env._refresh_observations()
            images.append(env.render()); diagnostics.append(env.diagnostics())
            conditions.append({"axis": axis, "amount": amount})
    np.savez_compressed(directory / "observability.npz", rgb=images,
                        **{k: [d[k] for d in diagnostics] for k in diagnostics[0]})
    write_json(directory / "observability.json", {"conditions": conditions,
               "kind": "Static simulator pose interventions, not action rollouts or STL trajectories",
               "reference_indices": [0, 6, 12], "mug_reference_unchanged": True})
    env.restore(snapshot)


def run(args):
    # EGL uses its own device index; CUDA_VISIBLE_DEVICES alone does not route
    # the renderer to the GPU assigned to this collection worker.
    args.render_gpu = getattr(args, "render_gpu", 0)
    if args.render_gpu < 0:
        raise ValueError("render_gpu must be a nonnegative EGL device index")
    if args.banks < 1:
        raise ValueError("banks must be positive")
    if args.steps < 2:
        raise ValueError("steps must be at least 2 for the reach_leave behavior")
    if not 0 <= args.seed <= args.seed + args.banks - 1 < 2**32:
        raise ValueError("Every scene seed must fit an unsigned 32-bit integer")
    with h5py.File(args.source, "r") as old:
        if not old.attrs.get("complete", False):
            raise ValueError("Source dataset is incomplete")
        cfg = json.loads(old.attrs["metadata_json"])["simulator_config"]
        reference_goal = old["references/goal_rgb"][:]
    cfg["render_gpu"] = args.render_gpu
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    env = ScenarioEnv(cfg, args.scenario)
    began = time.monotonic()
    rows = []
    try:
        metadata = env.metadata()
        with h5py.File(out / "trajectories.partial.h5", "x") as h:
            h.attrs["complete"] = False
            h.attrs["action_dim"] = 7
            h.attrs["action_low"], h.attrs["action_high"] = env.action_low, env.action_high
            for name in ("control_freq_hz", "frame_freq_hz", "action_repeat"):
                h.attrs[name] = metadata[name]
            h.attrs["metadata_json"] = json.dumps({**metadata, "purpose": "Held-out visual predicate benchmark",
                "collector_uses_state": True, "predicates_use_state": False,
                "goal_reference": "Original supplied goal image, unchanged under all stress conditions"})
            refs = h.create_group("references")
            refs.create_dataset("goal_rgb", data=reference_goal)
            episodes, snapshots = h.create_group("trajectories"), h.create_group("start_snapshots")
            for bank in range(args.banks):
                seed = args.seed + bank
                env.reset(seed); snapshot = env.snapshot()
                initial_image = env.render()
                if bank == 0:
                    refs.create_dataset("safe_rgb", data=initial_image)
                    observability_probe(env, snapshot, out)
                bank_id = f"{args.scenario}__{seed}"
                sg = snapshots.create_group(bank_id)
                for k, v in snapshot.items():
                    sg.create_dataset(k, data=v)
                start_hash = hashlib.sha256(snapshot["sim_state"].tobytes()).hexdigest()
                for index, policy in enumerate((*POLICIES, "reach_leave")):
                    restored = env.restore(snapshot)
                    if not np.array_equal(restored, initial_image):
                        raise RuntimeError("Candidate bank initial RGB mismatch")
                    trajectory_seed = seed * 100 + index
                    trajectory = (reach_leave(env, args.steps, trajectory_seed) if policy == "reach_leave"
                                  else collect_trajectory(env, policy, args.steps, trajectory_seed))
                    key = f"{bank:03d}_{policy}"
                    attrs = {"split": "test", "bank_id": bank_id, "policy": policy,
                             "scene_seed": seed, "trajectory_seed": trajectory_seed,
                             "start_state_sha256": start_hash, "scenario": args.scenario,
                             "kind": "trajectory"}
                    write_trajectory(episodes.create_group(key), trajectory, attrs)
                    d = trajectory["diagnostics"]
                    row = {"id": key, **attrs, "physical_safe": bool(d["safe"].all()),
                           "physical_reach": bool(d["reach"].any()),
                           "maximum_displacement_m": float(d["object_displacement"].max()),
                           "maximum_rotation_deg": float(d["object_rotation_deg"].max()),
                           "elapsed_seconds": time.monotonic()-began}
                    rows.append(row)
                    with (out / "progress.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    if bank == 0:
                        with imageio.get_writer(out / f"{policy}.mp4", fps=metadata["frame_freq_hz"], codec="libx264", quality=8) as writer:
                            for frame in trajectory["rgb"]:
                                writer.append_data(frame)
                    h.flush()
                    print(json.dumps(row), flush=True)
            h.attrs["complete"] = True
        (out / "trajectories.partial.h5").rename(out / "trajectories.h5")
        from .dataset import audit_dataset
        audit = audit_dataset(out / "trajectories.h5")
        with h5py.File(out / "trajectories.h5", "r") as h:
            for ep in h["trajectories"].values():
                diag = ep["diagnostics"]
                snapshot = h["start_snapshots"][ep.attrs["bank_id"]]
                pos, quat = diag["object_pos"][:], diag["object_quat"][:]
                initial_pos, initial_quat = snapshot["initial_object_pos"][:], snapshot["initial_object_quat"][:]
                displacement = np.linalg.norm(pos-initial_pos, axis=-1).max(-1)
                quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
                initial_quat = initial_quat / np.linalg.norm(initial_quat, axis=-1, keepdims=True)
                rotation = np.rad2deg(2*np.arccos(np.clip(np.abs((quat*initial_quat).sum(-1)), 0, 1))).max(-1)
                safe = (displacement <= cfg["displacement_tolerance"]) & (rotation <= cfg["rotation_tolerance_deg"])
                reach = np.linalg.norm(diag["eef_pos"][:]-cfg["goal_eef"], axis=-1) <= cfg["goal_tolerance"]
                if not np.array_equal(safe, diag["safe"][:].astype(bool)) or not np.array_equal(reach, diag["reach"][:].astype(bool)):
                    raise RuntimeError("Stored labels disagree with raw physical poses")
        audit["physical_labels_recomputed_from_raw_poses"] = True
        write_json(out / "audit.json", audit)
        write_json(out / "collection.json", {"complete": True, "args": vars(args), "episodes": rows,
                   "seconds": time.monotonic()-began})
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scenario", choices=SCENARIOS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--banks", type=int, default=4)
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--render-gpu", type=int, default=0,
                   help="EGL device index visible to the container, independent of CUDA_VISIBLE_DEVICES")
    run(p.parse_args())
