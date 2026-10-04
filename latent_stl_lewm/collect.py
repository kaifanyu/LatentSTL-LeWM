"""Collect genuine LIBERO image/action trajectories and held-out candidate banks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image

from .simulator import MugPreservationEnv, SimulatorConfig, minimum_jerk

POLICIES = ("noop", "low_motion", "safe_arc", "collision_straight",
            "near_collision", "safe_lateral", "random_smooth", "random_reach")


def control_action(env, policy, t, steps, start, rng, parameters):
    """Collection-only scripted behavior; the stored action is exactly executed."""
    if policy == "noop":
        return np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    if policy == "low_motion":
        target = start + parameters["small_delta"] * np.sin(2 * np.pi * t / steps)
    elif policy == "random_smooth":
        if t % 8 == 0:
            parameters["random_action"] = rng.uniform(-0.6, 0.6, env.action_dim)
            parameters["random_action"][3:6] *= 0.2
            parameters["random_action"][-1] = rng.choice([-1.0, 1.0])
        # Workspace correction supplies motion coverage while preventing table dives.
        action = parameters["random_action"].copy()
        eef = env.obs["robot0_eef_pos"]
        for axis, low, high in ((0, -0.22, 0.22), (1, -0.32, 0.32), (2, 0.50, 0.74)):
            if eef[axis] < low:
                action[axis] = abs(action[axis])
            if eef[axis] > high:
                action[axis] = -abs(action[axis])
        return np.asarray(action, dtype=np.float32)
    else:
        u = min(1.0, (t + 1) / max(steps - 12, 1))
        target = start + minimum_jerk(u) * (parameters["target"] - start)
        target[2] += parameters["arc"] * 4 * u * (1 - u)
        target[0] += parameters["lateral"] * 4 * u * (1 - u)
    action = env.action_to_target(target)
    if policy == "low_motion":
        action[3:6] = parameters["small_rotation"] * np.sin(2 * np.pi * t / steps)
    return action


def collect_trajectory(env, policy, steps, seed):
    rng = np.random.default_rng(seed)
    start = np.asarray(env.obs["robot0_eef_pos"]).copy()
    parameters = {"target": np.asarray(env.config.goal_eef).copy(), "arc": 0.0, "lateral": 0.0,
                  "small_delta": rng.uniform(-0.025, 0.025, 3),
                  "small_rotation": rng.uniform(-0.025, 0.025, 3)}
    if policy == "safe_arc":
        parameters["arc"] = rng.uniform(0.10, 0.16)
    elif policy == "near_collision":
        parameters["arc"] = rng.uniform(0.025, 0.07)
    elif policy == "safe_lateral":
        parameters["lateral"] = rng.choice([-1, 1]) * rng.uniform(0.09, 0.15)
    elif policy == "random_reach":
        parameters["target"] += rng.uniform([-0.10, -0.09, -0.015], [0.10, 0.18, 0.12])
        parameters["arc"] = rng.uniform(0, 0.15)
        parameters["lateral"] = rng.uniform(-0.10, 0.10)
    rgb, actions, done = [env.render()], [], []
    diagnostics = [env.diagnostics()]
    for t in range(steps):
        action = control_action(env, policy, t, steps, start, rng, parameters)
        frame, terminal, _ = env.step(action)
        rgb.append(frame)
        actions.append(action.copy())
        done.append(terminal)
        diagnostics.append(env.diagnostics())
    return {"rgb": np.stack(rgb), "action": np.asarray(actions, dtype=np.float32),
            "done": np.asarray(done, dtype=np.uint8), "control_step": np.arange(steps + 1, dtype=np.int32),
            "diagnostics": {key: np.stack([row[key] for row in diagnostics]) for key in diagnostics[0]}}


def write_trajectory(group, trajectory, attributes):
    for name in ("rgb", "action", "done", "control_step"):
        data = trajectory[name]
        kwargs = {"compression": "lzf", "chunks": (1, *data.shape[1:])} if name == "rgb" else {}
        group.create_dataset(name, data=data, **kwargs)
    diagnostic_group = group.create_group("diagnostics")
    for name, data in trajectory["diagnostics"].items():
        diagnostic_group.create_dataset(name, data=data)
    for name, value in attributes.items():
        group.attrs[name] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--val-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=32)
    parser.add_argument("--bank-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--render-gpu", type=int, default=2)
    parser.add_argument("--camera", default="agentview", choices=["agentview", "birdview", "frontview"])
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    if min(args.train_episodes, args.val_episodes, args.test_episodes, args.steps, args.bank_size) < 1:
        parser.error("All counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    config = SimulatorConfig(camera=args.camera, image_size=args.image_size, render_gpu=args.render_gpu)
    env = MugPreservationEnv(config)
    start_time = time.monotonic()
    metadata = {"schema_version": 1, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "collection_seed": args.seed, "bank_size": args.bank_size, "steps": args.steps,
                "inputs": ["rgb", "action"], "diagnostics_are_model_inputs": False,
                "split_policy": "disjoint scene seeds and start snapshots; all members of a bank share a split",
                "split_counts": {"train": args.train_episodes, "val": args.val_episodes, "test": args.test_episodes}}
    summary = []
    try:
        env.reset(args.seed - 1)
        metadata.update(env.metadata())
        reference_snapshot = env.snapshot()
        safe_rgb = env.render()
        reference = collect_trajectory(env, "safe_arc", args.steps, args.seed)
        goal_rgb = reference["rgb"][-1]
        metadata["reference_diagnostics"] = {"safe": bool(reference["diagnostics"]["safe"].all()),
                                             "reached": bool(reference["diagnostics"]["reach"][-1]),
                                             "goal_error": float(reference["diagnostics"]["goal_error"][-1])}
        if not metadata["reference_diagnostics"]["safe"] or not metadata["reference_diagnostics"]["reached"]:
            raise RuntimeError(f"Reference rollout failed: {metadata['reference_diagnostics']}")
        Image.fromarray(safe_rgb).save(args.output / "safe.png")
        Image.fromarray(goal_rgb).save(args.output / "goal.png")
        with h5py.File(args.output / "trajectories.partial.h5", "x") as handle:
            for name in ("schema_version", "action_dim", "camera", "control_freq_hz", "frame_freq_hz", "action_repeat"):
                handle.attrs[name] = metadata[name]
            handle.attrs["action_low"] = env.action_low
            handle.attrs["action_high"] = env.action_high
            handle.attrs["metadata_json"] = json.dumps(metadata)
            references = handle.create_group("references")
            references.create_dataset("safe_rgb", data=safe_rgb)
            references.create_dataset("goal_rgb", data=goal_rgb)
            rd = references.create_group("diagnostics")
            rd.create_dataset("goal_eef_pos", data=np.asarray(config.goal_eef))
            snapshots = handle.create_group("start_snapshots")
            trajectories = handle.create_group("trajectories")
            counter = 0
            for split_idx, (split, count) in enumerate(metadata["split_counts"].items()):
                for local in range(count):
                    bank_idx = local // args.bank_size
                    bank_id = f"{split}_{bank_idx:04d}"
                    scene_seed = args.seed + (split_idx + 1) * 100000 + bank_idx
                    if local % args.bank_size == 0:
                        env.reset(scene_seed)
                        snapshot = env.snapshot()
                        sg = snapshots.create_group(bank_id)
                        for name, value in snapshot.items():
                            sg.create_dataset(name, data=value)
                        start_hash = hashlib.sha256(snapshot["sim_state"].tobytes()).hexdigest()
                    env.restore(snapshot)
                    policy = POLICIES[local % len(POLICIES)]
                    trajectory_seed = args.seed + split_idx * 1000000 + local
                    trajectory = collect_trajectory(env, policy, args.steps, trajectory_seed)
                    attrs = {"split": split, "bank_id": bank_id, "policy": policy,
                             "scene_seed": scene_seed, "trajectory_seed": trajectory_seed,
                             "scene_id": "living_room_mug_obstacle", "start_state_sha256": start_hash}
                    write_trajectory(trajectories.create_group(f"{counter:06d}"), trajectory, attrs)
                    diagnostic = trajectory["diagnostics"]
                    record = {"trajectory": f"{counter:06d}", **attrs,
                              "safe": bool(diagnostic["safe"].all()), "reach": bool(diagnostic["reach"].any()),
                              "max_object_displacement_m": float(diagnostic["object_displacement"].max()),
                              "terminal_goal_error_m": float(diagnostic["goal_error"][-1]),
                              "elapsed_seconds": time.monotonic() - start_time}
                    summary.append(record)
                    with (args.output / "progress.jsonl").open("a") as progress:
                        progress.write(json.dumps(record) + "\n")
                    counter += 1
                    handle.flush()
                    print(json.dumps(record), flush=True)
            handle.attrs["complete"] = True
        (args.output / "trajectories.partial.h5").rename(args.output / "trajectories.h5")
        metadata["elapsed_seconds"] = time.monotonic() - start_time
        metadata["num_transitions"] = sum(metadata["split_counts"].values()) * args.steps
        metadata["physical_coverage"] = {f"safe={s},reach={r}": sum(x["safe"] == s and x["reach"] == r for x in summary)
                                        for s in (False, True) for r in (False, True)}
        with (args.output / "trajectories.h5").open("rb") as handle:
            metadata["hdf5_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"complete": True, "output": str(args.output), **metadata}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
