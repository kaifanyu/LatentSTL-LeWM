#!/usr/bin/env python3
"""Read-only CPU audit of completed waypoint research episodes.

Only execution_audit.json and execution_audit_summary.json are written. The
audited run directories, frozen sources, checkpoints, and reference artifacts
are never modified. Rerun after additional episodes finish; passing episode
audits are cached using input file metadata plus verified run/artifact hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import h5py
import numpy as np
import torch

torch.set_num_threads(2)
VERSION = 1
HASH_CACHE = {}
DATA_CACHE = {}
PRIOR_CACHE = {}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
        HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]


def finite(value, name):
    require(np.isfinite(value).all(), f"Nonfinite {name}")


def close(actual, expected, name, atol=2e-6):
    require(np.allclose(actual, expected, rtol=0, atol=atol), f"Mismatch: {name}")


def cosine(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-8)


def command_options(command):
    result = {}
    index = 0
    while index < len(command):
        if not command[index].startswith("--"):
            index += 1
            continue
        name = command[index][2:].replace("-", "_")
        index += 1
        values = []
        while index < len(command) and not command[index].startswith("--"):
            values.append(command[index])
            index += 1
        result[name] = values
    return result


def compare_options(options, actual):
    for name, values in options.items():
        require(name in actual, f"Missing configured option {name}")
        value = actual[name]
        if isinstance(value, bool):
            expected = not values or values[0].lower() == "true"
        elif isinstance(value, list):
            expected = [type(value[0])(v) for v in values] if value else values
        elif isinstance(value, (int, float)):
            require(len(values) == 1, f"Invalid manifest option {name}")
            expected = type(value)(values[0])
        else:
            require(len(values) == 1, f"Invalid manifest option {name}")
            expected = values[0]
        require(value == expected, f"Frozen parameter changed: {name}: {value!r} != {expected!r}")


def training_data(dataset, cwd):
    path = (cwd / dataset).resolve()
    if path not in DATA_CACHE:
        with h5py.File(path, "r") as data:
            metadata = json.loads(data.attrs["metadata_json"])
            actions = np.concatenate([ep["action"][:] for ep in data["trajectories"].values()
                                      if ep.attrs["split"] == "train"])
            episodes = {name: {"split": str(ep.attrs["split"]), "scene_seed": int(ep.attrs["scene_seed"]),
                               "actions": ep["action"][:]} for name, ep in data["trajectories"].items()}
        DATA_CACHE[path] = (metadata, actions.min(0), actions.max(0), episodes)
    return DATA_CACHE[path]


def prior_data(path):
    path = Path(path).resolve()
    if path not in PRIOR_CACHE:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        PRIOR_CACHE[path] = {name: value.numpy() if torch.is_tensor(value) else value
                             for name, value in artifact.items()}
    return PRIOR_CACHE[path]


def validate_run(run, job, frozen, cwd):
    config = read(run / "config.json")
    args = config["args"]
    compare_options(command_options(job["command"]), args)
    require(config["evaluation_role"] == "heldout", "Final run is not labeled heldout")
    require(args["steps"] == 160 and 1 <= args["execute_steps"] <= args["horizon"], "Invalid final execution budget")
    require(config["physical_state_for_action_selection"] is False, "Privileged selection declared")
    require(config["stop_on_physical_success"] is False, "Physical success changes stopping")
    require(config["cem"]["action_repeat"] == 1, "Unexpected action repetition")
    for name, expected in config["source_hashes"].items():
        require(digest(run / "source_snapshot" / name) == expected, f"Archived source changed: {name}")
        frozen_key = "latent_stl_lewm/" + name
        if frozen_key in frozen:
            require(expected == frozen[frozen_key], f"Source differs from frozen recipe: {name}")
    for argument, field in (("checkpoint", "checkpoint_sha256"), ("references", "reference_sha256"),
                            ("action_prior", "prior_sha256")):
        require(digest(cwd / args[argument]) == config[field], f"Artifact differs from run: {argument}")
    references = torch.load(cwd / args["references"], map_location="cpu", weights_only=True)
    prior = prior_data(cwd / args["action_prior"])
    dataset_hash = digest(cwd / args["dataset"])
    require(prior["dataset_sha256"] == dataset_hash == references["metadata"]["dataset_sha256"], "Dataset provenance mismatch")
    metadata, training_low, training_high, episodes = training_data(args["dataset"], cwd)
    low = np.maximum(training_low, -args["translation_bound"])
    high = np.minimum(training_high, args["translation_bound"])
    low[3:6] = high[3:6] = 0
    low[6] = high[6] = -1
    require(np.array_equal(low, np.asarray(config["action_low"], np.float32)), "Incorrect lower action bounds")
    require(np.array_equal(high, np.asarray(config["action_high"], np.float32)), "Incorrect upper action bounds")
    original_seeds = {x["scene_seed"] for x in episodes.values()}
    require(not original_seeds.intersection(args["scene_seeds"]), "Final scene overlaps original dataset")
    for name in prior["trajectory_ids"]:
        require(episodes[name]["split"] == "train", "Prior includes nontraining actions")
    return config, metadata, prior, references


def nominal_actions(prior, episode, progress, horizon, low, high):
    sequence = prior["actions"][episode, progress:progress + horizon]
    if len(sequence) < horizon:
        hold = np.zeros((horizon - len(sequence), 7), np.float32)
        hold[:, -1] = -1
        sequence = np.concatenate([sequence, hold])
    return np.clip(sequence, low, high)


def audit_episode(folder, config, metadata, prior, references):
    args = config["args"]
    summary = read(folder / "summary.json")
    rows = [read(p) for p in sorted(folder.glob("cycle_*.json"))]
    with np.load(folder / "trajectory.npz") as source:
        trajectory = {name: source[name] for name in source.files}
    with np.load(folder / "start_snapshot.npz") as source:
        snapshot = {name: source[name] for name in source.files}
    with np.load(folder / "actual_latent_trace.npz") as source:
        actual = {name: source[name] for name in source.files}
    count, horizon = args["steps"], args["horizon"]
    require(trajectory["action"].shape == (count, 7) and len(trajectory["rgb"]) == count + 1, "Incomplete execution")
    require(len(actual["latent"]) == count + 1, "Incorrect observation latent count")
    for name, value in trajectory.items():
        finite(value, name)
    for name, value in actual.items():
        if name == "history_safe_min":
            require(np.isposinf(value).all(), "Wrong empty-history neutral safety")
        elif name == "history_reach_max":
            require(np.isneginf(value).all(), "Wrong empty-history neutral reach")
        else:
            finite(value, name)
    sim = metadata["simulator_config"]
    require(sim["goal_tolerance"] == .03 and sim["displacement_tolerance"] == .002
            and sim["rotation_tolerance_deg"] == 2, "Physical criteria changed")
    object_pos = trajectory["diagnostic_object_pos"]
    displacement = np.linalg.norm(object_pos - snapshot["initial_object_pos"], axis=-1).max(-1)
    q = trajectory["diagnostic_object_quat"]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q0 = snapshot["initial_object_quat"]
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    rotation = np.rad2deg(2 * np.arccos(np.clip(np.abs((q * q0).sum(-1)), 0, 1))).max(-1)
    goal_error = np.linalg.norm(trajectory["diagnostic_eef_pos"] - np.asarray(sim["goal_eef"]), axis=-1)
    physically_safe = bool((displacement <= .002).all() and (rotation <= 2).all())
    reached = bool((goal_error <= .03).any())
    require(physically_safe == summary["physical_safe"] and reached == summary["physical_reached"], "Physical labels do not match raw poses")
    require((physically_safe and reached) == summary["physical_success"], "Physical success inconsistent")
    close(displacement, trajectory["diagnostic_object_displacement"], "mug displacement", atol=1e-7)
    close(rotation, trajectory["diagnostic_object_rotation_deg"], "mug rotation", atol=5e-6)
    close(goal_error, trajectory["diagnostic_goal_error"], "goal distance", atol=1e-7)
    close(summary["final_goal_error_m"], goal_error[-1], "summary final distance", atol=1e-7)
    close(summary["minimum_goal_error_m"], goal_error.min(), "summary minimum distance", atol=1e-7)
    safe, reach = actual["safe_trace"][0], actual["reach_trace"][0]
    require(safe.shape == reach.shape == (count + 1,), "Incorrect actual predicate trace shape")
    close(summary["actual_stl"], min(safe.min(), reach.max()), "summary exact STL")
    close(actual["stl"], [min(safe.min(), reach.max())], "saved exact STL")
    # Recompute safety directly from recorded features and the frozen banks.
    z = actual["latent"].astype(np.float32)
    z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    sb = references["safe_bank"].numpy().copy(); ub = references["unsafe_bank"].numpy().copy()
    sb /= np.linalg.norm(sb, axis=-1, keepdims=True); ub /= np.linalg.norm(ub, axis=-1, keepdims=True)
    close(safe, (z @ sb.T).max(1) - (z @ ub.T).max(1) - references["safe_margin"], "actual safety from frozen reference banks", atol=4e-6)
    position, last_progress, evaluations = 0, 0, 0
    sources = Counter()
    low, high = np.asarray(config["action_low"], np.float32), np.asarray(config["action_high"], np.float32)
    for cycle, row in enumerate(rows):
        require(row["cycle"] == cycle, "Missing/reordered replan")
        proposed = np.asarray(row["optimized_action_sequence"], np.float32)
        executed = np.asarray(row["executed_actions"], np.float32)
        require(proposed.shape == (horizon, 7), "Incorrect planning horizon")
        require(len(executed) == min(args["execute_steps"], count - position), "Incorrect executed block")
        require(np.array_equal(proposed[:len(executed)], executed), "Executed action is not chosen prefix")
        require(np.array_equal(trajectory["action"][position:position + len(executed)], executed), "Recorded simulator actions differ")
        require(((proposed >= low) & (proposed <= high)).all(), "Action outside training bounds")
        alignment, reference = row["alignment"], row["reference"]
        progress, ep = alignment["progress_frame"], reference["episode_index"]
        require(reference["split"] == "train" and reference["trajectory"] == prior["trajectory_ids"][ep], "Incorrect demonstration provenance")
        require(progress >= last_progress and progress <= last_progress + args["maximum_advance"], "Nonmonotone/unbounded phase advance")
        end = int((prior["frame_episode"] == ep).sum()) - 1
        require(alignment["waypoint_frame"] == min(end, progress + args["lookahead"]), "Incorrect waypoint lookahead")
        nominal = nominal_actions(prior, ep, progress, horizon, low, high)
        if args["mode"] == "prior":
            require(row["candidate_evaluations"] == 0 and row["selected_source"] == "training_demonstration_control", "Incorrect prior-only labeling")
            require(np.array_equal(proposed, nominal), "Prior control differs from training sequence")
        else:
            require(row["config"]["seed"] == args["seed"] + summary["scene_seed"] + cycle, "Unexpected CEM seed")
            for name, value in config["cem"].items():
                if name != "seed":
                    require(row["config"][name] == value, f"Changed cycle CEM parameter {name}")
            winner = None
            with np.load(folder / f"candidates_{cycle:03d}.npz") as pop:
                for iteration in range(args["iterations"]):
                    prefix = f"iteration_{iteration}_"
                    actions = pop[prefix + "candidate_actions"]
                    utility = pop[prefix + "candidate_goal_utility"]
                    keys = pop[prefix + "candidate_selection_keys"]
                    future_safe = pop[prefix + "score_future_safe_min"]
                    candidate_safe = pop[prefix + "score_safe_trace"]
                    candidate_reach = pop[prefix + "score_reach_trace"]
                    for field in pop.files:
                        if not field.startswith(prefix):continue
                        value = pop[field]
                        if cycle == 0 and field.endswith("score_history_safe_min"):
                            require(np.isposinf(value).all(), "Wrong initial history neutral safety")
                        elif cycle == 0 and field.endswith("score_history_reach_max"):
                            require(np.isneginf(value).all(), "Wrong initial history neutral reach")
                        else:finite(value, field)
                    require(actions.shape == (args["population"], horizon, 7), "Wrong CEM population")
                    require(((actions >= low) & (actions <= high)).all(), "Sampled action outside training bounds")
                    close(future_safe, candidate_safe[:, 1:].min(1), "future safety excludes current")
                    past_safe = safe[:position].min() if position else np.inf
                    past_reach = reach[:position].max() if position else -np.inf
                    expected_stl = np.minimum(np.minimum(past_safe, candidate_safe.min(1)),
                                              np.maximum(past_reach, candidate_reach.max(1)))
                    close(pop[prefix + "score_stl"], expected_stl, "candidate exact full-history STL", atol=4e-6)
                    feasible = future_safe >= config["cem"]["feasible_threshold"]
                    require(np.array_equal(feasible, pop[prefix + "candidate_feasible"].astype(bool)), "Incorrect feasibility labels")
                    if args["objective"] == "safe_goal":
                        expected_keys = np.stack([feasible.astype(np.float32), np.where(feasible, utility, future_safe), np.where(feasible, future_safe, utility)], axis=-1)
                    else:
                        expected_keys = np.stack([np.ones_like(utility), utility, future_safe], axis=-1)
                    require(np.array_equal(keys, expected_keys), "Incorrect lexicographic keys")
                    if args["deviation_weight"]:
                        deviation = ((actions[..., :3] - nominal[None, :, :3]) ** 2).mean((1, 2))
                        close(pop[prefix + "score_action_prior_deviation"], deviation, "prior deviation penalty")
                    index = max(range(len(keys)), key=lambda j: tuple(keys[j]))
                    record = (tuple(keys[index]), iteration, index)
                    require(row["iterations"][iteration]["selected_index"] == index, "Incorrect iteration winner")
                    if winner is None or record[0] > winner[0]:winner = record
                require((row["selected_iteration"], row["selected_index"]) == winner[1:], "Selected candidate is not global lexicographic winner")
                candidate = pop[f"iteration_{winner[1]}_candidate_actions"][winner[2]]
                require(np.array_equal(proposed, candidate), "Chosen sequence differs from archived sampled candidate")
                source = row["iterations"][winner[1]]["candidate_sources"][winner[2]]
                require(source == row["selected_source"], "Selected source mislabeled")
            require(row["candidate_evaluations"] == args["population"] * args["iterations"], "Incorrect evaluation count")
            evaluations += row["candidate_evaluations"]
        with np.load(folder / f"prediction_{cycle:03d}.npz") as pred:
            for name in pred.files:finite(pred[name], name)
            require(pred["latent"].shape[1] == horizon + 1, "Prediction timestep mismatch")
            require(np.array_equal(pred["actions"], proposed), "Prediction actions differ")
            require(np.array_equal(pred["executed_prefix_actual_latent"], actual["latent"][position + 1:position + 1 + len(executed)]), "Actual encoded prefix differs")
            error = 1 - cosine(pred["latent"][0, len(executed)], pred["executed_prefix_actual_latent"][-1])
            close(row["execution_prediction_cosine_error"], error, "execution prediction error")
        position += len(executed)
        require(row["environment_steps"] == position, "Duplicated/skipped simulator timestep")
        close(row["actual_stl"], min(safe[:position + 1].min(), reach[:position + 1].max()), "cycle actual exact STL", atol=4e-6)
        close(row["actual_goal_distance"], references["goal_threshold"] - reach[position], "cycle actual goal distance")
        sources[row["selected_source"]] += 1
        last_progress = progress
    require(position == count == summary["environment_steps"] and len(rows) == summary["cycles"], "Final execution count mismatch")
    require(dict(sources) == summary["selected_sources"], "Summary source counts differ")
    return {"passed": True, "cycles": len(rows), "actions": count, "candidate_evaluations": evaluations,
            "mode": args["mode"], "objective": args["objective"], "scenario": args["scenario"],
            "physical_safe": physically_safe, "physical_reached": reached,
            "physical_success": physically_safe and reached, "actual_stl": summary["actual_stl"],
            "final_goal_error_m": float(goal_error[-1]), "minimum_goal_error_m": float(goal_error.min()),
            "selected_sources": dict(sources)}


def episode_signature(folder, config):
    files = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(folder.iterdir())
             if p.suffix in {".json", ".npz"} and not p.name.startswith("latest")]
    payload = json.dumps({"version": VERSION, "files": files, "config": config}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Reaudit cached passing episodes")
    args = parser.parse_args()
    final = args.final_dir.resolve()
    cwd = Path(__file__).resolve().parents[1]
    output = final / "execution_audit.json"
    old = read(output).get("episodes", {}) if output.exists() and not args.force else {}
    report = {"audit_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "cpu_only": True, "writes_only_audit_reports": True, "frozen_hash_checks": {},
              "run_checks": {}, "episodes": {}, "pending_episodes": [], "errors": []}
    frozen = read(final / "freeze.json")["source_and_artifact_sha256"]
    for name, expected in frozen.items():
        passed = digest(cwd / name) == expected
        report["frozen_hash_checks"][name] = passed
        if not passed:report["errors"].append(f"Frozen hash mismatch: {name}")
    jobs = {}
    for manifest in sorted(final.glob("*.json")):
        if manifest.name.startswith("execution_audit"):continue
        document = read(manifest)
        if not isinstance(document, dict):continue
        for job in document.get("jobs", []):
            if not isinstance(job, dict) or "command" not in job:continue
            options = command_options(job["command"])
            if "output" in options:
                path = (cwd / options["output"][0]).resolve()
                jobs[path] = job
    for run, job in sorted(jobs.items()):
        options = command_options(job["command"])
        scene_seeds = [int(seed) for seed in options.get("scene_seeds", [])]
        if not (run / "config.json").exists():
            report["pending_episodes"].extend(f"{run.name}/scene_{seed}" for seed in scene_seeds)
            continue
        try:
            config, metadata, prior, references = validate_run(run, job, frozen, cwd)
            report["run_checks"][run.name] = {"passed": True, "parameters_match_manifest": True,
                                             "source_and_artifacts_match": True}
        except Exception as exc:
            report["run_checks"][run.name] = {"passed": False, "error": str(exc)}
            report["errors"].append(f"{run.name}: {exc}")
            continue
        for scene_seed in scene_seeds:
            folder = run / f"scene_{scene_seed}"
            key = f"{run.name}/{folder.name}"
            if not (folder / "summary.json").exists():
                report["pending_episodes"].append(key)
                continue
            signature = episode_signature(folder, config)
            if old.get(key, {}).get("signature") == signature and old[key].get("passed"):
                result = dict(old[key], cached=True)
            else:
                try:
                    result = audit_episode(folder, config, metadata, prior, references)
                    result.update(signature=signature, cached=False)
                except Exception as exc:
                    result = {"passed": False, "signature": signature, "error": str(exc), "cached": False}
            report["episodes"][key] = result
            if not result["passed"]:report["errors"].append(f"{key}: {result['error']}")
    good = [value for value in report["episodes"].values() if value["passed"]]
    report["summary"] = {"completed_episodes_audited": len(report["episodes"]),
                         "passing_episodes": len(good), "cached_episodes": sum(x.get("cached", False) for x in good),
                         "pending_episodes": len(report["pending_episodes"]), "errors": len(report["errors"]),
                         "actions_verified": sum(x["actions"] for x in good),
                         "candidate_evaluations_verified": sum(x["candidate_evaluations"] for x in good),
                         "physical_successes": sum(x["physical_success"] for x in good),
                         "all_completed_checks_passed": not report["errors"]}
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(output)
    (final / "execution_audit_summary.json").write_text(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["summary"], indent=2))
    for error in report["errors"]:print(error, file=sys.stderr)
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
