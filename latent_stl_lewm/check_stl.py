"""Check one timed STL formula across visual backends and a diagnostic oracle.

Calibration uses validation episodes only. The frozen calibration artifact is
written before evaluation episodes are accessed. Physical diagnostics never enter
visual feature extraction; they supervise calibration and provide a separate
comparison. This command monitors recorded images, not predicted trajectories.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from .evaluate import sha256, write_json
from .predicate_library import physical_mug_predicates
from .timed_stl import evaluate_timed_formula, formula_horizon, predicate_names
from .visual_predicate_benchmark import METHODS, GOAL_BOX, VisualFeatures, calibration


ATOMS = {"reach", "preserve", "safe"}


def default_formula(deadline=5.0, hold=0.0):
    """Preserve through deadline; complete reaching/holding by that deadline."""
    if not math.isfinite(deadline) or not math.isfinite(hold) or not 0 <= hold <= deadline:
        raise ValueError("Require finite 0 <= hold <= deadline")
    reach = {"op": "predicate", "name": "reach"}
    if hold:
        reach = {"op": "always", "interval": [0, hold], "arg": reach}
    return {"op": "and", "args": [
        {"op": "always", "interval": [0, deadline],
         "arg": {"op": "predicate", "name": "preserve"}},
        {"op": "eventually", "interval": [0, deadline - hold], "arg": reach},
    ]}


def episode_timestamps(episode, control_freq_hz):
    """Use recorded control indices; never stretch a short trace to a deadline."""
    if not np.isfinite(control_freq_hz) or control_freq_hz <= 0:
        raise ValueError("control_freq_hz must be finite and positive")
    if "control_step" not in episode:
        raise ValueError("Episode is missing recorded control_step timestamps")
    steps = np.asarray(episode["control_step"], dtype=np.float64)
    if (steps.ndim != 1 or len(steps) != len(episode["rgb"]) or not len(steps)
            or not np.isfinite(steps).all() or (steps < 0).any()
            or not np.equal(steps, np.floor(steps)).all() or (np.diff(steps) <= 0).any()):
        raise ValueError("control_step must be finite increasing integer indices aligned with RGB")
    return steps / float(control_freq_hz)


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _keys(handle, split, maximum):
    keys = [key for key, ep in sorted(handle["trajectories"].items())
            if _text(ep.attrs["split"]) == split]
    return keys if maximum is None else keys[:maximum]


def _metadata(handle):
    if not bool(handle.attrs.get("complete", False)):
        raise ValueError("Dataset is not marked complete")
    if "control_freq_hz" not in handle.attrs:
        raise ValueError("Dataset requires control_freq_hz metadata")
    frequency = float(handle.attrs["control_freq_hz"])
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("control_freq_hz must be finite and positive")
    raw = handle.attrs.get("metadata_json", "{}")
    metadata = json.loads(_text(raw))
    return frequency, metadata.get("simulator_config", {})


def _labels(episode):
    if "diagnostics" not in episode or any(name not in episode["diagnostics"] for name in ("safe", "reach")):
        raise ValueError("Canonical calibration/comparison needs diagnostics/safe and diagnostics/reach")
    labels = {}
    for name in ("safe", "reach"):
        value = np.asarray(episode["diagnostics"][name])
        if value.shape != (len(episode["rgb"]),) or not np.isin(value, [0, 1]).all():
            raise ValueError(f"Invalid aligned Boolean diagnostics/{name}")
        labels[name] = value.astype(bool)
    return labels


def physical_predicates(episode, simulator_config):
    """Use physical margins when fully specified; otherwise use recorded truth.

    The fallback +/-1 encodes Boolean physical labels only. Its magnitude is not
    a geometric distance and is never compared numerically to visual robustness.
    """
    labels = _labels(episode)
    diag = episode["diagnostics"]
    fields = ("goal_error", "object_displacement", "object_rotation_deg")
    tolerances = ("goal_tolerance", "displacement_tolerance", "rotation_tolerance_deg")
    if all(name in diag for name in fields) and all(name in simulator_config for name in tolerances):
        thresholds = [float(simulator_config[name]) for name in tolerances]
        if any(not math.isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError("Physical tolerances must be finite and positive")
        values = [np.asarray(diag[name], dtype=np.float64) for name in fields]
        if any(value.shape != labels["safe"].shape or not np.isfinite(value).all() for value in values):
            raise ValueError("Physical diagnostics must be finite and aligned")
        margins = physical_mug_predicates(*values, goal_tolerance=thresholds[0],
                                         translation_tolerance=thresholds[1], rotation_tolerance_deg=thresholds[2])
        reach, preserve = margins["reach"].numpy(), margins["preserve"].numpy()
        kind = "physical margins normalized by declared tolerances"
    else:
        reach = np.where(labels["reach"], 1., -1.)
        preserve = np.where(labels["safe"], 1., -1.)
        kind = "recorded Boolean physical labels encoded as +/-1; ordinal robustness only"
    return {"reach": torch.from_numpy(reach), "preserve": torch.from_numpy(preserve),
            "safe": torch.from_numpy(preserve)}, kind


def _bound(value):
    scalar = float(value)
    return scalar if math.isfinite(scalar) else ("+inf" if scalar > 0 else "-inf")


def monitor_record(result):
    lower, upper = _bound(result.lower), _bound(result.upper)
    return {"status": result.status, "complete": bool(result.complete),
            "lower": lower, "upper": upper,
            "robustness": lower if lower == upper else None}


def _visual_traces(raw, thresholds):
    preserve = (thresholds["safe_threshold"] - raw["safe_distance"]) / thresholds["safe_scale"]
    reach = (thresholds["goal_threshold"] - raw["goal_distance"]) / thresholds["goal_scale"]
    return {"preserve": torch.as_tensor(preserve), "safe": torch.as_tensor(preserve),
            "reach": torch.as_tensor(reach)}


def _raw_record(extractor, ep, goal):
    if ep["rgb"].shape[1:] != (128, 128, 3):
        raise ValueError("Canonical benchmark predicates require 128x128 RGB images")
    safe, reach, roi, fallback = extractor.traces(ep["rgb"], goal)
    labels = _labels(ep)
    return {"safe_distance": safe, "goal_distance": reach,
            "safe_label": labels["safe"], "reach_label": labels["reach"],
            "preserve_roi": list(roi), "roi_fallback": bool(fallback)}


def _definition(method):
    return {"reach": "(goal_threshold - distance(current, goal_reference)) / goal_scale",
            "preserve": "(safe_threshold - distance(current, initial_reference)) / safe_scale",
            "safe": "alias of preserve", "preserve_reference": "episode initial RGB",
            "goal_reference": "dataset references/goal_rgb", "goal_roi": list(GOAL_BOX),
            "preserve_roi": "mug_box(initial RGB), fixed throughout each episode",
            "method": method, "global_roi_behavior": "global method ignores image ROIs",
            "red_pixels_goal_method": "pixels" if method == "red_pixels" else None,
            "distance": "cosine" if method in ("global", "spatial") else "RMS pixel difference",
            "time_semantics": "inclusive recorded control_step / control_freq_hz, relative to first observation"}


def _validate_artifact(artifact, dataset_hash, checkpoint_hash, methods):
    if artifact.get("schema_version") != 1 or artifact.get("calibration_split") != "val":
        raise ValueError("Calibration artifact must use supported schema and validation split")
    if artifact.get("dataset_sha256") != dataset_hash:
        raise ValueError("Calibration dataset hash differs; recalibrate explicitly")
    if any(method in ("global", "spatial") for method in methods) and artifact.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Calibration checkpoint hash differs")
    for method in methods:
        if method not in artifact.get("methods", {}):
            raise ValueError(f"Calibration artifact lacks {method}")
        if artifact["methods"][method].get("predicate_definitions") != _definition(method):
            raise ValueError(f"Calibration predicate definitions differ for {method}")
        thresholds = artifact["methods"][method]["thresholds"]
        for name in ("safe_threshold", "goal_threshold", "safe_scale", "goal_scale"):
            if name not in thresholds or not math.isfinite(float(thresholds[name])):
                raise ValueError(f"Invalid {method} calibration {name}")
        if min(thresholds["safe_scale"], thresholds["goal_scale"]) <= 0:
            raise ValueError("Calibration scales must be positive")


def _summary(rows, method):
    pairs = [(row["methods"][method], row["physical"]) for row in rows]
    decided = [(visual, physical) for visual, physical in pairs
               if visual["status"] in ("satisfied", "violated")
               and physical["status"] in ("satisfied", "violated")]
    return {"episodes": len(rows),
            "visual_status_counts": {status: sum(a["status"] == status for a, _ in pairs)
                                     for status in ("satisfied", "violated", "pending", "boundary")},
            "physical_status_counts": {status: sum(b["status"] == status for _, b in pairs)
                                       for status in ("satisfied", "violated", "pending", "boundary")},
            "decided_comparisons": len(decided),
            "decided_agreement": (sum(a["status"] == b["status"] for a, b in decided) / len(decided)
                                  if decided else None),
            "false_satisfied_count": sum(a["status"] == "satisfied" and b["status"] == "violated"
                                          for a, b in decided),
            "note": "Pending and zero-boundary cases excluded from decided comparisons; no quantitative visual/physical score comparison"}


def run(args, *, extractor_factory=VisualFeatures):
    methods = list(dict.fromkeys(args.methods))
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"Methods must be selected from {METHODS}")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    if args.max_calibration_episodes is not None and args.max_calibration_episodes < 1:
        raise ValueError("max_calibration_episodes must be positive")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(method in ("global", "spatial") for method in methods) and args.checkpoint is None:
        raise ValueError("global and spatial require --checkpoint")
    formula = json.loads(Path(args.formula).read_text()) if args.formula else default_formula(args.deadline, args.hold)
    unsupported = predicate_names(formula) - ATOMS
    if unsupported:
        raise ValueError(f"CLI calibrated atoms are reach, preserve, safe; unsupported: {sorted(unsupported)}. "
                         "Use VisualSTLChecker for explicit custom predicates; no physical labels are inferred for them.")
    horizon = formula_horizon(formula)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    dataset_hash = sha256(args.dataset)
    checkpoint_hash = sha256(args.checkpoint) if args.checkpoint else None
    with h5py.File(args.dataset, "r") as data:
        frequency, simulator_config = _metadata(data)
        validation_keys = _keys(data, "val", args.max_calibration_episodes)
        test_keys = _keys(data, "test", args.max_episodes)
        if not test_keys or (not validation_keys and args.calibration_artifact is None):
            raise ValueError("Dataset needs test episodes and validation episodes for fresh calibration")
        if set(validation_keys) & set(test_keys):
            raise ValueError("Validation and test episodes must be disjoint")
        goal = np.asarray(data["references/goal_rgb"])
        extractors = {method: extractor_factory(method, args.checkpoint, args.device, args.batch_size)
                      for method in methods}
        if args.calibration_artifact:
            artifact = json.loads(Path(args.calibration_artifact).read_text())
            _validate_artifact(artifact, dataset_hash, checkpoint_hash, methods)
            artifact_ids = artifact.get("validation_episode_ids", [])
            all_validation_keys = set(_keys(data, "val", None))
            if not artifact_ids or len(artifact_ids) != len(set(artifact_ids)) or not set(artifact_ids) <= all_validation_keys:
                raise ValueError("Artifact calibration episodes must be distinct actual validation episodes")
        else:
            artifact = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "dataset": str(Path(args.dataset).resolve()), "dataset_sha256": dataset_hash,
                        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
                        "checkpoint_sha256": checkpoint_hash, "calibration_split": "val",
                        "checker_source_sha256": sha256(__file__),
                        "feature_source_sha256": sha256(Path(__file__).with_name("visual_predicate_benchmark.py")),
                        "validation_episode_ids": validation_keys, "control_freq_hz": frequency,
                        "operating_point": args.operating_point, "methods": {},
                        "selection_rule": "Existing benchmark validation-only per-frame calibration; frozen before test extraction",
                        "physical_definitions": simulator_config,
                        "limitations": "Learned/visual scores are proxies; calibration does not certify new horizons or continuous-time safety"}
            for method, extractor in extractors.items():
                records = []
                for key in validation_keys:
                    episode_timestamps(data["trajectories"][key], frequency)
                    records.append(_raw_record(extractor, data["trajectories"][key], goal))
                artifact["methods"][method] = {
                    "thresholds": calibration(records)[args.operating_point],
                    "predicate_definitions": _definition(method),
                    "calibration_frames": sum(len(row["safe_distance"]) for row in records)}
                print(f"Calibrated {method} on {len(records)} validation episodes", flush=True)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "calibration.json", artifact)
        frozen_hash = sha256(output / "calibration.json")
        write_json(output / "formula.json", formula)
        (output / "traces").mkdir()
        rows = []
        for index, key in enumerate(test_keys):
            ep = data["trajectories"][key]
            timestamps = episode_timestamps(ep, frequency)
            physical, physical_kind = physical_predicates(ep, simulator_config)
            physical_result = monitor_record(evaluate_timed_formula(formula, physical, timestamps))
            row = {"episode_id": key, "split": "test", "duration_seconds": float(timestamps[-1] - timestamps[0]),
                   "observations": len(timestamps), "physical": physical_result,
                   "physical_score_kind": physical_kind, "methods": {}}
            trace = {"episode_id": key, "timestamps_seconds": timestamps.tolist(),
                     "physical_predicates": {name: value.tolist() for name, value in physical.items()},
                     "physical": physical_result, "methods": {}}
            for method, extractor in extractors.items():
                raw = _raw_record(extractor, ep, goal)
                predicates = _visual_traces(raw, artifact["methods"][method]["thresholds"])
                result = monitor_record(evaluate_timed_formula(formula, predicates, timestamps))
                row["methods"][method] = result
                trace["methods"][method] = {"result": result, "preserve_roi": raw["preserve_roi"],
                                           "roi_fallback": raw["roi_fallback"],
                                           "predicates": {name: value.tolist() for name, value in predicates.items()}}
            trace_name = f"episode_{index:05d}.json"
            row["trace_file"] = f"traces/{trace_name}"
            write_json(output / "traces" / trace_name, trace)
            rows.append(row)
            print(f"Checked {key}: {len(timestamps)} observations over {row['duration_seconds']:.3f} seconds", flush=True)
    if sha256(output / "calibration.json") != frozen_hash:
        raise RuntimeError("Frozen calibration artifact changed during evaluation")
    report = {"schema_version": 1, "dataset_sha256": dataset_hash, "checkpoint_sha256": checkpoint_hash,
              "calibration_sha256": frozen_hash, "calibration_source": str(args.calibration_artifact) if args.calibration_artifact else "validation episodes",
              "formula": formula, "formula_horizon_seconds": horizon, "control_freq_hz": frequency,
              "evaluation_split": "test", "evaluation_episode_ids": test_keys,
              "scope": "Observed recorded trajectories; sampled-time monitoring; no forecast or closed-loop claims",
              "max_evaluation_episodes": args.max_episodes,
              "max_calibration_episodes": args.max_calibration_episodes,
              "methods": {method: _summary(rows, method) for method in methods}, "episodes": rows}
    write_json(output / "report.json", report)
    print(json.dumps(report["methods"], indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["pixels", "red_pixels"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formula", type=Path, help="Timed formula JSON; otherwise build reach/preserve task")
    parser.add_argument("--deadline", type=float, default=5.0)
    parser.add_argument("--hold", type=float, default=0.0, help="Seconds of continuous sampled hold completed by deadline")
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--operating-point", choices=("balanced", "false_safe_05"), default="balanced")
    parser.add_argument("--max-episodes", type=int, help="Limit test episodes, primarily for smoke runs")
    parser.add_argument("--max-calibration-episodes", type=int, help="Limit validation episodes; both classes must remain represented")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
