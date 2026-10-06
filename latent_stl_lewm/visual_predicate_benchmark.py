"""Reference-conditioned visual STL benchmark, with frozen validation calibration.

Observed-trajectory ranking is an offline diagnostic, not CEM or a control policy.
Simulator labels calibrate two scalar thresholds on validation only; they never
enter feature extraction or the predicate's runtime inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from .evaluate import classification, sha256, write_json
from .reference_predicates import calibrate_threshold

METHODS = ("global", "spatial", "pixels", "red_pixels")
GOAL_BOX = (80, 32, 116, 72)  # Supplied goal image's gripper region; fixed before evaluation.


def mug_box(initial_rgb):
    """Locate this red mug once in the initial RGB; never follow its later motion.

    The color prior and broad search rectangle are explicit visual task inputs.
    No object positions, simulator segmentation, or future images are accepted.
    """
    rgb = np.asarray(initial_rgb, dtype=np.float32)
    r, g, b = rgb.transpose(2, 0, 1)
    mask = (r > 35) & (r > 1.4 * g) & (b > .85 * g)
    region = np.zeros(mask.shape, bool)
    region[40:106, 30:100] = True
    labels, n = ndimage.label(mask & region)
    sizes = np.bincount(labels.ravel())[1:]
    if n == 0 or sizes.max() < 6:
        return (48, 50, 84, 94), True
    ys, xs = np.where(labels == sizes.argmax() + 1)
    return (max(0, int(xs.min()) - 5), max(0, int(ys.min()) - 5),
            min(128, int(xs.max()) + 6), min(128, int(ys.max()) + 6)), False


def token_weights(box, size=128, patch=16):
    """Pixel overlap weights: retain spatial correspondence, never pool tokens."""
    x0, y0, x1, y1 = box
    mask = np.zeros((size, size), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1
    weights = mask.reshape(size // patch, patch, size // patch, patch).mean((1, 3)).ravel()
    if weights.sum() == 0:
        raise ValueError("Empty visual region")
    return weights / weights.sum()


def temporal_scores(safe, reach, hold=10):
    """Inclusive sampled-time STL; all observed past violations remain in G."""
    safe, reach = np.asarray(safe), np.asarray(reach)
    if safe.ndim != 1 or safe.shape != reach.shape or not len(safe):
        raise ValueError("Expected matching nonempty predicate traces")
    if not np.isfinite(safe).all() or not np.isfinite(reach).all():
        raise ValueError("Non-finite predicate trace")
    sustained = (np.lib.stride_tricks.sliding_window_view(reach, hold).min(-1).max()
                 if len(reach) >= hold else -1e9)
    return {"stl": float(min(safe.min(), reach.max())),
            "reach_and_hold": float(min(safe.min(), sustained)),
            "safe_min": float(safe.min()), "reach_max": float(reach.max()),
            "terminal": float(reach[-1]),
            "weighted_penalty": float(reach.max() - np.maximum(-safe, 0).mean())}


def detection(safe_margin, safe_labels):
    """First alarm after violation, with prior false alarms recorded separately."""
    bad = np.flatnonzero(~np.asarray(safe_labels, bool))
    if not len(bad):
        return {"violation_frame": None, "detection_delay_frames": None,
                "missed_violation": False, "pre_violation_false_alarm": bool((safe_margin < 0).any())}
    onset = int(bad[0])
    alarms = np.flatnonzero(np.asarray(safe_margin)[onset:] < 0)
    return {"violation_frame": onset,
            "detection_delay_frames": int(alarms[0]) if len(alarms) else None,
            "missed_violation": not bool(len(alarms)),
            "pre_violation_false_alarm": bool((np.asarray(safe_margin)[:onset] < 0).any())}


class VisualFeatures:
    def __init__(self, method, checkpoint, device="cuda", batch_size=32):
        if method not in METHODS:
            raise ValueError(method)
        self.method, self.device, self.batch_size = method, device, batch_size
        self.model = None
        if method in ("global", "spatial"):
            from .world_model import load_checkpoint
            self.model, _ = load_checkpoint(checkpoint, device)
            self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, rgb):
        chunks = []
        for begin in range(0, len(rgb), self.batch_size):
            batch = torch.as_tensor(np.asarray(rgb[begin:begin+self.batch_size]), device=self.device)
            if self.method in ("pixels", "red_pixels"):
                value = batch.float() / 255.
                if self.method == "red_pixels":
                    # Continuous red evidence suppresses brown table/gray arm.
                    r, g, b = value.unbind(-1)
                    value = (F.relu(r - 1.25*g) * torch.clamp(b / (g + .01) - .75, 0, 1))[..., None]
                chunks.append(value.cpu().numpy())
            else:
                batch = batch.permute(0, 3, 1, 2)
                with torch.autocast(torch.device(self.device).type, dtype=torch.float16,
                                    enabled=str(self.device).startswith("cuda")):
                    if self.method == "global":
                        value = self.model.encode(batch)
                    else:
                        pixels = self.model._preprocess(batch)[:, 0]
                        value = self.model.model.encoder(pixels).last_hidden_state[:, 1:]
                chunks.append(F.normalize(value.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks)

    def distance(self, values, reference, box):
        if self.method == "global":
            return np.maximum(0, 1 - np.sum(values * reference, axis=-1))
        if self.method == "spatial":
            return np.maximum(0, 1 - np.sum(values * reference, axis=-1)) @ token_weights(box)
        x0, y0, x1, y1 = box
        return np.sqrt(np.mean((values[:, y0:y1, x0:x1] - reference[y0:y1, x0:x1])**2,
                               axis=(1, 2, 3)))

    def traces(self, rgb, goal_rgb):
        box, fallback = mug_box(rgb[0])
        features = self.encode(rgb)
        goal = self.encode(goal_rgb[None])[0]
        safe_distance = self.distance(features, features[0], box)
        # Red-only features are a mug comparator, not a reaching representation.
        if self.method == "red_pixels":
            a = np.asarray(rgb, dtype=np.float32) / 255
            g = np.asarray(goal_rgb, dtype=np.float32) / 255
            x0, y0, x1, y1 = GOAL_BOX
            goal_distance = np.sqrt(np.mean((a[:, y0:y1, x0:x1] - g[y0:y1, x0:x1])**2, (1, 2, 3)))
        else:
            goal_distance = self.distance(features, goal, GOAL_BOX)
        return safe_distance, goal_distance, box, fallback


def iter_episodes(dataset, splits):
    with h5py.File(dataset, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"Incomplete dataset: {dataset}")
        goal = handle["references/goal_rgb"][:]
        for key, ep in handle["trajectories"].items():
            if str(ep.attrs["split"]) not in splits:
                continue
            yield key, ep, goal


def calibration(records, maximum_false_safe=.05):
    """Two declared validation operating points; neither optimized on test data."""
    s = np.concatenate([x["safe_distance"] for x in records])
    g = np.concatenate([x["goal_distance"] for x in records])
    sl = np.concatenate([x["safe_label"] for x in records]).astype(bool)
    gl = np.concatenate([x["reach_label"] for x in records]).astype(bool)
    balanced = -calibrate_threshold(-s, sl)
    goal = -calibrate_threshold(-g, gl)
    unique = np.unique(s)
    candidates = np.r_[np.nextafter(unique[0], -np.inf), (unique[:-1]+unique[1:])/2,
                       np.nextafter(unique[-1], np.inf)]
    # Largest threshold meeting a per-frame validation false-safe budget.
    if sl.all() or not sl.any():
        raise ValueError("Safety calibration requires both classes")
    false_safe = np.searchsorted(np.sort(s[~sl]), candidates, side="right") / (~sl).sum()
    constrained = float(candidates[false_safe <= maximum_false_safe][-1])
    scale_s, scale_g = max(float(np.std(s)), 1e-6), max(float(np.std(g)), 1e-6)
    return {"balanced": {"safe_threshold": balanced, "goal_threshold": goal,
                          "safe_scale": scale_s, "goal_scale": scale_g},
            "false_safe_05": {"safe_threshold": constrained, "goal_threshold": goal,
                               "safe_scale": scale_s, "goal_scale": scale_g}}


def episode_record(key, ep, goal, extractor):
    s, g, box, fallback = extractor.traces(ep["rgb"], goal)
    d = ep["diagnostics"]
    return {"id": key, "bank": str(ep.attrs["bank_id"]), "policy": str(ep.attrs["policy"]),
            "scenario": str(ep.attrs.get("scenario", "historical_baseline")),
            "split": str(ep.attrs["split"]), "kind": str(ep.attrs.get("kind", "trajectory")),
            "safe_distance": s, "goal_distance": g,
            "safe_label": d["safe"][:].astype(bool), "reach_label": d["reach"][:].astype(bool),
            "object_displacement": d["object_displacement"][:],
            "object_rotation_deg": d["object_rotation_deg"][:],
            "goal_error": d["goal_error"][:], "mug_box": box, "roi_fallback": fallback}


def evaluate_records(records, thresholds):
    rows, safety, reach, safe_labels, reach_labels = [], [], [], [], []
    for rec in records:
        s = (thresholds["safe_threshold"] - rec["safe_distance"]) / thresholds["safe_scale"]
        g = (thresholds["goal_threshold"] - rec["goal_distance"]) / thresholds["goal_scale"]
        sl, gl = rec["safe_label"], rec["reach_label"]
        real = temporal_scores(np.where(sl, 1., -1.), np.where(gl, 1., -1.))
        row = {k: rec[k] for k in ("id", "bank", "policy", "scenario", "kind", "mug_box", "roi_fallback")}
        row.update(temporal_scores(s, g))
        row.update(detection(s, sl))
        row.update(physical_safe=bool(sl.all()), physical_reach=bool(gl.any()),
                   physical_success=bool(sl.all() and gl.any()),
                   physical_hold=real["reach_and_hold"] >= 0,
                   frame_safe=classification(s, sl), frame_reach=classification(g, gl))
        rows.append(row)
        safety.append(s); reach.append(g); safe_labels.append(sl); reach_labels.append(gl)
    summary = {"episodes": len(rows), "frames": sum(map(len, safety)),
               "frame_safe": classification(np.concatenate(safety), np.concatenate(safe_labels)),
               "frame_reach": classification(np.concatenate(reach), np.concatenate(reach_labels)),
               "trajectory_safe": classification([x["safe_min"] for x in rows], [x["physical_safe"] for x in rows]),
               "trajectory_stl": classification([x["stl"] for x in rows], [x["physical_success"] for x in rows]),
               "trajectory_hold": classification([x["reach_and_hold"] for x in rows], [x["physical_hold"] for x in rows]),
               "unsafe_episodes": sum(x["violation_frame"] is not None for x in rows),
               "missed_violations": sum(x["missed_violation"] for x in rows),
               "roi_fallbacks": sum(x["roi_fallback"] for x in rows)}
    delays = [x["detection_delay_frames"] for x in rows if x["detection_delay_frames"] is not None]
    summary["detection_delay_frames"] = {"median": float(np.median(delays)) if delays else None,
                                           "max": max(delays) if delays else None,
                                           "note": "Conditional on detection; misses reported separately"}
    banks = defaultdict(list)
    for row in rows:
        if row["kind"] == "trajectory":
            banks[row["bank"]].append(row)
    ranking = {}
    for objective in ("terminal", "reach_max", "weighted_penalty", "stl", "reach_and_hold"):
        selected = [max(bank, key=lambda x: (x[objective], x["id"])) for bank in banks.values()]
        ranking[objective] = {"banks": len(selected),
                              "safe_reach": sum(x["physical_success"] for x in selected),
                              "unsafe": sum(not x["physical_safe"] for x in selected),
                              "safe_hold": sum(x["physical_hold"] for x in selected),
                              "selected_ids": [x["id"] for x in selected]}
    summary["offline_candidate_ranking"] = ranking
    summary["oracle_bank_safe_reach"] = sum(any(x["physical_success"] for x in b) for b in banks.values())
    # Resample entire initial-state banks, never correlated frames independently.
    rng = np.random.default_rng(624)
    groups = defaultdict(list)
    for rec, s, g in zip(records, safety, reach):
        groups[rec["bank"]].append((rec, s, g))
    group_counts = []
    for group in groups.values():
        false_safe = sum(int(((s >= 0) & ~rec["safe_label"]).sum()) for rec, s, _ in group)
        negative = sum(int((~rec["safe_label"]).sum()) for rec, _, _ in group)
        agree = sum(int((temporal_scores(s, g)["stl"] >= 0) == bool(rec["safe_label"].all() and rec["reach_label"].any())) for rec, s, g in group)
        group_counts.append([false_safe, negative, agree, len(group)])
    counts = np.asarray(group_counts)
    boot = counts[rng.integers(0, len(counts), (500, len(counts)))].sum(1)
    fs = boot[boot[:, 1] > 0, 0] / boot[boot[:, 1] > 0, 1]
    summary["bank_bootstrap_95ci"] = {"frame_false_safe": np.quantile(fs, [.025, .975]).tolist() if len(fs) else None,
                                       "stl_agreement": np.quantile(boot[:, 2]/boot[:, 3], [.025, .975]).tolist(),
                                       "independent_banks": len(counts), "replicates": 500}
    return summary, rows


def process_dataset(extractor, dataset, splits, trace_path):
    records = []
    with h5py.File(trace_path, "x") as output:
        for key, ep, goal in iter_episodes(dataset, splits):
            rec = episode_record(key, ep, goal, extractor)
            group = output.create_group(key)
            for name, value in rec.items():
                if isinstance(value, np.ndarray):
                    group.create_dataset(name, data=value)
                else:
                    group.attrs[name] = value
            records.append(rec)
            if len(records) % 16 == 0:
                print(json.dumps({"phase": "encode", "dataset": str(dataset), "episodes": len(records)}), flush=True)
        output.attrs["complete"] = True
    if not records:
        raise ValueError("Empty evaluation set")
    return records


def run(args):
    torch.set_num_threads(4)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    extractor = VisualFeatures(args.method, args.checkpoint, args.device, args.batch_size)
    if args.calibration:
        frozen = json.loads(Path(args.calibration).read_text())
        if frozen["method"] != args.method or frozen["checkpoint_sha256"] != sha256(args.checkpoint):
            raise ValueError("Frozen representation mismatch")
    else:
        records = process_dataset(extractor, args.dataset, {"val"}, out / "validation_traces.h5")
        frozen = {"method": args.method, "checkpoint_sha256": sha256(args.checkpoint),
                  "dataset_sha256": sha256(args.dataset), "calibration_split": "val",
                  "goal_box": GOAL_BOX, "operating_points": calibration(records),
                  "supervision": "Validation physical safe/reach labels calibrate thresholds; RGB-only distances at runtime",
                  "source_sha256": sha256(__file__), "test_used_for_fitting": False}
        write_json(out / "calibration.json", frozen)
        write_json(out / "validation.json", {name: evaluate_records(records, cfg)[0]
                                            for name, cfg in frozen["operating_points"].items()})
    records = process_dataset(extractor, args.dataset, {"test"}, out / "test_traces.h5")
    report = {"method": args.method, "dataset": str(Path(args.dataset).resolve()),
              "dataset_sha256": sha256(args.dataset), "calibration": frozen,
              "evaluation": "Observed RGB monitoring and offline candidate-bank ranking; no action planning",
              "operating_points": {}}
    for name, cfg in frozen["operating_points"].items():
        summary, rows = evaluate_records(records, cfg)
        summary["by_scenario"] = {scenario: evaluate_records([x for x in records if x["scenario"] == scenario], cfg)[0]
                                  for scenario in sorted({x["scenario"] for x in records})}
        report["operating_points"][name] = summary
        write_json(out / f"episodes_{name}.json", rows)
    probe_path = Path(args.dataset).parent / "observability.npz"
    if probe_path.exists():
        with np.load(probe_path) as probe, h5py.File(args.dataset, "r") as h:
            sd, gd, box, fallback = extractor.traces(probe["rgb"], h["references/goal_rgb"][:])
            conditions = json.loads(probe_path.with_suffix(".json").read_text())["conditions"]
            report["static_observability"] = {
                "kind": "Static pose interventions, not simulated action sequences",
                "mug_box": box, "roi_fallback": fallback,
                "rows": [{**condition, "safe_distance": float(sd[i]),
                          "safe_label": bool(probe["safe"][i]),
                          "displacement_m": float(probe["object_displacement"][i]),
                          "rotation_deg": float(probe["object_rotation_deg"][i]),
                          "predicted_safe": {name: bool(sd[i] <= cfg["safe_threshold"])
                                             for name, cfg in frozen["operating_points"].items()}}
                         for i, condition in enumerate(conditions)]}
    write_json(out / "report.json", report)
    print(json.dumps({"complete": True, "method": args.method, "output": str(out)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    for name in ("dataset", "checkpoint", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--calibration")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    run(p.parse_args())
