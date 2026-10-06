"""Calibrated visual predicates using train-only safe and disturbed references.

Physical simulator labels supervise *offline* reference selection and validation
calibration. Runtime scoring sees visual latents only. Contact is not a negative
label: unsafe means displacement > 2 mm or rotation > 2 degrees in this dataset.
The full-history STL score is retained even after an irreversible past violation;
separate future-only scores support useful receding-horizon optimization.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class ReferencePredicates:
    safe_bank: torch.Tensor
    unsafe_bank: torch.Tensor
    safe_margin: float
    goal_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for bank in (self.safe_bank, self.unsafe_bank):
            if bank.ndim != 2 or len(bank) == 0 or not torch.isfinite(bank).all():
                raise ValueError("Reference banks must contain finite N,D features")
            if (bank.norm(dim=-1) < 1e-8).any():
                raise ValueError("Zero reference vectors cannot define cosine distance")
        if self.safe_bank.shape[1] != self.unsafe_bank.shape[1]:
            raise ValueError("Reference feature dimensions differ")
        if self.safe_bank.device != self.unsafe_bank.device:
            raise ValueError("Reference banks must share a device")
        if not np.isfinite(self.safe_margin) or not 0 < self.goal_threshold <= 2:
            raise ValueError("Invalid calibrated predicate thresholds")
        self.safe_bank = F.normalize(self.safe_bank.float(), dim=-1)
        self.unsafe_bank = F.normalize(self.unsafe_bank.float(), dim=-1)

    @classmethod
    def from_artifact(cls, path, device="cpu"):
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved.get("schema_version") != 1:
            raise ValueError("Unsupported reference predicate artifact")
        return cls(saved["safe_bank"], saved["unsafe_bank"],
                   float(saved["safe_margin"]), float(saved["goal_threshold"]),
                   saved.get("metadata", {}))

    def save(self, path):
        torch.save({"schema_version": 1, "safe_bank": self.safe_bank.cpu(),
                    "unsafe_bank": self.unsafe_bank.cpu(),
                    "safe_margin": self.safe_margin, "goal_threshold": self.goal_threshold,
                    "metadata": self.metadata}, path)

    def raw_safe(self, latents):
        """d(nearest unsafe) - d(nearest safe); positive favors preservation."""
        z = F.normalize(latents.float(), dim=-1)
        safe_similarity = torch.matmul(z, self.safe_bank.T).amax(-1)
        unsafe_similarity = torch.matmul(z, self.unsafe_bank.T).amax(-1)
        return safe_similarity - unsafe_similarity

    def score(self, latents, goal, history=None):
        """Score B,T,D rollouts; t=0 is current, history excludes current.

        Returned traces contain rollout entries only. ``stl`` includes history,
        current, and future. ``future_*`` excludes history AND current so a fixed
        past/current violation cannot make all future candidates tie. ``terminal``
        is negative goal distance (larger is better); ``goal_progress`` is its
        improvement over current. No physical measurements are accepted.
        """
        if latents.ndim != 3 or latents.shape[1] < 2:
            raise ValueError("Expected B,T,D with current and at least one future state")
        if goal.ndim != 1 or goal.shape[0] != latents.shape[-1]:
            raise ValueError("Goal must be one D-dimensional latent")
        if not torch.isfinite(latents).all() or not torch.isfinite(goal).all():
            raise ValueError("Non-finite latent inputs")
        distance = 1 - F.cosine_similarity(latents.float(), goal.float(), dim=-1)
        safe = self.raw_safe(latents) - self.safe_margin
        reach = self.goal_threshold - distance
        batch = latents.shape[0]
        past_safe = latents.new_full((batch,), float("inf"), dtype=torch.float32)
        past_reach = latents.new_full((batch,), -float("inf"), dtype=torch.float32)
        if history is not None:
            if history.ndim == 2:
                history = history[None]
            if history.ndim != 3 or history.shape[0] not in (1, batch) or history.shape[-1] != latents.shape[-1]:
                raise ValueError("History must be H,D or broadcastable B,H,D")
            if history.shape[1]:
                past_safe = (self.raw_safe(history) - self.safe_margin).amin(-1).expand(batch)
                past_reach = (self.goal_threshold - (1 - F.cosine_similarity(history.float(), goal.float(), dim=-1))).amax(-1).expand(batch)
        total_safe = torch.minimum(past_safe, safe.amin(-1))
        total_reach = torch.maximum(past_reach, reach.amax(-1))
        future_safe = safe[:, 1:].amin(-1)
        future_reach = reach[:, 1:].amax(-1)
        return {"safe_trace": safe, "preserve_trace": safe, "reach_trace": reach,
                "future_safe_trace": safe[:, 1:], "future_reach_trace": reach[:, 1:],
                "stl": torch.minimum(total_safe, total_reach),
                "future_stl": torch.minimum(future_safe, future_reach),
                "future_safe_min": future_safe, "future_reach_max": future_reach,
                "terminal": -distance[:, -1], "eventual": reach.amax(-1),
                "goal_progress": distance[:, 0] - distance[:, -1],
                "history_safe_min": past_safe, "history_reach_max": past_reach,
                "full_safe_min": total_safe, "full_reach_max": total_reach}


def calibrate_threshold(scores, labels, *, maximum=None):
    """Maximize validation balanced accuracy; ties favor negative specificity.

    The threshold is applied as score >= threshold. An optional upper bound can
    keep a supplied known-safe goal reference feasible. Test data never enter.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Calibration needs finite aligned scalar scores and labels")
    if not labels.any() or labels.all():
        raise ValueError("Calibration requires both physical classes")
    unique = np.unique(scores)
    thresholds = np.r_[np.nextafter(unique[0], -np.inf),
                       (unique[:-1] + unique[1:]) / 2,
                       np.nextafter(unique[-1], np.inf)]
    if maximum is not None:
        thresholds = np.r_[thresholds[thresholds <= maximum], maximum]
    order = np.argsort(scores)
    sorted_scores, sorted_labels = scores[order], labels[order]
    positive_prefix = np.r_[0, np.cumsum(sorted_labels)]
    negative_prefix = np.arange(len(labels) + 1) - positive_prefix
    positions = np.searchsorted(sorted_scores, thresholds, side="left")
    recall = 1 - positive_prefix[positions] / labels.sum()
    specificity = negative_prefix[positions] / (~labels).sum()
    balanced = (recall + specificity) / 2
    best = np.lexsort((thresholds, specificity, balanced))[-1]
    return float(thresholds[best])


@torch.no_grad()
def diverse_reference_indices(latents, eligible, count, device="cpu"):
    """Deterministic farthest-point coverage over eligible training embeddings."""
    indices = np.flatnonzero(eligible)
    if len(indices) == 0:
        raise ValueError("No training references for one physical class")
    if len(indices) <= count:
        return indices
    z = F.normalize(latents[indices].to(device).float(), dim=-1)
    # Begin at a central observed vector; every selected prototype is real data.
    first = int((z @ F.normalize(z.mean(0), dim=0)).argmax())
    chosen, nearest = [], torch.full((len(z),), float("inf"), device=device)
    index = first
    for _ in range(count):
        chosen.append(index)
        nearest = torch.minimum(nearest, 1 - z @ z[index])
        nearest[chosen] = -1
        index = int(nearest.argmax())
    return indices[np.asarray(chosen)]


def _label_str(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def encoder_feature_hash(model):
    """Stable encoder/projector and image-normalization hash across predictors."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(("model.encoder.", "model.projector.")) or name in {"image_mean", "image_std"}:
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def fit(checkpoint, dataset, output, *, device="cpu", bank_size=512, batch_size=64):
    from .evaluate import classification, sha256, write_json
    from .world_model import load_checkpoint

    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to replace {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    model, _ = load_checkpoint(checkpoint, device=device)
    model.requires_grad_(False)
    data_by_split = {}

    def encode(rgb):
        encoded = []
        for start in range(0, len(rgb), batch_size):
            image = torch.as_tensor(np.asarray(rgb[start:start + batch_size]), device=device).permute(0, 3, 1, 2)
            # CEM uses FP16 inference on Turing GPUs; calibrate matching features.
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                                enabled=torch.device(device).type == "cuda"):
                encoded.append(model.encode(image).float().cpu())
        return torch.cat(encoded)

    with h5py.File(dataset, "r") as data:
        goal = encode(data["references"]["goal_rgb"][()][None])[0]
        safe_reference = encode(data["references"]["safe_rgb"][()][None])[0]
        for split in ("train", "val", "test"):
            chunks, safe, reach, contact, ids, indices, episodes = [], [], [], [], [], [], []
            offset = 0
            for key, ep in data["trajectories"].items():
                if _label_str(ep.attrs["split"]) != split:
                    continue
                z = encode(ep["rgb"])
                count = len(z)
                chunks.append(z)
                safe.extend(ep["diagnostics"]["safe"][:].astype(bool).tolist())
                reach.extend(ep["diagnostics"]["reach"][:].astype(bool).tolist())
                contact.extend((ep["diagnostics"]["robot_mug_contact_count"][:] > 0).tolist())
                ids.extend([key] * count)
                indices.extend(range(count))
                episodes.append({"id": key, "begin": offset, "end": offset + count,
                                 "policy": _label_str(ep.attrs["policy"]),
                                 "bank_id": _label_str(ep.attrs["bank_id"])})
                offset += count
            if not chunks:
                raise ValueError(f"Dataset has no {split} split")
            data_by_split[split] = {"z": torch.cat(chunks), "safe": np.asarray(safe),
                                   "reach": np.asarray(reach), "contact": np.asarray(contact),
                                   "ids": ids, "indices": indices, "episodes": episodes}
            print(f"Encoded {split}: {offset} frames", flush=True)

    train, val = data_by_split["train"], data_by_split["val"]
    safe_indices = diverse_reference_indices(train["z"], train["safe"], bank_size, device)
    unsafe_indices = diverse_reference_indices(train["z"], ~train["safe"], bank_size, device)
    scorer = ReferencePredicates(train["z"][safe_indices].to(device), train["z"][unsafe_indices].to(device), 0., .1)

    def safe_scores(z):
        return torch.cat([scorer.raw_safe(batch.to(device)).cpu()
                          for batch in z.split(512)]).numpy()

    raw_validation = safe_scores(val["z"])
    goal_safe_raw = float(scorer.raw_safe(goal.to(device)))
    # The external supplied goal image is known-safe from collection. This is an
    # explicit reference constraint, not a threshold fitted to test outcomes.
    unconstrained_safe_margin = calibrate_threshold(raw_validation, val["safe"])
    scorer.safe_margin = calibrate_threshold(raw_validation, val["safe"], maximum=goal_safe_raw - 1e-6)
    raw_goal_validation = (F.cosine_similarity(val["z"], goal, dim=-1) - 1).numpy()
    scorer.goal_threshold = -calibrate_threshold(raw_goal_validation, val["reach"], maximum=-1e-6)

    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(checkpoint),
                  "encoder_feature_sha256": encoder_feature_hash(model),
                  "dataset": str(Path(dataset).resolve()), "dataset_sha256": sha256(dataset),
                  "supervision": "Offline simulator safe/reach labels select training references and calibrate validation thresholds; no diagnostics enter runtime scoring.",
                  "safety_label": "Per-frame displacement <= 0.002 m AND rotation <= 2 degrees; contact alone is not unsafe.",
                  "reference_split": "train", "calibration_split": "val", "diagnostic_split": "test",
                  "test_used_for_fitting": False, "bank_selection": "Farthest-point cosine coverage of real training frames in each class",
                  "safe_score": "cosine similarity to nearest safe reference minus cosine similarity to nearest unsafe reference minus margin",
                  "calibration": "Maximum validation balanced accuracy; ties favor specificity; safe margin constrained to accept external known-safe goal image",
                  "unconstrained_safe_margin": unconstrained_safe_margin,
                  "goal_safe_raw": goal_safe_raw,
                  "goal_reference_feasible": bool(goal_safe_raw - scorer.safe_margin >= 0),
                  "known_safe_goal_constraint_active": unconstrained_safe_margin > goal_safe_raw - 1e-6,
                  "bank_size_requested": bank_size,
                  "safe_references": [{"trajectory": train["ids"][int(i)] , "frame": train["indices"][int(i)]} for i in safe_indices],
                  "unsafe_references": [{"trajectory": train["ids"][int(i)], "frame": train["indices"][int(i)]} for i in unsafe_indices]}
    scorer.metadata = provenance
    report = {"provenance": provenance, "safe_margin": scorer.safe_margin,
              "goal_threshold": scorer.goal_threshold,
              "safe_reference_count": len(safe_indices), "unsafe_reference_count": len(unsafe_indices),
              "original_safe_reference_margin": float(scorer.raw_safe(safe_reference.to(device))) - scorer.safe_margin,
              "goal_reference_safe_margin": goal_safe_raw - scorer.safe_margin, "splits": {}}
    # Thresholds are frozen before these diagnostics, including the test split.
    for split, values in data_by_split.items():
        safety = safe_scores(values["z"]) - scorer.safe_margin
        reaching = scorer.goal_threshold - (1 - F.cosine_similarity(values["z"], goal, dim=-1)).numpy()
        physical, scores, details = [], [], []
        for episode in values["episodes"]:
            sl = slice(episode["begin"], episode["end"])
            rho = float(min(safety[sl].min(), reaching[sl].max()))
            good = bool(values["safe"][sl].all() and values["reach"][sl].any())
            physical.append(good)
            scores.append(rho)
            details.append({**episode, "stl": rho, "physical_success": good,
                            "physical_safe": bool(values["safe"][sl].all()),
                            "physical_reach": bool(values["reach"][sl].any())})
        safe_contact = values["safe"] & values["contact"]
        report["splits"][split] = {"frame_safety": classification(safety, values["safe"]),
                                   "frame_reach": classification(reaching, values["reach"]),
                                   "trajectory_success": classification(scores, physical),
                                   "safe_contact_frames": int(safe_contact.sum()),
                                   "safe_contact_acceptance": float((safety[safe_contact] >= 0).mean()) if safe_contact.any() else None,
                                   "trajectories": details}
    scorer.save(output)
    write_json(output.with_suffix(".json"), report)
    print(f"Saved {output}", flush=True)
    print({split: {name: values[name] for name in ("frame_safety", "frame_reach", "trajectory_success")}
           for split, values in report["splits"].items()}, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bank-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.bank_size < 1 or args.batch_size < 1:
        parser.error("Bank and batch sizes must be positive")
    fit(args.checkpoint, args.dataset, args.output, device=args.device,
        bank_size=args.bank_size, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
