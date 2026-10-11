"""The same timestamped STL monitor for RGB and predicted visual features.

These predicates measure visual reference similarity, not physical centimeters,
collision clearance, or probabilities. Calibration happens elsewhere; this module
never loads simulator diagnostics or fits thresholds. References are explicit and
remain fixed when a trajectory is evaluated in chunks or during replanning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .visual_predicate_benchmark import GOAL_BOX, METHODS, VisualFeatures, mug_box


@dataclass(frozen=True)
class VisualPredicateSpec:
    """Signed ``(threshold - distance) / scale`` (opposite sign for far/avoid).

    ``roi`` uses half-open pixel coordinates (x0, y0, x1, y1). A nonnegative
    pixel mask can refine it; spatial features also accept a per-token mask.
    ``feature_method`` allows RGB goal matching with red-only object preservation.
    Threshold and scale must be calibrated for exactly this representation/ROI.
    """

    name: str
    relation: str = "near"
    reference: str = "goal"
    threshold: float = 0.0
    scale: float = 1.0
    roi: tuple[int, int, int, int] | None = None
    mask: Any = None
    feature_method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Predicate name must be nonempty")
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("Reference name must be nonempty")
        if self.relation not in {"near", "reach", "preserve", "far", "avoid"}:
            raise ValueError(f"Unsupported visual relation {self.relation!r}")
        if not math.isfinite(self.threshold) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Threshold must be finite and scale finite and positive")
        if self.feature_method is not None and self.feature_method not in METHODS:
            raise ValueError(f"Unsupported feature method {self.feature_method!r}")
        if self.roi is not None:
            if len(self.roi) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in self.roi):
                raise ValueError("ROI must contain four integer pixel coordinates")
            x0, y0, x1, y1 = self.roi
            if not 0 <= x0 < x1 or not 0 <= y0 < y1:
                raise ValueError("ROI must have positive area and nonnegative coordinates")
        if self.mask is not None:
            mask = torch.as_tensor(self.mask)
            if mask.ndim not in (1, 2) or not torch.isfinite(mask).all() or (mask < 0).any() or mask.sum() <= 0:
                raise ValueError("Mask must be a finite, nonnegative, nonempty pixel or token mask")


def _rgb_array(rgb, image_size, *, single=False):
    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()
    value = np.asarray(rgb)
    expected = (image_size, image_size, 3)
    if (single and value.shape != expected) or (not single and (value.ndim != 4 or value.shape[1:] != expected or len(value) == 0)):
        raise ValueError(f"Expected {'H,W,3 reference' if single else 'nonempty T,H,W,3 RGB'} with H=W={image_size}")
    if not np.isfinite(value).all():
        raise ValueError("RGB contains non-finite values")
    if value.dtype == np.uint8:
        return value.copy()
    if not np.issubdtype(value.dtype, np.floating) or value.min() < 0 or value.max() > 1:
        raise ValueError("RGB must be uint8 [0,255] or floating point [0,1]")
    # VisualFeatures expects the 0..255 convention even for floating input.
    return value.astype(np.float32) * 255.0


def canonical_reach_preserve_specs(calibration: Mapping[str, float], *, method="pixels",
                                   initial_rgb=None, goal_roi=GOAL_BOX,
                                   preserve_roi=None) -> list[VisualPredicateSpec]:
    """Match the historical benchmark's four-field calibration operating point.

    The default regions/color detector are specific to its 128x128 mug scene.
    A detector fallback remains explicitly marked in the returned metadata.
    For another scene, supply explicit object and goal regions.
    """
    if method not in METHODS:
        raise ValueError(f"Unsupported feature method {method!r}")
    required = {"safe_threshold", "goal_threshold", "safe_scale", "goal_scale"}
    if not required <= calibration.keys():
        raise ValueError(f"Calibration needs {sorted(required)}")
    fallback = False
    if method == "global":
        preserve_roi = goal_roi = None
    elif preserve_roi is None:
        if initial_rgb is None:
            raise ValueError("Object preservation needs initial_rgb or an explicit preserve_roi")
        initial = _rgb_array(initial_rgb, 128, single=True)
        preserve_roi, fallback = mug_box(initial)
    return [
        VisualPredicateSpec("preserve", "preserve", "initial", float(calibration["safe_threshold"]),
                            float(calibration["safe_scale"]), preserve_roi,
                            metadata={"roi_fallback": fallback, "reference_role": "fixed_initial_object"}),
        VisualPredicateSpec("reach", "near", "goal", float(calibration["goal_threshold"]),
                            float(calibration["goal_scale"]), goal_roi,
                            feature_method="pixels" if method == "red_pixels" else None),
    ]


class VisualSTLChecker:
    """Evaluate declared visual margins with one exact timed STL implementation.

    RGB inputs are T,H,W,3. Feature inputs are (...,T,D) for global,
    (...,T,N,D) for spatial, or (...,T,H,W,C) for pixels. ``references`` on the
    feature methods maps reference names to feature tensors, or to mappings of
    method names to tensors for mixed representations. It never denotes RGB.
    Global predicted latents cannot be passed as spatial feature predictions.

    Construction without a checkpoint is useful for precomputed/predicted
    features. A checkpoint (or injected VisualFeatures-compatible extractor) is
    only necessary when encoding global/spatial RGB.
    """

    def __init__(self, method: str, predicates: Sequence[VisualPredicateSpec],
                 references: Mapping[str, Any] | None = None, *, checkpoint=None,
                 device="cpu", batch_size=32, image_size=128, patch_size=16,
                 feature_extractor=None):
        if method not in METHODS:
            raise ValueError(f"Unsupported feature method {method!r}")
        if not isinstance(image_size, int) or image_size < 1 or not isinstance(patch_size, int) or patch_size < 1:
            raise ValueError("Image and patch sizes must be positive integers")
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        self.method, self.predicates = method, tuple(predicates)
        if not self.predicates or any(not isinstance(p, VisualPredicateSpec) for p in self.predicates):
            raise ValueError("Supply at least one VisualPredicateSpec")
        if len({p.name for p in self.predicates}) != len(self.predicates):
            raise ValueError("Predicate names must be unique")
        self.image_size, self.patch_size = image_size, patch_size
        self.checkpoint, self.device, self.batch_size = checkpoint, device, batch_size
        self._extractors = {} if feature_extractor is None else {method: feature_extractor}
        self._reference_rgb = {name: _rgb_array(value, image_size, single=True)
                               for name, value in (references or {}).items()}
        self._reference_features = {}
        for spec in self.predicates:
            method_i = spec.feature_method or self.method
            if method_i == "global" and (spec.roi is not None or spec.mask is not None):
                raise ValueError("Global latents cannot honor object-specific ROI or masks; use spatial features")
            if method_i == "spatial" and image_size % patch_size:
                raise ValueError("Spatial image size must be divisible by patch size")
            if spec.roi is not None and (spec.roi[2] > image_size or spec.roi[3] > image_size):
                raise ValueError("ROI extends beyond the image")
            if method_i != "global":
                self._weights(spec, method_i, device="cpu")

    def _extractor(self, method):
        if method not in self._extractors:
            if method in {"global", "spatial"} and self.checkpoint is None:
                raise ValueError("Encoding LeWM RGB requires a checkpoint; precomputed features do not")
            self._extractors[method] = VisualFeatures(method, self.checkpoint, self.device, self.batch_size)
        return self._extractors[method]

    def _encode(self, prepared_rgb, method):
        extractor = self._extractor(method)
        model = getattr(extractor, "model", None)
        if method in {"global", "spatial"} and model is not None:
            if model.config.image_size != self.image_size or model.config.patch_size != self.patch_size:
                raise ValueError("Checker image/patch settings must match the LeWM checkpoint")
            chunks = []
            for begin in range(0, len(prepared_rgb), self.batch_size):
                batch = torch.as_tensor(prepared_rgb[begin:begin + self.batch_size], device=self.device)
                batch = batch.permute(0, 3, 1, 2)
                if batch.dtype != torch.uint8:
                    batch = batch.float() / 255.0
                with torch.inference_mode(), torch.autocast(torch.device(self.device).type, dtype=torch.float16,
                                                           enabled=torch.device(self.device).type == "cuda"):
                    # Preserve the native dynamics representation. Normalization
                    # belongs in the cosine predicate, never before model.rollout.
                    encoded = model.encode(batch) if method == "global" else model.encode_spatial(batch)
                # Ordinary tensors outside inference_mode permit differentiable
                # comparisons against a separately predicted latent trajectory.
                chunks.append(encoded.float().clone())
            return torch.cat(chunks)
        value = extractor.encode(prepared_rgb)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def encode_rgb(self, rgb, *, method=None):
        """Encode RGB; LeWM returns raw native features, safe to feed dynamics."""
        method = method or self.method
        if method not in METHODS:
            raise ValueError(f"Unsupported feature method {method!r}")
        return self._encode(_rgb_array(rgb, self.image_size), method)

    def _weights(self, spec, method, *, device):
        size, patch = self.image_size, self.patch_size
        weight = torch.ones((size, size), device=device, dtype=torch.float32)
        if spec.roi is not None:
            x0, y0, x1, y1 = spec.roi
            weight.zero_()
            weight[y0:y1, x0:x1] = 1
        mask = None if spec.mask is None else torch.as_tensor(spec.mask, device=device, dtype=torch.float32)
        if mask is not None and mask.ndim == 2:
            if mask.shape != (size, size):
                raise ValueError("Pixel mask shape must match image H,W")
            weight = weight * mask
        if method == "spatial":
            weight = weight.reshape(size // patch, patch, size // patch, patch).mean((1, 3)).flatten()
        if mask is not None and mask.ndim == 1:
            if method != "spatial" or mask.shape != weight.shape:
                raise ValueError("Token masks require matching spatial token count")
            weight = weight * mask
        if weight.sum() <= 0:
            raise ValueError("Mask and ROI overlap must be nonempty")
        return weight / weight.sum()

    def _validate_features(self, features, method, *, reference=False):
        value = torch.as_tensor(features)
        if not value.is_floating_point():
            raise ValueError("Features must be floating point")
        if not torch.isfinite(value).all():
            raise ValueError("Features must be finite")
        value = value.float()
        feature_dims = 1 if method == "global" else 2 if method == "spatial" else 3
        if value.ndim < feature_dims + (0 if reference else 1):
            raise ValueError(f"Invalid {method} feature dimensions")
        if reference and value.ndim != feature_dims:
            raise ValueError("A reference is one feature state without a batch or time axis")
        if not reference and value.shape[-feature_dims-1] < 1:
            raise ValueError("Feature trace must contain at least one observation")
        if method in {"global", "spatial"}:
            if value.shape[-1] < 1 or (value.norm(dim=-1) < 1e-8).any():
                raise ValueError("Cosine features must have nonzero finite vectors")
            if method == "spatial" and value.shape[-2] != (self.image_size // self.patch_size) ** 2:
                raise ValueError("Spatial features need the configured patch-token grid, not a global latent")
        else:
            channels = 1 if method == "red_pixels" else 3
            if value.shape[-3:] != (self.image_size, self.image_size, channels):
                raise ValueError(f"Invalid {method} image feature shape")
            if (value < 0).any() or (value > 1).any():
                raise ValueError("Pixel features must be in [0,1]")
        return value

    def _reference(self, name, method, supplied):
        if supplied is not None and name in supplied:
            value = supplied[name]
            if isinstance(value, Mapping):
                if method not in value:
                    raise ValueError(f"Reference {name!r} is missing {method!r} features")
                value = value[method]
            return self._validate_features(value, method, reference=True)
        key = (name, method)
        if key not in self._reference_features:
            if name not in self._reference_rgb:
                raise ValueError(f"Missing explicit reference {name!r}; initial references are never reset implicitly")
            self._reference_features[key] = self._encode(self._reference_rgb[name][None], method)[0]
        return self._validate_features(self._reference_features[key], method, reference=True)

    def predicates_from_features(self, features, references=None):
        """Score encoded or predicted features without RGB or physical labels."""
        feature_map = features if isinstance(features, Mapping) else {self.method: features}
        validated, traces = {}, {}
        trace_shape = None
        for spec in self.predicates:
            method = spec.feature_method or self.method
            if method not in feature_map:
                raise ValueError(f"Missing {method!r} features required by predicate {spec.name!r}")
            if method not in validated:
                validated[method] = self._validate_features(feature_map[method], method)
            value = validated[method]
            reference = self._reference(spec.reference, method, references).to(value.device)
            if value.shape[-reference.ndim:] != reference.shape:
                raise ValueError(f"Feature/reference shape mismatch for predicate {spec.name!r}")
            if method == "global":
                distance = (1 - F.cosine_similarity(value, reference, dim=-1)).clamp_min(0)
            elif method == "spatial":
                token_distances = (1 - F.cosine_similarity(value, reference, dim=-1)).clamp_min(0)
                distance = (token_distances * self._weights(spec, method, device=value.device)).sum(-1)
            else:
                weight = self._weights(spec, method, device=value.device)
                weighted = (value - reference) * (weight / value.shape[-1]).sqrt().unsqueeze(-1)
                # vector_norm defines a zero gradient at an exact image match;
                # sqrt(sum(square)) otherwise produces NaNs at that point.
                distance = torch.linalg.vector_norm(weighted.flatten(-3), dim=-1)
            margin = (spec.threshold - distance) / spec.scale
            if spec.relation in {"far", "avoid"}:
                margin = -margin
            if trace_shape is not None and margin.shape != trace_shape:
                raise ValueError("All predicate feature streams must have identical batch/time axes")
            trace_shape = margin.shape
            traces[spec.name] = margin
        return traces

    def predicates_from_rgb(self, rgb):
        prepared = _rgb_array(rgb, self.image_size)
        methods = {spec.feature_method or self.method for spec in self.predicates}
        features = {method: self._encode(prepared, method) for method in methods}
        return self.predicates_from_features(features)

    def metadata(self, *, source="observed_rgb"):
        return {"method": self.method, "source": source,
                "checkpoint": None if self.checkpoint is None else str(self.checkpoint),
                "image_size": self.image_size, "patch_size": self.patch_size,
                "score_units": "calibrated visual-distance margin divided by positive scale",
                "physical_units": None, "runtime_physical_labels": False,
                "reference_policy": "explicit fixed references; never reset from a new chunk",
                "dynamics_validation": "not established by this checker",
                "roi_fallback_predicates": [p.name for p in self.predicates if p.metadata.get("roi_fallback", False)],
                "predicates": [{"name": p.name, "relation": p.relation, "reference": p.reference,
                                "threshold": p.threshold, "scale": p.scale,
                                "roi": None if p.roi is None else list(p.roi),
                                "has_mask": p.mask is not None,
                                "roi_status": "fallback_unverified" if p.metadata.get("roi_fallback", False) else "declared",
                                "feature_method": p.feature_method or self.method,
                                "metadata": dict(p.metadata)} for p in self.predicates]}

    def check_rgb(self, rgb, timestamps, formula, **evaluation_kwargs):
        from .timed_stl import evaluate_timed_formula
        traces = self.predicates_from_rgb(rgb)
        return {"predicate_traces": traces,
                "result": evaluate_timed_formula(formula, traces, timestamps, **evaluation_kwargs),
                "metadata": self.metadata(source="observed_rgb")}

    def check_features(self, features, timestamps, formula, *, references=None,
                       source="encoded_or_predicted_features", **evaluation_kwargs):
        from .timed_stl import evaluate_timed_formula
        traces = self.predicates_from_features(features, references)
        return {"predicate_traces": traces,
                "result": evaluate_timed_formula(formula, traces, timestamps, **evaluation_kwargs),
                "metadata": self.metadata(source=source)}
