"""Frozen inference adapters for exploratory LIBERO CEM experiments.

The official Cube model is a domain-transfer baseline, not a LIBERO model.
Its complete published state dict is loaded strictly, without retraining.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .evaluate import sha256
from .world_model import LeWMConfig, LeWMWorldModel, load_checkpoint


def cube_actions(raw):
    """Map OSC xyz/yaw/gripper commands into OGBench command units.

    Both environments scale xyz by .05 m. Yaw scales are .5 vs .3 rad.
    Roll/pitch are unsupported and must be fixed to zero. In these experiments
    yaw is also zero and both grippers are held open with command -1.
    This mapping does not claim equivalent robot dynamics or camera frames.
    """
    if raw.shape[-1] != 7:
        raise ValueError("Expected seven-dimensional LIBERO commands")
    if torch.any(raw[..., 3:5].abs() > 1e-6):
        raise ValueError("Cube transfer does not support roll/pitch commands")
    mapped = raw[..., [0, 1, 2, 5, 6]].clone()
    mapped[..., 3] *= .5 / .3
    if torch.any(mapped.abs() > 1 + 1e-6):
        raise ValueError("Mapped action exceeds OGBench normalized command bounds")
    return mapped


class CEMModel:
    def __init__(self, model, *, kind, action_repeat=5):
        self.model = model.eval().requires_grad_(False)
        self.kind = kind
        self.action_repeat = action_repeat
        self.native_repeat = 5 if kind == "official_cube_transfer" else 1
        if kind == "official_cube_transfer" and action_repeat != 5:
            raise ValueError("Official Cube checkpoint requires five-action blocks")
        self.context_latents = None
        self.context_actions = None

    def encode(self, rgb):
        rgb = rgb.float().div(255) if rgb.dtype == torch.uint8 else rgb
        size = self.model.config.image_size
        if rgb.shape[-2:] != (size, size):
            rgb = F.interpolate(rgb, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        return self.model.encode(rgb).float()

    def set_history(self, latents, actions):
        """Supply only observed RGB latents and previously executed actions."""
        if latents.ndim != 2 or actions.shape != (len(latents)-1, 7):
            raise ValueError("History must contain one more observation than actions")
        repeat = self.native_repeat
        count = min(self.model.config.history_size - 1, len(actions) // repeat)
        start = len(actions) - count * repeat
        self.context_latents = latents[start:len(actions):repeat].clone()
        selected = actions[start:]
        if self.kind == "official_cube_transfer":
            selected = cube_actions(selected).reshape(count, 25)
        self.context_actions = selected.clone()

    def rollout(self, z0, actions):
        """CEM controls are held for action_repeat physical control steps."""
        raw = actions.repeat_interleave(self.action_repeat, dim=1)
        if self.kind == "official_cube_transfer":
            native = cube_actions(raw).reshape(raw.shape[0], -1, 25)
        else:
            native = raw
        count = 0 if self.context_latents is None else len(self.context_latents)
        latents = [] if not count else list(self.context_latents[None].expand(z0.shape[0], -1, -1).unbind(1))
        latents.append(z0)
        if count:
            previous = self.context_actions[None].expand(z0.shape[0], -1, -1)
            inputs = torch.cat([previous, native], dim=1)
        else:
            inputs = native
        predictions = [z0]
        for t in range(native.shape[1]):
            end = count + t + 1
            begin = max(0, end - self.model.config.history_size)
            context = torch.stack(latents[begin:end], dim=1)
            pred = self.model.predict_sequence(context, inputs[:, begin:end])[:, -1].float()
            latents.append(pred)
            predictions.append(pred)
        return torch.stack(predictions, dim=1)


def load_cem_model(kind, checkpoint, *, device="cuda", action_repeat=5):
    checkpoint = Path(checkpoint)
    if kind == "local":
        model, saved = load_checkpoint(checkpoint, device=device)
        metadata = {"kind": kind, "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256(checkpoint), "model_config": saved["model_config"],
                    "training_updates": saved.get("optimizer_updates", saved.get("global_step")),
                    "action_normalization": "Saved local training mean/std",
                    "native_control_steps_per_prediction": 1}
    elif kind == "official_cube_transfer":
        cfg = json.loads((checkpoint / "config.json").read_text())
        if cfg["encoder"]["size"] != "tiny" or cfg["action_encoder"]["input_dim"] != 25:
            raise ValueError("Unsupported official checkpoint architecture")
        c = LeWMConfig(image_size=cfg["encoder"]["image_size"],
                       patch_size=cfg["encoder"]["patch_size"], action_dim=25,
                       history_size=cfg["predictor"]["num_frames"],
                       predictor_depth=cfg["predictor"]["depth"],
                       predictor_heads=cfg["predictor"]["heads"],
                       predictor_dim_head=cfg["predictor"]["dim_head"],
                       predictor_mlp_dim=cfg["predictor"]["mlp_dim"])
        model = LeWMWorldModel(c)
        weights = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
        model.model.load_state_dict(weights, strict=True)
        model.to(device).eval()
        metadata = {"kind": kind, "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256(checkpoint / "weights.pt"),
                    "config_sha256": sha256(checkpoint / "config.json"), "model_config": model.config_dict(),
                    "source": "https://huggingface.co/quentinll/lewm-cube",
                    "revision": checkpoint.name, "strict_weight_loading": True,
                    "weights_retrained": False, "native_control_steps_per_prediction": 5,
                    "action_conversion": "LIBERO xyz -> Cube xyz; yaw *= 0.5/0.3; gripper unchanged; roll/pitch must be zero. Flatten five successive 5D commands in temporal order.",
                    "action_normalization": "Identity (mean=0,std=1): original action normalization statistics are not included in the published model repository.",
                    "transfer_limitations": ["UR5e/Cube training differs from Panda/LIBERO robot, camera, scene, and dynamics",
                                             "Source action standardization is unavailable; identity scaling is an explicit approximation",
                                             "128px LIBERO images are resized to the checkpoint's native 224px",
                                             "Preservation can only be predicted at five-control-step boundaries"]}
    else:
        raise ValueError(kind)
    adapter = CEMModel(model, kind=kind, action_repeat=action_repeat)
    metadata["inference_parameters"] = sum(p.numel() for p in model.parameters())
    metadata["trainable_parameters_during_experiment"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return adapter, metadata
