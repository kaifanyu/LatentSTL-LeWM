"""Small action-conditioned LeWM adapter using the unmodified official modules.

Inputs are raw RGB in [0, 1] (or uint8); actions retain the environment's
convention. The trained representation is the projected CLS token, as in LeWM.
No simulator diagnostics enter this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]


def _official_module(filename: str):
    name = f"_latent_stl_official_lewm_{filename}"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "third_party" / "le-worldmodel" / f"{filename}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Official LeWM source missing: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class LeWMConfig:
    action_dim: int = 7
    image_size: int = 128
    patch_size: int = 16
    history_size: int = 3
    hidden_dim: int = 192
    embed_dim: int = 192
    encoder_depth: int = 12
    encoder_heads: int = 3
    encoder_mlp_dim: int = 768
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_dim_head: int = 64
    predictor_mlp_dim: int = 2048
    projector_hidden_dim: int = 2048
    dropout: float = 0.1
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    backend: str = "lewm"

    def __post_init__(self):
        if self.backend not in {"lewm", "frozen_encoder_learned_predictor"}:
            raise ValueError(f"Unsupported backend {self.backend}")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.history_size < 1 or self.action_dim < 1:
            raise ValueError("history_size and action_dim must be positive")


class LatentWorldModel(nn.Module):
    """Interface shared by latent dynamics backends."""

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict_next(self, z, action, state=None):
        raise NotImplementedError

    def rollout(self, z0, actions, states=None):
        raise NotImplementedError

    @staticmethod
    def latent_distance(z_a, z_b, metric="cosine"):
        a, b = z_a.float(), z_b.float()
        if metric in {"cosine", "global_cosine"}:
            return 1 - F.cosine_similarity(a, b, dim=-1)
        if metric in {"l2", "global_l2"}:
            return (a - b).norm(dim=-1)
        if metric in {"mse", "native"}:
            return (a - b).square().mean(dim=-1)
        if metric in {"normalized_l2", "norm_l2"}:
            return (F.normalize(a, dim=-1) - F.normalize(b, dim=-1)).norm(dim=-1)
        if metric in {"mean_token_cosine", "token_cosine"}:
            if a.ndim < 3:
                raise ValueError("token distance requires explicit [..., tokens, features]")
            return (1 - F.cosine_similarity(a, b, dim=-1)).mean(dim=-1)
        raise ValueError(f"Unknown latent metric: {metric}")


class LeWMWorldModel(LatentWorldModel):
    """Official LeWM JEPA/predictor/loss, with a self-contained ViT constructor.

    The ViT is the Hugging Face ViT-tiny configuration used by the upstream
    stable-pretraining factory. Patch size is 16 to tile the requested 128 input.
    Training predicts each of three context frames' successor. Both sides receive
    gradients; SIGReg regularizes the entire four-frame embedding sequence.
    """

    latent_layout = "global"

    def __init__(self, config: LeWMConfig | dict[str, Any] | None = None):
        super().__init__()
        from transformers import ViTConfig, ViTModel

        self.config = config if isinstance(config, LeWMConfig) else LeWMConfig(**(config or {}))
        c = self.config
        official = _official_module("module")
        encoder_cfg = ViTConfig(
            image_size=c.image_size, patch_size=c.patch_size,
            hidden_size=c.hidden_dim, num_hidden_layers=c.encoder_depth,
            num_attention_heads=c.encoder_heads, intermediate_size=c.encoder_mlp_dim,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            qkv_bias=True, layer_norm_eps=1e-12,
        )
        # Use PyTorch SDPA (math/memory-efficient fallback on the RTX 2080 Ti).
        encoder_cfg._attn_implementation = "sdpa"
        encoder = ViTModel(encoder_cfg, add_pooling_layer=False, use_mask_token=False)
        predictor = official.ARPredictor(
            num_frames=c.history_size, input_dim=c.embed_dim,
            hidden_dim=c.hidden_dim, output_dim=c.hidden_dim,
            depth=c.predictor_depth, heads=c.predictor_heads,
            dim_head=c.predictor_dim_head, mlp_dim=c.predictor_mlp_dim,
            dropout=c.dropout, emb_dropout=0.0,
        )
        projector = official.MLP(c.hidden_dim, c.projector_hidden_dim,
                                 output_dim=c.embed_dim, norm_fn=nn.BatchNorm1d)
        pred_proj = official.MLP(c.hidden_dim, c.projector_hidden_dim,
                                 output_dim=c.embed_dim, norm_fn=nn.BatchNorm1d)
        self.model = _official_module("jepa").JEPA(
            encoder=encoder, predictor=predictor,
            action_encoder=official.Embedder(input_dim=c.action_dim, emb_dim=c.embed_dim),
            projector=projector, pred_proj=pred_proj,
        )
        self.sigreg = official.SIGReg(knots=c.sigreg_knots, num_proj=c.sigreg_num_proj)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        self.register_buffer("action_mean", torch.zeros(c.action_dim))
        self.register_buffer("action_std", torch.ones(c.action_dim))
        if c.backend == "frozen_encoder_learned_predictor":
            self.freeze_encoder()

    def freeze_encoder(self):
        for component in (self.model.encoder, self.model.projector):
            component.requires_grad_(False)
            component.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.backend == "frozen_encoder_learned_predictor":
            self.freeze_encoder()
        return self

    def _preprocess(self, obs):
        if obs.ndim not in (4, 5):
            raise ValueError("RGB must have shape B,C,H,W or B,T,C,H,W")
        obs = obs.unsqueeze(1) if obs.ndim == 4 else obs
        if obs.shape[-3] != 3 or obs.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError(f"Expected RGB channels and {self.config.image_size} square input")
        obs = obs.float().div(255) if obs.dtype == torch.uint8 else obs.float()
        return (obs - self.image_mean) / self.image_std

    def encode(self, obs):
        singleton_time = obs.ndim == 4
        pixels = self._preprocess(obs)
        emb = self.model.encode({"pixels": pixels})["emb"]
        return emb[:, 0] if singleton_time else emb

    def encode_spatial(self, obs):
        """Observed patch features B,N,D or B,T,N,D, excluding the CLS token.

        These features support object-region predicate checking. The existing
        autoregressive predictor still forecasts global projected CLS vectors;
        this method does not introduce a spatial dynamics model.
        """
        singleton_time = obs.ndim == 4
        pixels = self._preprocess(obs)
        batch, time = pixels.shape[:2]
        tokens = self.model.encoder(pixels.flatten(0, 1)).last_hidden_state[:, 1:]
        tokens = tokens.reshape(batch, time, *tokens.shape[1:])
        return tokens[:, 0] if singleton_time else tokens

    def _encode_actions(self, actions):
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError(f"Expected action dimension {self.config.action_dim}")
        return self.model.action_encoder((actions.float() - self.action_mean) / self.action_std)

    def predict_sequence(self, z, actions):
        if z.ndim != 3 or actions.ndim != 3 or z.shape[:2] != actions.shape[:2]:
            raise ValueError("Context latents and actions must have matching B,T axes")
        if not 1 <= z.shape[1] <= self.config.history_size:
            raise ValueError("Predictor context exceeds configured history_size")
        return self.model.predict(z, self._encode_actions(actions))

    def predict_next(self, z, action, state=None):
        if state is not None:
            raise ValueError("This image-only backend does not accept privileged state")
        z = z.unsqueeze(1) if z.ndim == 2 else z
        action = action.unsqueeze(1) if action.ndim == 2 else action
        return self.predict_sequence(z, action)[:, -1]

    def rollout(self, z0, actions, states=None):
        if states is not None:
            raise ValueError("This image-only backend does not accept privileged state")
        if z0.ndim != 2 or actions.ndim != 3 or z0.shape[0] != actions.shape[0]:
            raise ValueError("rollout expects z0[B,D] and actions[B,H,A]")
        latents = [z0]
        for t in range(actions.shape[1]):
            begin = max(0, t + 1 - self.config.history_size)
            z_context = torch.stack(latents[begin:], dim=1)
            latents.append(self.predict_next(z_context, actions[:, begin:t + 1]))
        return torch.stack(latents, dim=1)

    def training_loss(self, rgb, actions):
        if rgb.shape[1] != self.config.history_size + 1 or actions.shape[1] != self.config.history_size:
            raise ValueError("Training needs history_size+1 observations and history_size aligned actions")
        emb = self.encode(rgb)
        pred = self.predict_sequence(emb[:, :-1], actions)
        # The official loss does not detach the target. SIGReg prevents collapse.
        # FP32 is essential for characteristic-function trigonometry under FP16 AMP.
        with torch.autocast(device_type=emb.device.type, enabled=False):
            pred_loss = F.mse_loss(pred.float(), emb[:, 1:].float())
            reg = self.sigreg(emb.float().transpose(0, 1))
            loss = pred_loss + self.config.sigreg_weight * reg
            latent_std = emb.float().flatten(0, 1).std(dim=0).mean()
        return {"loss": loss, "pred_loss": pred_loss, "sigreg_loss": reg,
                "latent_std": latent_std}

    def config_dict(self):
        return asdict(self.config)


def load_checkpoint(path: str | Path, device="cpu"):
    """Load a state-dict checkpoint, returning (model in eval mode, metadata)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config", checkpoint.get("config", {}).get("model"))
    if config is None:
        raise ValueError("Checkpoint has no model_config")
    model = LeWMWorldModel(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def load_model(path: str | Path, device="cpu"):
    return load_checkpoint(path, device)[0]
