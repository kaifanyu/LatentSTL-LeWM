"""Check autoregressive alignment, gradient flow, and immutable image features."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from latent_stl_lewm.finetune_multistep import (
    LatentWindows, autoregressive_predictions, encode_splits, freeze_representation,
    prediction_losses, representation_hash,
)
from latent_stl_lewm.world_model import LeWMConfig, LeWMWorldModel, load_checkpoint


class AdditiveDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(history_size=3)

    def predict_next(self, z, actions):
        return z[:, -1] + self.gain * actions[:, -1]

    def predict_sequence(self, z, actions):
        return z + self.gain * actions


def test_autoregressive_actions_align_and_gradients_cross_predictions():
    model = AdditiveDynamics()
    observed = torch.tensor([[[0.], [2.], [5.]]])
    actions = torch.tensor([[[2.], [3.], [7.], [11.], [13.]]])
    future = autoregressive_predictions(model, observed, actions, context=3, future_steps=3)
    torch.testing.assert_close(future, torch.tensor([[[12.], [23.], [36.]]]))
    future[:, -1].sum().backward()
    assert model.gain.grad.item() == 31.0  # 7 + 11 + 13, including earlier predictions


def test_perfect_multistep_predictions_have_zero_loss():
    model = AdditiveDynamics()
    latents = torch.tensor([[[0.], [2.], [5.], [12.], [23.], [36.]]])
    actions = torch.tensor([[[2.], [3.], [7.], [11.], [13.]]])
    losses = prediction_losses(model, latents, actions, future_steps=3)
    assert losses["loss"].item() == 0.0
    with pytest.raises(ValueError, match="aligned actions"):
        autoregressive_predictions(model, latents, actions[:, :-1], 3, 3)


def test_windows_do_not_cross_episode_boundaries():
    episodes = {"a": {"latents": torch.arange(7).view(-1, 1), "actions": torch.arange(6).view(-1, 1)},
                "b": {"latents": torch.arange(100, 107).view(-1, 1), "actions": torch.arange(100, 106).view(-1, 1)}}
    windows = LatentWindows(episodes, context=3, future_steps=3)
    assert len(windows) == 4
    z, a = windows.batch([1, 2], "cpu")
    assert z.tolist() == [[[1], [2], [3], [4], [5], [6]], [[100], [101], [102], [103], [104], [105]]]
    torch.testing.assert_close(a, z[:, :-1])


def test_encoding_never_reads_test_frames_and_rejects_bank_leakage(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as data:
        data.attrs["complete"] = True
        for split, bank in (("train", "a"), ("val", "b"), ("test", "c")):
            episode = data.create_group(f"trajectories/{split}")
            episode.attrs.update(split=split, bank_id=bank)
            # Test data deliberately has no pixels/actions: any access must fail.
            if split != "test":
                episode.create_dataset("rgb", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
                episode.create_dataset("action", data=np.zeros((3, 1), dtype=np.float32))
    model = SimpleNamespace(encode=lambda rgb: rgb.float().mean((2, 3)))
    encoded = encode_splits(model, path, torch.device("cpu"), False, 2)
    assert set(encoded["train"]) == {"train"}
    assert set(encoded["val"]) == {"val"}
    with h5py.File(path, "r+") as data:
        data["trajectories/val"].attrs["bank_id"] = "a"
    with pytest.raises(ValueError, match="overlap"):
        encode_splits(model, path, torch.device("cpu"), False, 2)


def test_training_preserves_encoder_outputs_and_checkpoint_compatibility(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(32)
    model = LeWMWorldModel(LeWMConfig(
        image_size=32, patch_size=16, hidden_dim=24, embed_dim=24,
        encoder_depth=1, encoder_heads=3, encoder_mlp_dim=48,
        predictor_depth=1, predictor_heads=2, predictor_dim_head=12,
        predictor_mlp_dim=48, projector_hidden_dim=48, sigreg_num_proj=16,
    ))
    freeze_representation(model)
    rgb = torch.rand(2, 6, 3, 32, 32)
    with torch.no_grad():
        latents = model.encode(rgb)
    before = representation_hash(model)
    initial_prediction = model.predict_next(latents[:, 0], torch.zeros(2, 7)).detach()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    losses = prediction_losses(model, latents, torch.rand(2, 5, 7), future_steps=3)
    losses["loss"].backward()
    assert all(p.grad is None for p in model.model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.predictor.parameters())
    optimizer.step()
    assert representation_hash(model) == before
    torch.testing.assert_close(model.encode(rgb), latents, rtol=0, atol=0)
    assert not torch.equal(initial_prediction, model.predict_next(latents[:, 0], torch.zeros(2, 7)))
    path = tmp_path / "best.pt"
    torch.save({"model_config": model.config_dict(), "model_state_dict": model.state_dict()}, path)
    loaded, _ = load_checkpoint(path)
    assert representation_hash(loaded) == before
    torch.testing.assert_close(loaded.encode(rgb), latents, rtol=0, atol=0)
