"""Behavioral tests for temporal alignment, action conditioning and checkpoints."""
from dataclasses import asdict

import pytest
import torch

from latent_stl_lewm.world_model import LeWMConfig, LeWMWorldModel, load_checkpoint


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    return LeWMWorldModel(LeWMConfig(
        image_size=32, patch_size=16, hidden_dim=24, embed_dim=24,
        encoder_depth=1, encoder_heads=3, encoder_mlp_dim=48,
        predictor_depth=1, predictor_heads=2, predictor_dim_head=12,
        predictor_mlp_dim=48, projector_hidden_dim=48,
        sigreg_num_proj=32, dropout=0.0,
    ))


def test_rollout_aligns_current_frame_and_first_action(model):
    model.eval()
    z = model.encode(torch.rand(2, 3, 32, 32))
    actions = torch.rand(2, 5, 7) * 2 - 1
    result = model.rollout(z, actions)
    assert result.shape == (2, 6, 24)
    torch.testing.assert_close(result[:, 0], z)
    torch.testing.assert_close(result[:, 1], model.predict_next(z, actions[:, 0]))
    assert torch.isfinite(result).all()
    assert model.rollout(z, actions[:, :0]).shape == (2, 1, 24)
    with pytest.raises(ValueError, match="privileged"):
        model.rollout(z, actions, states=torch.zeros(2, 5, 7))


def test_official_loss_and_gradients_are_finite_and_action_conditioned(model):
    rgb = torch.rand(3, 4, 3, 32, 32)
    actions = torch.rand(3, 3, 7) * 2 - 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model.training_loss(rgb, actions)
    torch.testing.assert_close(output["loss"], output["pred_loss"] + model.config.sigreg_weight * output["sigreg_loss"])
    assert output["sigreg_loss"].dtype == torch.float32
    output["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.model.encoder.embeddings.patch_embeddings.projection.weight.grad.abs().sum() > 0
    optimizer.step()
    model.eval()
    z = model.encode(rgb[:, 0])
    first = model.predict_next(z, torch.zeros(3, 7))
    second = model.predict_next(z, torch.ones(3, 7))
    assert (first - second).abs().max() > 1e-7


def test_official_objective_matches_direct_computation(model):
    model.eval()
    rgb = torch.rand(2, 4, 3, 32, 32)
    actions = torch.rand(2, 3, 7)
    torch.manual_seed(37)
    output = model.training_loss(rgb, actions)
    emb = model.model.encode({"pixels": model._preprocess(rgb)})["emb"]
    predictions = model.model.predict(emb[:, :-1], model._encode_actions(actions))
    mse = (predictions - emb[:, 1:]).square().mean()
    torch.manual_seed(37)
    regularizer = model.sigreg(emb.transpose(0, 1))
    torch.testing.assert_close(output["loss"], mse + 0.09 * regularizer)


def test_uint8_and_float_inputs_are_equivalent(model):
    model.eval()
    pixels = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    torch.testing.assert_close(model.encode(pixels), model.encode(pixels.float() / 255))


def test_frozen_encoder_and_projector_preserve_batchnorm_state(model):
    config = asdict(model.config) | {"backend": "frozen_encoder_learned_predictor"}
    frozen = LeWMWorldModel(config)
    frozen.model.encoder.load_state_dict(model.model.encoder.state_dict())
    frozen.model.projector.load_state_dict(model.model.projector.state_dict())
    frozen.train()
    before = {k: v.clone() for k, v in frozen.model.projector.state_dict().items()}
    frozen.training_loss(torch.rand(2, 4, 3, 32, 32), torch.rand(2, 3, 7))["loss"].backward()
    assert all(p.grad is None for p in frozen.model.encoder.parameters())
    assert any(p.grad is not None for p in frozen.model.predictor.parameters())
    for k, v in before.items():
        torch.testing.assert_close(v, frozen.model.projector.state_dict()[k])


def test_state_dict_checkpoint_round_trip(model, tmp_path):
    model.eval()
    path = tmp_path / "model.pt"
    torch.save({"model_config": asdict(model.config), "model_state_dict": model.state_dict(), "global_step": 4}, path)
    loaded, checkpoint = load_checkpoint(path)
    assert checkpoint["global_step"] == 4
    rgb = torch.rand(2, 3, 32, 32)
    actions = torch.rand(2, 3, 7)
    torch.testing.assert_close(model.rollout(model.encode(rgb), actions), loaded.rollout(loaded.encode(rgb), actions))


def test_distances_and_history_limits(model):
    a = torch.tensor([[1., 0.], [0., 1.]])
    torch.testing.assert_close(model.latent_distance(a, a, "cosine"), torch.zeros(2))
    torch.testing.assert_close(model.latent_distance(a, -a, "normalized_l2"), torch.full((2,), 2.))
    with pytest.raises(ValueError, match="history_size"):
        model.predict_sequence(torch.rand(2, 4, 24), torch.rand(2, 4, 7))


def test_epoch_sampler_resume_and_split_are_deterministic():
    from latent_stl_lewm.train import EpochBatchSampler
    whole = list(EpochBatchSampler(37, 4, 3, 7))
    resumed = list(EpochBatchSampler(37, 4, 3, 7, start_batch=3))
    assert resumed == whole[3:]
    flat = [index for batch in whole for index in batch]
    assert len(flat) == len(set(flat)) == 36
    assert whole != list(EpochBatchSampler(37, 4, 3, 8))
