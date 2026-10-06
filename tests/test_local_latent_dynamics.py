from types import SimpleNamespace
import torch
from latent_stl_lewm.local_latent_dynamics import LocalLatentDynamics


class EncoderStub(torch.nn.Module):
    config = SimpleNamespace(history_size=3)


def test_local_fit_recovers_action_effect_and_zero_command():
    torch.manual_seed(2)
    torch.set_num_threads(2)
    z=torch.ones(64,3)
    actions=torch.randn(64,7)
    gain=torch.randn(7,3)*.05
    artifact={'z':z,'delta':actions@gain,'velocity':torch.zeros_like(z),
              'actions':actions,'basis':torch.eye(3),'z_scale':torch.ones(3),
              'v_scale':torch.ones(3),'action_scale':torch.ones(7)}
    model=LocalLatentDynamics(EncoderStub(),artifact,neighbors=64,ridge=1e-4,
                              latent_dims=3,velocity_dims=3,delta_clip=0.,device='cpu')
    query=torch.randn(4,7)*.2
    predicted=model.predict_next(torch.ones(4,3),torch.zeros(4,3),query)
    torch.testing.assert_close(predicted,torch.ones(4,3)+query@gain,atol=2e-5,rtol=1e-4)
    hold=model.predict_next(torch.ones(1,3),torch.zeros(1,3),torch.zeros(1,7))
    torch.testing.assert_close(hold,torch.ones(1,3),atol=2e-5,rtol=0)


def test_history_is_visual_only_and_rollout_preserves_start():
    z=torch.randn(12,3)
    artifact={'z':z,'delta':torch.zeros_like(z),'velocity':torch.zeros_like(z),
              'actions':torch.zeros(12,7),'basis':torch.eye(3),'z_scale':torch.ones(3),
              'v_scale':torch.ones(3),'action_scale':torch.ones(7)}
    model=LocalLatentDynamics(EncoderStub(),artifact,neighbors=12,latent_dims=3,velocity_dims=3,device='cpu')
    model.set_history(z[:3],torch.zeros(2,7))
    predicted=model.rollout(z[2:3],torch.zeros(1,4,7))
    torch.testing.assert_close(predicted,z[2:3,None].expand(-1,5,-1))
