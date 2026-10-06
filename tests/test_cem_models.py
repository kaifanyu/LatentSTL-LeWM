from types import SimpleNamespace

import pytest
import torch

from latent_stl_lewm.cem_models import CEMModel, cube_actions
from latent_stl_lewm.logic import PredicateConfig
from latent_stl_lewm.planner import CEMConfig, plan_cem


class NativeIntegrator(torch.nn.Module):
    def __init__(self, repeat=1):
        super().__init__()
        self.config = SimpleNamespace(history_size=3)
        self.repeat = repeat
        self.contexts = []

    def predict_sequence(self, z, a):
        self.contexts.append((z.clone(), a.clone()))
        delta = a[..., :1] if self.repeat == 1 else a[..., [0,5,10,15,20]].sum(-1,keepdim=True)
        return z + delta


def test_native_timing_and_observed_history_alignment():
    controls = torch.zeros(1, 2, 7)
    controls[..., 0] = .1
    controls[..., 6] = -1
    observed = torch.arange(11).float()[:,None]
    previous = torch.zeros(10,7); previous[:,0] = torch.arange(10)
    previous[:,6] = -1
    for kind, repeat in [('local',1),('official_cube_transfer',5)]:
        native = NativeIntegrator(repeat)
        adapter = CEMModel(native,kind=kind,action_repeat=5)
        adapter.set_history(observed, previous if repeat==1 else previous/10)
        z = adapter.rollout(observed[-1:],controls)
        assert z.shape[1] == 1 + 10//repeat
        assert torch.allclose(z[:,-1], torch.tensor([[11.]]),atol=1e-5)
        ctx, acts = native.contexts[0]
        expected = [8.,9.,10.] if repeat==1 else [0.,5.,10.]
        assert ctx[0,:,0].tolist() == expected
        if repeat==1:
            assert torch.allclose(acts[0,:,0],torch.tensor([8.,9.,.1]))
        else:
            assert torch.allclose(acts[0,0,[0,5,10,15,20]],torch.arange(5)/10)
            assert torch.allclose(acts[0,1,[0,5,10,15,20]],torch.arange(5,10)/10)


def test_cube_conversion_rejects_unsupported_actions_and_scales_yaw():
    a = torch.tensor([[.2,-.3,.4,0.,0.,.3,-1.]])
    assert torch.allclose(cube_actions(a),torch.tensor([[.2,-.3,.4,.5,-1.]]))
    a[0,3] = .1
    with pytest.raises(ValueError,match='roll/pitch'):
        cube_actions(a)
    a[0,3] = 0; a[0,5] = 1
    with pytest.raises(ValueError,match='bounds'):
        cube_actions(a)


class Integrator:
    def rollout(self,z0,actions):
        return torch.cat([z0[:,None],z0[:,None]+actions.cumsum(1)],dim=1)


def test_cem_warm_start_is_evaluated_and_does_not_replace_sampling():
    mean = torch.tensor([[.4],[.6]])
    run = plan_cem(Integrator(),torch.zeros(1,1),torch.ones(1),torch.zeros(1),[-1.],[1.],
                   objective='terminal',config=CEMConfig(horizon=2,population=32,elites=4,iterations=2),
                   predicate_config=PredicateConfig(metric='l2'),initial_mean=mean)
    candidates = torch.tensor(run['iterations'][0]['candidate_actions'])
    assert torch.equal(candidates[0],mean)
    assert torch.unique(candidates.flatten(1),dim=0).shape[0] > 20
    assert run['score'] == 0
    assert run['candidate_evaluations'] == 64
    with pytest.raises(ValueError,match='Warm-start'):
        plan_cem(Integrator(),torch.zeros(1,1),torch.ones(1),torch.zeros(1),[-1.],[1.],
                 config=CEMConfig(horizon=2),initial_mean=torch.zeros(1,1))
