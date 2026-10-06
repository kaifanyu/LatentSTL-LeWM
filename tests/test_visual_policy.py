import torch

from latent_stl_lewm.visual_policy import VisualDemonstrationPolicy, ResidualDynamics


def test_initial_visual_selection_and_fixed_clock_action_alignment():
    artifact={"trajectory_ids":["a","b"],"actions":torch.tensor([[[.1,0,-1],[.2,0,-1]],[[.3,0,-1],[.4,0,-1]]]),
              "frame_episode":torch.tensor([0,0,0,1,1,1]),"frame_index":torch.tensor([0,1,2,0,1,2]),
              "latents":torch.tensor([[1.,0],[.7,.7],[0,1],[0.,1],[.7,.7],[1,0]])}
    policy=VisualDemonstrationPolicy(artifact,{"a","b"},phase_mode="clock")
    choice=policy.reset(torch.tensor([1.,0]))
    assert choice["train_trajectory"]=="a"
    actions,meta=policy.propose(torch.tensor([.7,.7]),1,3)
    assert torch.equal(actions[:,0],torch.tensor([.2,0,0]))
    assert torch.equal(actions[:,-1],torch.tensor([-1.,-1.,-1.]))
    assert meta["train_phase"]==1


def test_residual_dynamics_adds_to_nominal_once():
    class Model:
        def rollout(self,z,actions):return actions
    nominal=torch.tensor([[.5,-1.],[.4,-1.]])
    model=ResidualDynamics(Model(),nominal)
    result=model.rollout(torch.ones(2,1),torch.tensor([[[.01,0],[0,0]],[[-.01,0],[0,0]]]))
    assert torch.allclose(result[:,0,0],torch.tensor([.51,.49]))
    assert torch.equal(result[:,:,-1],torch.full((2,2),-1.))
