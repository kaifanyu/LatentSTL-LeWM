import torch
import pytest
from types import SimpleNamespace
from latent_stl_lewm.waypoint_cem import WaypointGuide, WaypointPredicates
from latent_stl_lewm.reference_predicates import ReferencePredicates


def artifact():
    path=torch.tensor([[1.,0.],[.8,.2],[.5,.5],[.2,.8],[0.,1.]])
    return {"latents":path,"frame_episode":torch.zeros(5,dtype=torch.int64),
            "frame_index":torch.arange(5),"actions":torch.ones(1,4,7),"trajectory_ids":["train_example"]}


def test_visual_tracking_advances_monotonically_and_pads_end_with_hold():
    guide=WaypointGuide(artifact(),lookahead=2)
    guide.start(torch.tensor([1.,0.]))
    assert guide.update(torch.tensor([.5,.5]),2)["progress_frame"]==2
    assert guide.update(torch.tensor([1.,0.]),3)["progress_frame"]==2
    assert guide.target_index==4
    seq=guide.action_sequence(4)
    assert torch.equal(seq[:2],torch.ones(2,7))
    assert torch.equal(seq[2:,:6],torch.zeros(2,6))
    assert torch.equal(seq[2:,6],-torch.ones(2))


def test_waypoint_surrogate_leaves_exact_original_goal_stl_untouched():
    predicates=ReferencePredicates(torch.tensor([[1.,0.],[0.,1.]]),torch.tensor([[-1.,0.]]),.1,.1)
    latent=torch.tensor([[[1.,0.],[.6,.4]]])
    final_goal=torch.tensor([1.,0.])
    waypoint=torch.tensor([0.,1.])
    old=predicates.score(latent,final_goal)
    new=WaypointPredicates(predicates,waypoint).score(latent,final_goal)
    assert torch.equal(old["stl"],new["stl"])
    assert torch.equal(old["terminal"],new["terminal"])
    assert old["goal_progress"]<0<new["goal_progress"]


def test_hybrid_alignment_uses_clock_only_as_explicit_regularizer():
    guide=WaypointGuide(artifact(),lookahead=1,tracking="hybrid",clock_weight=.1)
    guide.start(torch.tensor([1.,0.]))
    info=guide.update(torch.tensor([.5,.5]),3)
    assert info["progress_frame"] in (2,3)
    assert info["tracking"]=="hybrid"


@pytest.mark.parametrize("steps,execute,horizon", [(0, 2, 8), (10, 0, 8), (10, 9, 8)])
def test_runner_rejects_invalid_budgets_before_creating_outputs(steps, execute, horizon):
    from latent_stl_lewm.waypoint_cem import run
    with pytest.raises(ValueError, match="Require steps"):
        run(SimpleNamespace(steps=steps, execute_steps=execute, horizon=horizon))
