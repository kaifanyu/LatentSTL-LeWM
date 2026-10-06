import torch
from torch.nn import functional as F
from latent_stl_lewm.relative_waypoint_cem import RelativeWaypointGuide


def test_initial_offset_maps_start_exactly_and_targets_use_same_offset():
    path=torch.tensor([[1.,0.,0.],[.8,.6,0.],[0.,1.,0.]])
    artifact={"latents":path,"frame_episode":torch.zeros(3,dtype=torch.int64),
              "frame_index":torch.arange(3),"actions":torch.zeros(1,2,7),"trajectory_ids":["train"]}
    guide=RelativeWaypointGuide(artifact,lookahead=1)
    initial=torch.tensor([1.,0.,.3])
    guide.start(initial)
    result=guide.update(initial,0)
    assert result["matched_frame"]==0
    assert abs(result["visual_distance"])<1e-6
    assert torch.allclose(guide.target,F.normalize(path[1],dim=0)+guide.initial_offset)
