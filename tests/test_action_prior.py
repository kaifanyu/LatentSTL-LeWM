import torch

from latent_stl_lewm.action_prior import VisualActionPrior


def test_visual_prior_distinct_train_episodes_and_end_hold():
    artifact={'latents':torch.tensor([[1.,0.],[1.,.01],[0.,1.]]),
              'frame_episode':torch.tensor([0,0,1]), 'frame_index':torch.tensor([1,0,0]),
              'actions':torch.tensor([[[.2,0,0,0,0,0,-1]],[[.4,0,0,0,0,0,-1]]]),
              'trajectory_ids':['training_a','training_b']}
    prior=VisualActionPrior(artifact)
    sequence, meta=prior.retrieve(torch.tensor([[1.,0.]]),horizon=3,count=2)
    assert sequence.shape==(2,3,7)
    assert [x['trajectory'] for x in meta]==['training_a','training_b']
    assert all(x['split']=='train' for x in meta)
    assert torch.equal(sequence[0,:,:6],torch.zeros(3,6))
    assert torch.equal(sequence[:,:,-1],torch.full((2,3),-1.))
    assert sequence[1,0,0]==.4
    assert sequence[1,1:,0].sum()==0
