import torch
from latent_stl_lewm.goal_inverse_dynamics import GoalInverseDynamics, make_pairs


def test_inverse_pairs_are_aligned_split_isolated_and_translation_only():
    z=torch.arange(5,dtype=torch.float32)[:,None].expand(-1,192)
    actions=torch.zeros(4,7);actions[:,-1]=-1;actions[:,:3]=torch.arange(4)[:,None]
    cache={'episodes':{'train0':{'split':'train','z':z,'a':actions},
                       'val0':{'split':'val','z':z+100,'a':actions},
                       'test_unreadable':{'split':'test'}}}
    pairs=make_pairs(cache,'train')
    assert pairs['trajectory_ids']==['train0']
    assert len(pairs['current'])==8  # 4 one-step + 3 two-step + 1 four-step
    torch.testing.assert_close(pairs['target'][:,0]-pairs['current'][:,0],pairs['horizon'])
    expected=pairs['current'][:,0]+(pairs['horizon']-1)/2
    torch.testing.assert_close(pairs['actions'][:,0],expected)
    assert pairs['current'].max()<100


def test_inverse_checkpoint_round_trip_and_goal_conditioning(tmp_path):
    torch.manual_seed(1)
    model=GoalInverseDynamics(latent_dim=6,hidden_dim=16).eval()
    current=torch.randn(4,6);target=torch.randn(4,6)
    prediction=model(current,target,4.)
    assert prediction.shape==(4,3)
    assert not torch.equal(prediction,model(current,current,4.))
    path=tmp_path/'inverse.pt'
    torch.save({'model_config':model.config,'model_state_dict':model.state_dict()},path)
    loaded,_=GoalInverseDynamics.load(path)
    torch.testing.assert_close(prediction,loaded(current,target,4.))
