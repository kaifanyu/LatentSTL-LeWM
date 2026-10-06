import numpy as np

from latent_stl_lewm.temporal_calibration import temporal_thresholds


def test_temporal_calibration_separates_whole_traces_despite_safe_frame_outlier():
    safe=[np.array([.8,.8,-.1]),np.array([.8,.8,-.2]),np.array([.8,.8,-.8]),np.array([.8,.8,-.7])]
    reach=[np.array([-.9,-.1]),np.array([-.9,-.2]),np.array([-.9,-.8]),np.array([-.9,-.7])]
    safe_margin,goal_threshold=temporal_thresholds(safe,reach,[True,True,False,False],
                                                  [True,True,False,False],.5)
    assert -.7<safe_margin<=-.2
    assert .2<=goal_threshold<.7
    assert all((trace.min()>=safe_margin)==label for trace,label in zip(safe,[True,True,False,False]))
    assert all((trace.max()+goal_threshold>=0)==label for trace,label in zip(reach,[True,True,False,False]))
