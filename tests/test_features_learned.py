import numpy as np
from learned_amdi.features import estimate_scales

def test_scale_estimation_positive():
    x=np.array([[1,2,3,0,0,0],[2,4,6,0,0,0]],float)
    s=estimate_scales(x)
    assert s.s_q>0 and s.s_g>0 and s.s_v>0
