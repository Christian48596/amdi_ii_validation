import numpy as np
from amdi.energy import EnergyParameters
from amdi.graph import GraphParameters
from amdi.haar import adaptive_tree_from_detail_threshold, full_coefficients
from amdi.solver import adaptive_frozen_denoise
from amdi.synthetic import add_gaussian_noise, four_region_image
from learned_amdi.backend import HaarAMDIBackend

def test_wrapper_matches_baseline_one_step():
    n=16; noisy=add_gaussian_noise(four_region_image(n),0.05,seed=5); full=full_coefficients(noisy,max_level=4)
    ep=EnergyParameters(alpha=.01,beta=2e-4,tau=1e-7,smooth_l1=False); gp=GraphParameters(sigma_c=.12,refinement_decay=.35)
    initial=adaptive_tree_from_detail_threshold(full,2,4,threshold=.012,min_level=2)
    image,tree,coeffs,_=adaptive_frozen_denoise(noisy,initial,full,ep,gp,h=.6,outer_iterations=1,adapt=True,zeta=1e-6,refine_fraction=.1,coarsen_fraction=.1)
    b=HaarAMDIBackend(noisy,initial_threshold=.012,min_level=2,eparams=ep,gparams=gp,h=.6,zeta=1e-6,refine_fraction=.1,coarsen_fraction=.1); b.deterministic_baseline_step()
    assert tree.refined==b.tree.refined
    assert np.linalg.norm(image-b.current_field())<1e-12
