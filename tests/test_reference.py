from amdi.energy import EnergyParameters
from amdi.graph import GraphParameters
from amdi.synthetic import add_gaussian_noise, four_region_image
from learned_amdi.reference import uniform_reference_trajectory


def test_uniform_reference_has_full_basis_and_correct_length():
    n=16
    noisy=add_gaussian_noise(four_region_image(n),0.05,seed=2)
    traj,h=uniform_reference_trajectory(noisy,eparams=EnergyParameters(alpha=.01,beta=2e-4,tau=1e-7,smooth_l1=False),gparams=GraphParameters(sigma_c=.12,refinement_decay=.35),h=.6,n_steps=2)
    assert len(traj)==3
    assert all(row['basis_size']==n*n for row in h)
