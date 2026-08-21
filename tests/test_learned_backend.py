import numpy as np

from amdi.energy import EnergyParameters
from amdi.graph import GraphParameters
from amdi.synthetic import add_gaussian_noise, four_region_image
from learned_amdi.backend import COARSEN, REFINE, RETAIN, HaarAMDIBackend


def backend(n=16):
    noisy=add_gaussian_noise(four_region_image(n),0.05,seed=1)
    return HaarAMDIBackend(noisy,initial_threshold=0.012,min_level=2,eparams=EnergyParameters(alpha=.01,beta=2e-4,tau=1e-7,smooth_l1=False),gparams=GraphParameters(sigma_c=.12,refinement_decay=.35),h=.6,zeta=1e-6,refine_fraction=.1,coarsen_fraction=.1)


def test_masks_always_retain_and_respect_max_level():
    b=backend()
    for r in b.candidate_records():
        assert bool(r.mask[RETAIN])
        if r.level == b.max_level:
            assert not bool(r.mask[REFINE])


def test_coarsening_requires_unanimous_sibling_votes():
    b=backend()
    records=b.candidate_records()
    eligible=[r for r in records if r.mask[COARSEN]]
    if not eligible:
        # Refine once to create an eligible sibling group.
        r=next(r for r in records if r.mask[REFINE])
        b.apply_actions({r.cell:REFINE})
        eligible=[r for r in b.candidate_records() if r.mask[COARSEN]]
    r=eligible[0]
    siblings=b._siblings(r.cell)
    before=frozenset(b.tree.refined)
    b.apply_actions({siblings[0]:COARSEN})
    assert frozenset(b.tree.refined)==before
    b.apply_actions({s:COARSEN for s in siblings})
    assert frozenset(b.tree.refined)!=before


def test_modular_control_matches_baseline_order_exactly():
    b1=backend()
    b2=backend()
    for _ in range(3):
        b1.deterministic_baseline_step()
        b2.deterministic_control_step()
    assert b1.tree.refined == b2.tree.refined
    assert np.linalg.norm(b1.current_field()-b2.current_field()) < 1e-14
    assert abs(b1.occupancy-b2.occupancy) < 1e-15
