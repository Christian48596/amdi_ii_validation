"""Uniformly resolved reference trajectories using the deterministic AMDI propagator."""

from __future__ import annotations

import numpy as np

from amdi.energy import EnergyParameters, total_energy
from amdi.graph import GraphParameters
from amdi.haar import AdaptiveHaarTree, coefficients_vector, full_coefficients, reconstruct
from amdi.solver import frozen_weight_fista, safeguard_candidate


def uniform_reference_trajectory(
    noisy: np.ndarray,
    *,
    eparams: EnergyParameters,
    gparams: GraphParameters,
    h: float,
    n_steps: int,
) -> tuple[list[np.ndarray], list[dict]]:
    """Compute the no-adaptation full-Haar trajectory used in the RL reward."""
    noisy = np.asarray(noisy, dtype=float)
    n = int(noisy.shape[0])
    max_level = int(np.log2(n))
    tree = AdaptiveHaarTree.uniform(dim=2, level=max_level)
    full = full_coefficients(noisy, max_level=max_level)
    coeffs = {idx: full.get(idx, 0.0) for idx in tree.basis_indices()}
    trajectory = [reconstruct(coeffs, tree, noisy.shape)]
    e0, _ = total_energy(coefficients_vector(coeffs, tree), tree, full, eparams, gparams)
    history = [{"step": 0, "energy": float(e0), "basis_size": tree.basis_size()}]
    for step in range(1, n_steps + 1):
        proposal = frozen_weight_fista(tree, coeffs, full, h, eparams, gparams)
        coeffs, energy, accepted, sg = safeguard_candidate(
            tree, coeffs, proposal, full, eparams, gparams, return_diagnostics=True
        )
        trajectory.append(reconstruct(coeffs, tree, noisy.shape))
        history.append(
            {
                "step": step,
                "energy": float(energy),
                "basis_size": tree.basis_size(),
                "safeguard_accepted": bool(accepted),
                "safeguard_factor": float(sg["factor"]),
                "safeguard_backtracks": int(sg["backtracks"]),
            }
        )
    return trajectory, history
