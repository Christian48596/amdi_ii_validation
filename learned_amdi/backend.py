"""Learned AMDI backend using the exact deterministic AMDI adaptive Haar coefficient algebra.

The deterministic AMDI numerical core in :mod:`amdi` is copied verbatim from the uploaded
``amdi_validation_v0.4`` archive.  This module adds only the decision interface
required by the learned policy.

The learned and deterministic pathways use the same deterministic AMDI outer-iteration
ordering: frozen-weight propagation/safeguard first, followed by adaptive tree
selection.  the modular deterministic control replaces only the deterministic selector with a learned
selector; the propagation step and its position in the iteration are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np

from amdi.energy import EnergyParameters, total_energy
from amdi.graph import GraphParameters
from amdi.haar import (
    AdaptiveHaarTree,
    BasisIndex,
    Cell,
    adaptive_tree_from_detail_threshold,
    coefficients_vector,
    full_coefficients,
    reconstruct,
    transfer_coefficients,
)
from amdi.solver import (
    candidate_trees_from_data,
    choose_tree_variationally,
    frozen_weight_fista,
    safeguard_candidate,
)

COARSEN = 0
RETAIN = 1
REFINE = 2
ACTION_NAMES = ("coarsen", "retain", "refine")


@dataclass(frozen=True)
class CandidateRecord:
    cell: Cell
    support: tuple[int, int, int, int]
    level: int
    coeff_block: np.ndarray
    neighborhood_cells: tuple[Cell, ...]
    mask: np.ndarray


@dataclass
class StepDiagnostics:
    energy: float
    safeguard_accepted: bool
    safeguard_factor: float
    safeguard_backtracks: int
    proposed_coarsen: int = 0
    proposed_retain: int = 0
    proposed_refine: int = 0
    executed_coarsen_groups: int = 0
    executed_refine: int = 0


class HaarAMDIBackend:
    """Stateful adapter around the deterministic AMDI Haar AMDI numerical core."""

    def __init__(
        self,
        noisy: np.ndarray,
        *,
        initial_threshold: float,
        min_level: int,
        eparams: EnergyParameters,
        gparams: GraphParameters,
        h: float,
        zeta: float,
        refine_fraction: float,
        coarsen_fraction: float,
    ) -> None:
        noisy = np.asarray(noisy, dtype=float)
        if noisy.ndim != 2 or noisy.shape[0] != noisy.shape[1]:
            raise ValueError("Learned AMDI backend expects a square 2D image")
        n = int(noisy.shape[0])
        if n <= 0 or (n & (n - 1)):
            raise ValueError("image size must be a positive power of two")
        self.noisy = noisy.copy()
        self.shape = noisy.shape
        self.max_level = int(np.log2(n))
        self.min_level = int(min_level)
        self.initial_threshold = float(initial_threshold)
        self.eparams = eparams
        self.gparams = gparams
        self.h = float(h)
        self.zeta = float(zeta)
        self.refine_fraction = float(refine_fraction)
        self.coarsen_fraction = float(coarsen_fraction)
        self.full_data = full_coefficients(self.noisy, max_level=self.max_level)
        self.reset()

    @property
    def n_full(self) -> int:
        return int(np.prod(self.shape))

    @property
    def n_active(self) -> int:
        return int(self.tree.basis_size())

    @property
    def occupancy(self) -> float:
        return self.n_active / float(self.n_full)

    def reset(self) -> None:
        self.tree = adaptive_tree_from_detail_threshold(
            self.full_data,
            dim=2,
            max_level=self.max_level,
            threshold=self.initial_threshold,
            min_level=self.min_level,
        )
        self.coeffs = {idx: self.full_data.get(idx, 0.0) for idx in self.tree.basis_indices()}
        self.step_index = 0
        self.last_diagnostics: StepDiagnostics | None = None

    def clone(self) -> "HaarAMDIBackend":
        other = HaarAMDIBackend(
            self.noisy,
            initial_threshold=self.initial_threshold,
            min_level=self.min_level,
            eparams=self.eparams,
            gparams=self.gparams,
            h=self.h,
            zeta=self.zeta,
            refine_fraction=self.refine_fraction,
            coarsen_fraction=self.coarsen_fraction,
        )
        other.tree = self.tree.copy()
        other.coeffs = dict(self.coeffs)
        other.step_index = int(self.step_index)
        other.last_diagnostics = self.last_diagnostics
        return other

    def current_field(self) -> np.ndarray:
        return reconstruct(self.coeffs, self.tree, self.shape)

    def active_dof_indices(self) -> set[BasisIndex]:
        return set(self.tree.basis_indices())

    def active_leaf_cells(self) -> set[Cell]:
        return set(self.tree.leaves())

    def _bounds(self, cell: Cell) -> tuple[int, int, int, int]:
        level, tr = cell
        width = self.shape[0] // (2**level)
        r0 = int(tr[0] * width)
        c0 = int(tr[1] * width)
        return r0, r0 + width, c0, c0 + width

    def _siblings(self, cell: Cell) -> tuple[Cell, ...]:
        level, tr = cell
        if level == 0:
            return (cell,)
        parent_tr = tuple(v // 2 for v in tr)
        return tuple(
            (level, tuple(2 * parent_tr[d] + bits[d] for d in range(2)))
            for bits in product((0, 1), repeat=2)
        )

    def _parent(self, cell: Cell) -> Cell | None:
        level, tr = cell
        if level == 0:
            return None
        return level - 1, tuple(v // 2 for v in tr)

    def _neighborhood(self, cell: Cell) -> tuple[Cell, ...]:
        level, tr = cell
        out: set[Cell] = {cell}
        out.update(self._siblings(cell))
        parent = self._parent(cell)
        if parent is not None:
            out.add(parent)
        for d in range(2):
            for step in (-1, 1):
                kt = list(tr)
                kt[d] += step
                if 0 <= kt[d] < 2**level:
                    out.add((level, tuple(kt)))
        return tuple(sorted(out))

    def _prospective_detail_block(self, cell: Cell) -> np.ndarray:
        level, tr = cell
        if level >= self.max_level:
            return np.zeros(3, dtype=float)
        return np.asarray(
            [
                self.full_data.get(BasisIndex("wavelet", level, tr, o), 0.0)
                for o in range(1, 4)
            ],
            dtype=float,
        )

    def candidate_records(self) -> list[CandidateRecord]:
        leaves = set(self.tree.leaves())
        records: list[CandidateRecord] = []
        for cell in sorted(leaves):
            level, _ = cell
            siblings = self._siblings(cell)
            coarsen_ok = level > self.min_level and all(s in leaves for s in siblings)
            refine_ok = level < self.max_level
            records.append(
                CandidateRecord(
                    cell=cell,
                    support=self._bounds(cell),
                    level=level,
                    coeff_block=self._prospective_detail_block(cell),
                    neighborhood_cells=self._neighborhood(cell),
                    mask=np.asarray([coarsen_ok, True, refine_ok], dtype=bool),
                )
            )
        return records

    def apply_actions(self, actions: dict[Cell, int]) -> dict[str, int]:
        """Apply local proposals with deterministic global hierarchy resolution.

        Coarsening requires unanimous ``coarsen`` votes from all four leaf
        siblings.  Refinement is then applied to surviving leaves.  This is the
        concrete implementation of the manuscript's constrained update
        ``U_MR``.
        """
        old_tree = self.tree.copy()
        leaves = set(old_tree.leaves())
        counts = {
            "proposed_coarsen": sum(int(a == COARSEN) for a in actions.values()),
            "proposed_retain": sum(int(a == RETAIN) for a in actions.values()),
            "proposed_refine": sum(int(a == REFINE) for a in actions.values()),
            "executed_coarsen_groups": 0,
            "executed_refine": 0,
        }

        new_tree = old_tree.copy()
        processed_parents: set[Cell] = set()
        for cell in sorted(leaves):
            parent = old_tree.parent(cell)
            if parent is None or parent in processed_parents:
                continue
            siblings = tuple(old_tree.children(parent))
            if not all(s in leaves for s in siblings):
                continue
            if parent[0] < self.min_level:
                continue
            if all(actions.get(s, RETAIN) == COARSEN for s in siblings):
                new_tree.coarsen(parent)
                counts["executed_coarsen_groups"] += 1
                processed_parents.add(parent)

        # Refinement proposals are honored only if the cell survived coarsening.
        for cell in sorted(old_tree.leaves()):
            if actions.get(cell, RETAIN) != REFINE:
                continue
            if cell in new_tree.leaves() and cell[0] < self.max_level:
                before = frozenset(new_tree.refined)
                new_tree.refine(cell)
                if frozenset(new_tree.refined) != before:
                    counts["executed_refine"] += 1

        self.coeffs = transfer_coefficients(self.coeffs, self.tree, new_tree)
        self.tree = new_tree
        return counts

    def _propagate_on_current_tree(self) -> StepDiagnostics:
        candidate = frozen_weight_fista(
            self.tree,
            self.coeffs,
            self.full_data,
            self.h,
            self.eparams,
            self.gparams,
        )
        accepted_coeffs, energy, accepted, sg = safeguard_candidate(
            self.tree,
            self.coeffs,
            candidate,
            self.full_data,
            self.eparams,
            self.gparams,
            return_diagnostics=True,
        )
        self.coeffs = accepted_coeffs
        self.step_index += 1
        diag = StepDiagnostics(
            energy=float(energy),
            safeguard_accepted=bool(accepted),
            safeguard_factor=float(sg["factor"]),
            safeguard_backtracks=int(sg["backtracks"]),
        )
        self.last_diagnostics = diag
        return diag

    def propagate_for_adaptation(self) -> StepDiagnostics:
        """Advance the field on the current tree before any adaptive decision.

        This is the first half of the deterministic AMDI outer iteration.  The tree is
        unchanged, while the coefficients are updated by the frozen-weight
        propagator and energy safeguard.
        """
        return self._propagate_on_current_tree()

    def learned_adapt(
        self,
        actions: dict[Cell, int],
        propagation_diag: StepDiagnostics,
    ) -> StepDiagnostics:
        """Apply learned tree actions after propagation, as in deterministic AMDI."""
        counts = self.apply_actions(actions)
        propagation_diag.energy = float(self.energy())
        for key, value in counts.items():
            setattr(propagation_diag, key, int(value))
        self.last_diagnostics = propagation_diag
        return propagation_diag

    def deterministic_adapt(
        self,
        propagation_diag: StepDiagnostics,
    ) -> StepDiagnostics:
        """Apply the original deterministic AMDI deterministic selector after propagation."""
        candidates = candidate_trees_from_data(
            self.tree,
            self.coeffs,
            self.full_data,
            refine_fraction=self.refine_fraction,
            coarsen_fraction=self.coarsen_fraction,
        )
        selected = choose_tree_variationally(
            self.tree,
            self.coeffs,
            candidates,
            self.full_data,
            self.eparams,
            self.gparams,
            self.zeta,
        )
        self.tree = selected.tree
        self.coeffs = selected.coeffs
        propagation_diag.energy = float(selected.energy)
        self.last_diagnostics = propagation_diag
        return propagation_diag

    def deterministic_control_step(self) -> StepDiagnostics:
        """Modular deterministic AMDI control: propagate first, then deterministic adapt."""
        diag = self.propagate_for_adaptation()
        return self.deterministic_adapt(diag)

    def deterministic_baseline_step(self) -> StepDiagnostics:
        """Exact propagate-then-adapt ordering used in deterministic AMDI validation."""
        # First propagate and safeguard on the current tree.
        candidate = frozen_weight_fista(
            self.tree,
            self.coeffs,
            self.full_data,
            self.h,
            self.eparams,
            self.gparams,
        )
        accepted_coeffs, _, accepted, sg = safeguard_candidate(
            self.tree,
            self.coeffs,
            candidate,
            self.full_data,
            self.eparams,
            self.gparams,
            return_diagnostics=True,
        )
        # Then choose the next tree, exactly as adaptive_frozen_denoise.
        candidates = candidate_trees_from_data(
            self.tree,
            accepted_coeffs,
            self.full_data,
            refine_fraction=self.refine_fraction,
            coarsen_fraction=self.coarsen_fraction,
        )
        selected = choose_tree_variationally(
            self.tree,
            accepted_coeffs,
            candidates,
            self.full_data,
            self.eparams,
            self.gparams,
            self.zeta,
        )
        self.tree = selected.tree
        self.coeffs = selected.coeffs
        self.step_index += 1
        diag = StepDiagnostics(
            energy=float(selected.energy),
            safeguard_accepted=bool(accepted),
            safeguard_factor=float(sg["factor"]),
            safeguard_backtracks=int(sg["backtracks"]),
        )
        self.last_diagnostics = diag
        return diag

    def energy(self) -> float:
        value, _ = total_energy(
            coefficients_vector(self.coeffs, self.tree),
            self.tree,
            self.full_data,
            self.eparams,
            self.gparams,
        )
        return float(value)
