from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .backend import CandidateRecord
from .config import FeatureScales


def _gradient_norm(patch: np.ndarray) -> float:
    patch = np.asarray(patch, dtype=float)
    if min(patch.shape) < 2:
        return 0.0
    gx, gy = np.gradient(patch)
    return float(np.sqrt(np.sum(gx * gx + gy * gy)))


def _patch_with_halo(field: np.ndarray, support: tuple[int, int, int, int]) -> np.ndarray:
    """Return the local support plus a one-pixel halo.

    A Haar leaf is piecewise constant inside its own cell, so gradients and
    variances evaluated strictly inside the leaf can be identically zero.
    The halo keeps the feature tied to the current reconstructed field u^n
    while exposing variation across neighboring adaptive cells.
    """
    r0, r1, c0, c1 = support
    nr, nc = field.shape
    return np.asarray(
        field[max(0, r0-1):min(nr, r1+1), max(0, c0-1):min(nc, c1+1)],
        dtype=float,
    )


def raw_feature_rows(
    field: np.ndarray,
    records: Sequence[CandidateRecord],
    active_leaf_cells: set,
    max_level: int,
    step_index: int,
    n_steps: int,
) -> np.ndarray:
    rows = []
    for rec in records:
        patch = _patch_with_halo(field, rec.support)
        q = float(np.linalg.norm(rec.coeff_block))
        g = _gradient_norm(patch)
        v = float(np.var(patch))
        lev = float(rec.level) / float(max(max_level, 1))
        neigh = rec.neighborhood_cells
        rho = (
            sum(c in active_leaf_cells for c in neigh) / float(len(neigh))
            if neigh
            else 1.0
        )
        tau = float(step_index) / float(max(n_steps, 1))
        rows.append([q, g, v, lev, rho, tau])
    return np.asarray(rows, dtype=float)


def build_feature_matrix(
    field: np.ndarray,
    records: Sequence[CandidateRecord],
    active_leaf_cells: set,
    max_level: int,
    step_index: int,
    n_steps: int,
    scales: FeatureScales,
) -> np.ndarray:
    raw = raw_feature_rows(
        field, records, active_leaf_cells, max_level, step_index, n_steps
    )
    if raw.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    out = raw.copy()
    out[:, 0] /= scales.s_q + scales.epsilon_feat
    out[:, 1] /= scales.s_g + scales.epsilon_feat
    out[:, 2] /= scales.s_v + scales.epsilon_feat
    return out.astype(np.float32)


def estimate_scales(raw_rows: np.ndarray, quantile: float = 0.95) -> FeatureScales:
    raw_rows = np.asarray(raw_rows, dtype=float)
    if raw_rows.ndim != 2 or raw_rows.shape[1] != 6:
        raise ValueError("raw_rows must have shape [N,6]")
    if len(raw_rows) == 0:
        raise ValueError("cannot estimate scales from empty data")
    vals = np.quantile(np.abs(raw_rows[:, :3]), quantile, axis=0)
    vals = np.maximum(vals, 1.0e-12)
    return FeatureScales(float(vals[0]), float(vals[1]), float(vals[2]), 1.0e-12)


def normalized_error(adaptive: np.ndarray, reference: np.ndarray, eps: float) -> float:
    adaptive = np.asarray(adaptive, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return float(np.linalg.norm(adaptive - reference) / (np.linalg.norm(reference) + eps))


def switching_measure(before: set, after: set, n_full: int) -> float:
    return len(before.symmetric_difference(after)) / float(n_full)
