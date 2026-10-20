"""Contamination: the treatment effect bleeding into neighbouring units.

A share of each treated unit's expected incremental conversions is moved to
its nearest units (option A: the total effect is conserved, it only lands
somewhere else). Spillover into control units contaminates the
counterfactual of every control-based estimator; spillover into other
treated units stays within the treatment group.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from geo_simulator.config import ContaminationConfig

EARTH_RADIUS_KM = 6371.0


def _unit_vectors(lat, lon) -> np.ndarray:
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def neighbor_matrix(lat, lon, config: ContaminationConfig) -> sparse.csr_matrix:
    """Row-stochastic neighbourhood matrix ``W`` (``n x n``, zero diagonal).

    Row ``i`` holds the share of unit ``i``'s bleed received by each of its
    ``n_neighbors`` nearest units, with weights ``exp(-d / bandwidth_km)``
    normalised to sum to 1. Rows of units without any neighbour within
    ``max_distance_km`` are all zero.
    """
    n = len(lat)
    if n < 2:
        return sparse.csr_matrix((n, n))
    xyz = _unit_vectors(lat, lon)
    k = min(config.n_neighbors, n - 1)
    chord, idx = cKDTree(xyz).query(xyz, k=k + 1)
    chord, idx = chord[:, 1:], idx[:, 1:]  # drop self
    km = 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0, 1))
    weight = np.exp(-km / config.bandwidth_km)
    if config.max_distance_km is not None:
        weight[km > config.max_distance_km] = 0.0
    total = weight.sum(1, keepdims=True)
    weight = np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)
    rows = np.repeat(np.arange(n), k)
    return sparse.csr_matrix((weight.ravel(), (rows, idx.ravel())), shape=(n, n))


def redistribute(effect: np.ndarray, neighbors: sparse.csr_matrix, share: float) -> tuple[np.ndarray, np.ndarray]:
    """Move ``share`` of each unit's expected effect to its neighbours.

    Args:
        effect: Expected incremental conversions created by the campaign,
            ``(n_units, n_periods)``; non-zero only for treated units.
        neighbors: Row-stochastic matrix from ``neighbor_matrix``.
        share: Share of the effect that bleeds out.

    Returns:
        ``(direct, spill_in)``: the part of each unit's own effect that stays
        home, and the spillover each unit receives. ``direct + spill_in``
        sums to ``effect`` in total (the effect is conserved).
    """
    if share == 0 or not effect.any():
        return effect.copy(), np.zeros_like(effect)
    has_neighbors = np.asarray(neighbors.sum(axis=1)).ravel() > 0
    leaving = effect * (share * has_neighbors)[:, None]
    direct = effect - leaving
    spill_in = np.asarray(neighbors.T @ leaving)
    return direct, spill_in
