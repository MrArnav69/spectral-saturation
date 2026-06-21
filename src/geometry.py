"""Intrinsic-dimension alternative estimators.

Hosts the alternative summary statistics used in §6.5 of the manuscript to
demonstrate that ``S(K)`` is stable under summary-statistic choice:

* :func:`effective_rank`    — entropy-based effective rank with strict handling
                              of negative eigenvalues (numerically distinct
                              from :func:`src.saturation.effective_rank` for
                              rank-deficient covariances; the §6.5 head-to-head
                              uses the entropy form here for apples-to-apples
                              comparability with TwoNN / MLE / stable rank)
* :func:`stable_rank`       — participation ratio / numerical rank
* :func:`two_nn_intrinsic_dim` — Facco et al. (2017) Sci. Reports
* :func:`mle_intrinsic_dim` — Levina–Bickel (2004) NeurIPS
* :func:`trial_summary_stats` — per-trial aggregator used by
                                :mod:`src.protocols`

Every routine here is deterministic given a fixed random number generator.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def effective_rank(matrix: np.ndarray) -> float:
    """Entropy effective rank with negative-eigen handling.

    Returns ``erank(A) = exp(-Σ p_i log p_i)`` with ``p_i = λ_i / Σ λ_j``
    after projecting any negative or near-zero eigenvalues to zero.
    Numerically equivalent to :func:`src.saturation.effective_rank` on
    positive-definite inputs but robust to rank-deficient covariances
    where numerical noise drives some eigenvalues below zero.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    eigenvalues = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
    eigenvalues = np.where(eigenvalues < 1e-12, 0.0, eigenvalues)
    trace = eigenvalues.sum()
    if trace <= 0:
        return 0.0
    p = eigenvalues / trace
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def stable_rank(matrix: np.ndarray) -> float:
    """Participation-ratio / numerical rank.

    Defined as ``||A||_F^2 / ||A||_op^2 = Σ λ_i^2 / λ_max^2``. Returns
    ``0.0`` for the rank-deficient case where the operator norm is zero.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    eigenvalues = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
    if eigenvalues.max() <= 0:
        return 0.0
    return float((eigenvalues ** 2).sum() / (eigenvalues.max() ** 2))


def two_nn_intrinsic_dim(
    points: np.ndarray,
    rng: np.random.Generator | None = None,
    subsample: int | None = None,
    seed: int | None = None,
) -> float:
    """Two-nearest-neighbor intrinsic dimension (Facco et al. 2017).

    For each point i, take the ratio ``r2_i = d_2 / d_1`` of the second to
    the first nearest neighbor distance. The Facco closed-form estimator
    returns ``d̂ = N / Σ log(r2_i / r2_min)``; for numerical stability we
    sort ``r2`` and use the first (smallest) value rather than the minimum
    literal. We clip pathological estimates to ``d`` (the ambient dimension)
    when ``K`` is small or samples nearly co-linear.

    Subsampling at ``min(2K, 200)`` is recommended for large supports to
    keep the KD-tree query economical.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if subsample is not None and subsample < points.shape[0]:
        indices = rng.choice(points.shape[0], size=subsample, replace=False)
        points = points[indices]
    n, d = points.shape
    if n < 4:
        return float(d)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=3)
    d1 = dists[:, 1]
    d2 = dists[:, 2]
    valid = (d1 > 1e-12) & (d2 > 1e-12)
    if valid.sum() < 3:
        return float(d)
    r2 = d2[valid] / d1[valid]
    r2_sorted = np.sort(r2)
    n_valid = len(r2_sorted)
    d_hat = n_valid / np.sum(np.log(r2_sorted[r2_sorted > 0] / r2_sorted[0] + 1e-12))
    if not np.isfinite(d_hat) or d_hat <= 0:
        return float(d)
    if d_hat > 2 * d:
        d_hat = float(d)
    return float(d_hat)


def mle_intrinsic_dim(
    points: np.ndarray,
    k_neighbours: int | None = None,
) -> float:
    """Levina–Bickel MLE intrinsic-dimension estimator.

    Uses the standard maximum-likelihood form

        d̂ = (k - 1) / Σ_{j < k} log(r_k / r_j)

    averaged across all points. Defaults to ``k = max(5, K // 2)`` to remain
    stable without hyperparameter tuning; results are clipped to ``d`` when
    numerical instability produces a non-finite or zero estimate.
    """
    n, d = points.shape
    if k_neighbours is None:
        k_neighbours = max(5, n // 2)
    k_neighbours = min(k_neighbours, n - 1)
    if k_neighbours < 2:
        return float(d)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k_neighbours + 1)
    dists = dists[:, 1:]
    dists = np.where(dists < 1e-12, 1e-12, dists)
    log_ratios = np.log(dists[:, -1:] / dists[:, :-1])
    per_point = (k_neighbours - 1) / log_ratios.sum(axis=1)
    d_hat = float(np.mean(per_point))
    if not np.isfinite(d_hat) or d_hat <= 0:
        return float(d)
    return min(d_hat, float(d))


def trial_summary_stats(
    points_0: np.ndarray,
    points_1: np.ndarray,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compute the four summary statistics on a single support split.

    Returns ``{erank, stable_rank, two_nn, mle}`` on the pooled (2K, d)
    per-trial support. The per-statistic saturation indices ``S_x`` follow
    from ``x / (2K)``; the per-statistic ``S`` arrays are derived by
    :mod:`src.protocols`.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pooled = np.vstack([points_0, points_1])
    cov_0 = np.cov(points_0, rowvar=False, bias=True)
    cov_1 = np.cov(points_1, rowvar=False, bias=False)
    cov_pooled = 0.5 * (cov_0 + cov_1)

    erank = effective_rank(cov_pooled)
    srank = stable_rank(cov_pooled)
    K, _ = points_0.shape
    sub = min(2 * K, 200) if 2 * K > 200 else None
    d_two_nn = two_nn_intrinsic_dim(pooled, rng=rng, subsample=sub)
    d_mle = mle_intrinsic_dim(pooled, k_neighbours=max(5, K // 2))

    return {
        "erank": float(erank),
        "stable_rank": float(srank),
        "two_nn": float(d_two_nn),
        "mle": float(d_mle),
    }
