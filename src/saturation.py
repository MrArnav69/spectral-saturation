"""Effective rank and the saturation index S(K).

The headline quantity of the paper. ``erank`` is the exponential Shannon
entropy of the normalized eigenvalue spectrum of a covariance matrix; the
saturation index is the per-sample ratio

    S(K) = erank(Σ̂_W^(K)) / K,

where ``Σ̂_W^(K)`` is the pooled within-class sample covariance on a support
set of size ``K`` per class. ``S(K) → 0`` indicates that the pool geometry
has saturated relative to the classifier's degrees of freedom.

This module houses the *production* formula used by every cached result in
:mod:`results`. A separate, numerically stricter variant appears in
:mod:`src.geometry` as ``effective_rank``; the two are equivalent on
positive-definite covariances but diverge on rank-deficient ones (the
``geometry`` version drops negative eigenvalues to zero).
"""

from __future__ import annotations

import numpy as np


def effective_rank(covariance: np.ndarray) -> float:
    """Production effective-rank definition used in all K-sweep results.

    Computes ``erank(Σ) = exp(-Σ p_i log p_i)`` with ``p_i = λ_i / tr(λ)``,
    after clipping eigenvalues to ``1e-12``. Matches the cached values in
    ``results/all_results.json`` and downstream ``paper/`` table builders
    exactly.

    Parameters
    ----------
    covariance : (d, d) ndarray
        Symmetric (preferably positive semi-definite) covariance matrix.
    """
    eigvals = np.linalg.eigvalsh(covariance)
    eigvals = np.maximum(eigvals, 1e-12)
    normalization = eigvals.sum()
    p = eigvals / normalization
    return float(np.exp(-np.sum(p * np.log(p))))


def saturation_index(erank_value: float, k: int) -> float:
    """Return the saturation index ``S(K) = erank / K``.

    Returns ``0.0`` for any non-positive ``K`` to preserve a safe boundary
    for plotting and aggregation.
    """
    if k <= 0:
        return 0.0
    return float(erank_value / k)
