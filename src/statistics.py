"""Statistical analyses for the spectral-saturation paper.

This module is the single source of truth for every ρ, p-value, K_sat, CIs,
and multi-statistic comparison that appears in ``tab:pertask``,
``tab:clip_pertask``, ``fig:decoupling``, and the appendices of the
manuscript. All routines operate on the K-sweep result rows produced by
:mod:`src.protocols` and are deterministic given the cached JSON files
under ``results/``.

Methodology (matches the protocol paragraph, lines 813-826 of
``index.tex``):

    ΔA(K) = A(2K) - A(K)                  [doubling-pair marginal]
    S(K)  = erank(K) / K                  [saturation index]
    ρ     = Spearman(S(K), ΔA(K)) per task
    ρ_pool = Spearman(all S, all ΔA) over all doubling pairs

Reference points used elsewhere in the paper:

* All 17 binary tasks use the dense K-grid
  ``{2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096}``.
* 14 CLIP binary tasks share the same K-grid.
* 5 N-way tasks use the 15-element grid
  ``{2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024}``.
* Pooled observation count: 246 (17 binary × ~15 doubling steps).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

results_dir = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# Doubling-pair extractor (canonical convention)
# ---------------------------------------------------------------------------
def doubling_pairs(results: list[dict]) -> tuple[list[float], list[float]]:
    """Return ``(S_pair, ΔA_pair)`` for a single task, restricted to doubles.

    Convention (canonical; reproduces every ρ, p, n, and CI in the
    manuscript and ``tab:pertask`` / ``tab:phases``):

        For each valid doubling pair ``(K, 2K)`` on the grid, indexed by
        ``K`` (the smaller of the pair):

            ``S(K)  = erank(K) / K``   → saturation index *before* doubling
            ``ΔA(K) = A(2K) - A(K)``   → marginal gain upon doubling to 2K

    The ``S(K)`` is sampled at the *smaller* endpoint of the pair. This
    is the convention actually used to produce the headline numbers in
    the manuscript: pooled Spearman ρ = 0.548 (p = 1.1×10⁻²⁰, N = 246),
    within-task median ρ = 0.811 (16/17 tasks positive), and the
    30/123/93 phase counts in ``tab:phases``.
    """
    Ks = [r["K"] for r in results]
    acc = [r["mean_acc"] for r in results]
    S = [r["mean_erank"] / r["K"] for r in results]
    S_pair, M_pair = [], []
    for i, K in enumerate(Ks):
        if K % 2 == 0 and K > 2 and (K // 2) in Ks:
            j = Ks.index(K // 2)
            S_pair.append(S[j])
            M_pair.append(acc[i] - acc[j])
    return S_pair, M_pair


# ---------------------------------------------------------------------------
# Multi-statistic doubling-pair extractor (extends the canonical form)
# ---------------------------------------------------------------------------
def doubling_pairs_multistat(
    results: list[dict],
    stat: str = "S",
    k_size: int = 2,
) -> tuple[list[float], list[float]]:
    """Return ``(S_x, ΔA)`` for ``stat`` ∈ {S, S_stable_rank, S_two_nn, S_mle}.

    Convention: ``S_x`` is sampled at the *smaller* K of the doubling pair
    (same as :func:`doubling_pairs`).
    """
    Ks = [r["K"] for r in results]
    acc = [r["mean_acc"] for r in results]
    if stat == "S":
        S = [r["mean_erank"] / r["K"] for r in results]
    elif stat == "S_stable_rank":
        S = [r["mean_stable_rank"] / (2 * r["K"]) for r in results]
    elif stat == "S_two_nn":
        S = [r["mean_two_nn"] / (2 * r["K"]) for r in results]
    elif stat == "S_mle":
        S = [r["mean_mle"] / (2 * r["K"]) for r in results]
    else:
        raise ValueError(f"Unknown stat: {stat}")

    S_pair, M_pair = [], []
    for i, K in enumerate(Ks):
        if K % k_size == 0 and K > k_size and (K // k_size) in Ks:
            j = Ks.index(K // k_size)
            S_pair.append(S[j])
            M_pair.append(acc[i] - acc[j])
    return S_pair, M_pair


# ---------------------------------------------------------------------------
# Per-task summaries
# ---------------------------------------------------------------------------
def per_task_spearman(results: list[dict]) -> tuple[float, float]:
    """Within-task Spearman correlation between ``S(K)`` and ``ΔA(K)``.

    Returns ``(rho, pvalue)`` using the doubling-pair convention the
    paper protocol commits to (``index.tex`` line 826).
    """
    S_p, M_p = doubling_pairs(results)
    if len(S_p) < 3:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(S_p, M_p)
    return float(rho), float(p)


def per_task_spearman_multistat(results: list[dict], stat: str = "S") -> tuple[float, float]:
    """Within-task Spearman between stat-saturated index and ``ΔA(K)``."""
    S_p, M_p = doubling_pairs_multistat(results, stat=stat)
    if len(S_p) < 3:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(S_p, M_p)
    return float(rho), float(p)


def first_k_below(results: list[dict], threshold: float) -> float | str:
    """Smallest K with ``S(K) ≤ threshold``, or ``"---"`` if never crosses.

    The paper's ``tab:pertask`` caption documents the threshold as 0.3,
    but the actual reported ``K_sat`` values correspond to threshold =
    0.02 (the "deep-saturation" boundary introduced in §4). Use
    ``threshold=0.02`` for paper reproduction; ``threshold=0.3`` for the
    standard phase crossing.
    """
    for r in results:
        if (r["mean_erank"] / r["K"]) <= threshold:
            return r["K"]
    return "---"


def per_task_table(
    results: list[dict], task_name: str, dataset_label: str | None = None
) -> dict[str, Any]:
    """One row of ``tab:pertask`` — ``erank_inf``, ``peak_acc``, ``K_sat``, ``ρ``, ``p``.

    Defaults to the paper-reported deep-saturation threshold K_sat = 0.02.
    """
    erank_inf = float(results[-1]["mean_erank"])
    peak_acc = float(max(r["mean_acc"] for r in results))
    K_sat = first_k_below(results, 0.02)
    rho, p = per_task_spearman(results)
    return {
        "task": task_name,
        "dataset": dataset_label,
        "erank_inf": erank_inf,
        "peak_acc": peak_acc,
        "K_sat": K_sat,
        "rho": rho,
        "p": p,
    }


# ---------------------------------------------------------------------------
# Aggregations over the population of tasks
# ---------------------------------------------------------------------------
def pooled_spearman(all_results: dict[str, list[dict]]) -> tuple[float, float, int]:
    """Pooled Spearman ρ across every binary doubling pair in ``all_results``.

    Counts every observation once (each K contributes one ``ΔA(K)`` if
    ``2K`` is also on the grid), so ``N ≈ 17 tasks × 15 ≈ 246``.
    """
    S_all, M_all = [], []
    for results in all_results.values():
        S_p, M_p = doubling_pairs(results)
        S_all += S_p
        M_all += M_p
    rho, p = stats.spearmanr(S_all, M_all)
    return float(rho), float(p), len(S_all)


def pooled_spearman_multistat(
    all_results: dict, stat: str = "S"
) -> tuple[float, float, int]:
    """Pooled Spearman ρ using any summary statistic key."""
    S_all, M_all = [], []
    for r in all_results.values():
        s, m = doubling_pairs_multistat(r, stat=stat)
        S_all += s
        M_all += m
    rho, p = stats.spearmanr(S_all, M_all)
    return float(rho), float(p), len(S_all)


def pooled_excluding_cifar(
    all_results: dict[str, list[dict]],
) -> tuple[float, float, int]:
    """Pooled Spearman ρ restricted to non-CIFAR tasks (§4.1)."""
    S_all, M_all = [], []
    for name, results in all_results.items():
        if "CIFAR" in name:
            continue
        S_p, M_p = doubling_pairs(results)
        S_all += S_p
        M_all += M_p
    rho, p = stats.spearmanr(S_all, M_all)
    return float(rho), float(p), len(S_all)


def positive_per_task_count(
    all_results: dict[str, list[dict]],
) -> tuple[int, int, list[float]]:
    """``(positive_count, total, sorted_per_task_rhos)``."""
    rhos = [per_task_spearman(r)[0] for r in all_results.values()]
    pos = int(sum(1 for r in rhos if r > 0))
    return pos, len(rhos), sorted(rhos)


def median_per_task_rho(all_results: dict[str, list[dict]]) -> float:
    """Median of the per-task Spearman ρ across tasks."""
    return float(np.median([per_task_spearman(r)[0] for r in all_results.values()]))


# ---------------------------------------------------------------------------
# Decoupling (erank_∞ vs peak acc) — both Spearman and Pearson
# ---------------------------------------------------------------------------
def decoupling_corr(all_results: dict[str, list[dict]]) -> dict[str, float]:
    """Spearman and Pearson for the decoupling plot."""
    erank_inf = [r[-1]["mean_erank"] for r in all_results.values()]
    peaks = [max(x["mean_acc"] for x in r) for r in all_results.values()]

    rho_s, p_s = stats.spearmanr(erank_inf, peaks)
    r_p, p_p = stats.pearsonr(erank_inf, peaks)
    return {
        "spearman_rho": float(rho_s),
        "spearman_p": float(p_s),
        "pearson_r": float(r_p),
        "pearson_p": float(p_p),
        "N": len(erank_inf),
    }


def decoupling_ci(erank_inf: list[float], peaks: list[float]) -> dict[str, Any]:
    """Fisher-z 95% CI on the decoupling Spearman correlation."""
    rho, p = stats.spearmanr(erank_inf, peaks)
    n = len(erank_inf)
    dof = n - 3
    se = 1.0 / np.sqrt(max(dof, 1))
    z = np.arctanh(np.clip(rho, -0.9999, 0.9999))
    lo = np.tanh(z - 1.96 * se)
    hi = np.tanh(z + 1.96 * se)
    return {
        "point_rho": float(rho),
        "p": float(p),
        "n": int(n),
        "dof": int(dof),
        "fisher_z_lo": float(lo),
        "fisher_z_hi": float(hi),
    }


# ---------------------------------------------------------------------------
# Cluster bootstrap on pooled ρ (respects task-level nesting)
# ---------------------------------------------------------------------------
def cluster_bootstrap_pooled_rho(
    all_results: dict,
    stat: str = "S",
    B: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster-bootstrap 95% CI for pooled Spearman ρ over tasks.

    Resamples *tasks* with replacement (cluster = task), then within
    each sampled task refits Spearman on its doubling pairs. Returns the
    point estimate, the percentile CI, the bootstrap median, and the
    bootstrap SE.
    """
    rng = np.random.default_rng(seed)
    task_names = list(all_results.keys())
    indices = np.arange(len(task_names))

    def pooled_rho_on_subset(idx_subset):
        S_all, M_all = [], []
        for i in idx_subset:
            s, m = doubling_pairs_multistat(all_results[task_names[i]], stat=stat)
            S_all += s
            M_all += m
        if len(S_all) < 3:
            return float("nan")
        rho, _ = stats.spearmanr(S_all, M_all)
        return float(rho)

    boot = np.empty(B)
    for b in range(B):
        boot[b] = pooled_rho_on_subset(
            rng.choice(indices, size=len(indices), replace=True)
        )

    point = pooled_rho_on_subset(indices)
    return {
        "point": float(point),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
        "median": float(np.median(boot)),
        "se": float(np.std(boot)),
    }


# ---------------------------------------------------------------------------
# Binary stopping-rule AUC for any dichotomy scorer
# ---------------------------------------------------------------------------
def binary_stopping_auc(
    all_results: dict,
    stat: str = "S",
    threshold_geo_acc: float = 0.005,
    signed: bool = False,
) -> dict[str, float]:
    """Compute stopping-rule AUC for ``stat`` like in the paper.

    A "stop" event is the ``(K, task)`` pair where ``ΔA(K) < threshold_geo_acc``
    on the next doubling. The scorer is ``stat`` at the smaller K of the
    pair. AUC is computed over every observation; positive class =
    ``ΔA < threshold`` ("saturated").

    By default the scorer sign is *negated* so that higher scorer ==
    more likely saturated (matching paper convention: AUC > 0.5 = good).
    Pass ``signed=True`` to get the raw score's discrimination.
    """
    scorers, labels = [], []
    for r in all_results.values():
        S_p, M_p = doubling_pairs_multistat(r, stat=stat)
        if signed:
            scorers.extend(S_p)
        else:
            scorers.extend([-s for s in S_p])
        labels.extend([int(m < threshold_geo_acc) for m in M_p])
    if not scorers:
        return {"auc": float("nan"), "n_pos": 0, "n_neg": 0}

    a = np.array(scorers)
    y = np.array(labels)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, a))
    pos_scores = a[y == 1]
    neg_scores = a[y == 0]
    V10 = _placement_value(pos_scores, a)
    V01 = _placement_value(neg_scores, a)
    n1, n0 = n_pos, n_neg
    s01 = np.var(V01, ddof=1) / n0 + np.var(V10, ddof=1) / n1
    se_auc = float(np.sqrt(max(s01, 1e-12)))
    return {
        "auc": auc,
        "se_auc": se_auc,
        "auc_lo": auc - 1.96 * se_auc,
        "auc_hi": auc + 1.96 * se_auc,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "stat": stat,
    }


def _placement_value(scores: np.ndarray, all_scores: np.ndarray) -> np.ndarray:
    """Per-observation placement values for DeLong variance estimation."""
    n = len(all_scores)
    return np.array(
        [np.mean(all_scores < s) + 0.5 * np.mean(all_scores == s) for s in scores]
    )


# ---------------------------------------------------------------------------
# Multi-statistic table compile
# ---------------------------------------------------------------------------
def multistat_summary_table(all_results: dict) -> pd.DataFrame:
    """Per-task ρ table comparing {erank, stable_rank, two_nn, mle}.

    Also includes the asymptotic value of each statistic at the largest K.
    """
    rows = []
    stat_keys = ["S", "S_stable_rank", "S_two_nn", "S_mle"]
    for task, r in all_results.items():
        row = {"task": task}
        row["erank_inf"] = r[-1].get("mean_erank", float("nan"))
        row["stable_rank_inf"] = r[-1].get("mean_stable_rank", float("nan"))
        row["two_nn_inf"] = r[-1].get("mean_two_nn", float("nan"))
        row["mle_inf"] = r[-1].get("mean_mle", float("nan"))
        for stat in stat_keys:
            rho, _ = per_task_spearman_multistat(r, stat=stat)
            row[f"rho_{stat}"] = rho
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------
def load_json(name: str) -> dict[str, list[dict]]:
    with open(results_dir / name) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI — sanity print
# ---------------------------------------------------------------------------
def _print_summary() -> None:
    all_results = load_json("all_results.json")
    print("=" * 78)
    print("Per-task ρ from results/all_results.json (binary, PCA)")
    print("=" * 78)
    rows = [per_task_table(r, name) for name, r in all_results.items()]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    pos, total, rhos = positive_per_task_count(all_results)
    rho_pool, p_pool, N_pool = pooled_spearman(all_results)
    rho_nc, p_nc, N_nc = pooled_excluding_cifar(all_results)
    dec = decoupling_corr(all_results)
    print()
    print(f"Within-task median ρ      = {np.median(rhos):.3f}")
    print(f"Within-task positive rate = {pos}/{total}")
    print(f"Pooled Spearman ρ         = {rho_pool:.3f}  (p = {p_pool:.2e}, N = {N_pool})")
    print(f"  excluding CIFAR:        = {rho_nc:.3f}  (p = {p_nc:.2e}, N = {N_nc})")
    print()
    print(
        f"Decoupling Spearman ρ_s   = {dec['spearman_rho']:.3f}  (p = {dec['spearman_p']:.3f})"
    )
    print(f"Decoupling Pearson r      = {dec['pearson_r']:.3f}  (p = {dec['pearson_p']:.3f})")


def main() -> None:
    _print_summary()


if __name__ == "__main__":
    main()
