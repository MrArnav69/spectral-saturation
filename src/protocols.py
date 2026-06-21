"""K-sweep protocols: the central experimental driver.

This module is *the* place where every per-trial sample-and-evaluate call
lives. Two public runners cover the empirical surface:

* :func:`run_binary_sweep` — for every ``(K, task, trial)`` cell, sample
  ``K`` points per class, fit a (configurable) linear classifier, and
  return the held-out accuracy and pooled-within-class effective rank.

* :func:`run_nway_sweep` — same idea extended to ``N`` classes (the
  5-way studies of §6).

* :func:`sample_and_evaluate` / :func:`sample_and_evaluate_nway` — the
  per-trial primitive used by both; exported for direct programmatic use
  (the τ-transfer and active-learning modules wrap them).

The same ``X, y`` arrays are used both for classifier fit and for the
covariance / nearest-neighbour summaries; the seed is propagated so each
trial is fully deterministic. Optional ``track_intrinsic_dim=True``
extends the per-trial summary to also compute stable rank, TwoNN, and MLE
estimators — used in the §6.5 multistat head-to-head.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.geometry import mle_intrinsic_dim, stable_rank, two_nn_intrinsic_dim
from src.saturation import effective_rank, saturation_index


# --------------------------------------------------------------------------
# Per-trial primitives
# --------------------------------------------------------------------------
def _linear_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float,
    classifier: str,
) -> float:
    """Fit the requested linear classifier and return held-out accuracy."""
    if classifier == "logistic":
        clf = LogisticRegression(max_iter=5000, C=C)
    elif classifier == "nearest_centroid":
        clf = NearestCentroid()
    elif classifier == "svm":
        svc_C = 1e6 if C == np.inf else C
        clf = LinearSVC(C=svc_C, max_iter=10000, dual="auto")
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def sample_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    test_size: int = 200,
    seed: int | None = None,
    C: float = np.inf,
    classifier: str = "logistic",
    track_intrinsic_dim: bool = False,
) -> tuple[float, dict[str, float]]:
    """Run a single trial for a binary task and return ``(accuracy, summary)``.

    ``summary`` always contains ``mean_erank`` (the pooled within-class
    effective rank). When ``track_intrinsic_dim`` is true the summary
    also contains ``stable_rank``, ``two_nn``, and ``mle`` evaluated on
    the per-trial support pool.

    Raises ``ValueError`` when ``K`` exceeds the smaller of the two class
    populations minus ``test_size``.
    """
    rng = np.random.default_rng(seed)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    if K > len(idx_0) - test_size or K > len(idx_1) - test_size:
        raise ValueError(f"K={K} too large for available data")

    train_0 = rng.choice(idx_0, size=K, replace=False)
    train_1 = rng.choice(idx_1, size=K, replace=False)

    rem_0 = np.setdiff1d(idx_0, train_0)
    rem_1 = np.setdiff1d(idx_1, train_1)

    test_0 = rng.choice(rem_0, size=test_size, replace=False)
    test_1 = rng.choice(rem_1, size=test_size, replace=False)

    X_train = np.vstack([X[train_0], X[train_1]])
    y_train = np.array([0] * K + [1] * K)
    X_test = np.vstack([X[test_0], X[test_1]])
    y_test = np.array([0] * test_size + [1] * test_size)

    X_train_mean = X_train.mean(axis=0)
    X_train_c = X_train - X_train_mean
    X_test_c = X_test - X_train_mean

    accuracy = _linear_accuracy(X_train_c, y_train, X_test_c, y_test, C, classifier)

    cov_0 = np.cov(X_train_c[:K], rowvar=False, bias=True)
    cov_1 = np.cov(X_train_c[K:], rowvar=False, bias=True)
    cov_pooled = 0.5 * (cov_0 + cov_1)

    erank_value = effective_rank(cov_pooled)
    summary: dict[str, float] = {"mean_erank": float(erank_value)}

    if track_intrinsic_dim:
        X0 = X_train_c[:K]
        X1 = X_train_c[K:]
        summary.update({
            "stable_rank": float(stable_rank(cov_pooled)),
            "two_nn": float(two_nn_intrinsic_dim(np.vstack([X0, X1]), rng=rng)),
            "mle": float(
                mle_intrinsic_dim(np.vstack([X0, X1]), k_neighbours=max(5, K // 2))
            ),
        })

    return accuracy, summary


def sample_and_evaluate_nway(
    X: np.ndarray,
    y: np.ndarray,
    classes: list[int],
    K: int,
    test_size: int = 200,
    seed: int | None = None,
    classifier: str = "logistic",
) -> tuple[float, float]:
    """Per-trial primitive for ``N``-way classification; returns ``(acc, erank)``."""
    rng = np.random.default_rng(seed)
    n_classes = len(classes)

    class_indices = {c: np.where(y == c)[0] for c in classes}

    for c in classes:
        available = len(class_indices[c])
        if K > available - test_size:
            raise ValueError(
                f"K={K} too large for class {c}: only {available} samples, "
                f"need at least {K + test_size}"
            )

    train_idx, test_idx = [], []
    for c in classes:
        idx = class_indices[c]
        chosen_train = rng.choice(idx, size=K, replace=False)
        remaining = np.setdiff1d(idx, chosen_train)
        chosen_test = rng.choice(remaining, size=test_size, replace=False)
        train_idx.append(chosen_train)
        test_idx.append(chosen_test)

    X_train = np.vstack([X[idx] for idx in train_idx])
    y_train = np.repeat(np.arange(n_classes), K)
    X_test = np.vstack([X[idx] for idx in test_idx])
    y_test = np.repeat(np.arange(n_classes), test_size)

    X_train_mean = X_train.mean(axis=0)
    X_train_c = X_train - X_train_mean
    X_test_c = X_test - X_train_mean

    if classifier == "logistic":
        clf = LogisticRegression(max_iter=5000, C=np.inf)
    elif classifier == "nearest_centroid":
        clf = NearestCentroid()
    elif classifier == "svm":
        clf = LinearSVC(C=1e6, max_iter=10000, dual="auto")
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    clf.fit(X_train_c, y_train)
    accuracy = clf.score(X_test_c, y_test)

    per_class_covs = []
    for i in range(n_classes):
        X_c = X_train_c[i * K : (i + 1) * K]
        per_class_covs.append(np.cov(X_c, rowvar=False, bias=True))
    cov_pooled = np.mean(per_class_covs, axis=0)

    return accuracy, effective_rank(cov_pooled)


# --------------------------------------------------------------------------
# Per-task K sweeps
# --------------------------------------------------------------------------
def run_binary_sweep(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    class_a: int,
    class_b: int,
    Ks: list[int],
    test_size: int,
    n_trials: int = 50,
    pca_dims: int | None = 50,
    C: float = np.inf,
    classifier: str = "logistic",
    track_intrinsic_dim: bool = False,
) -> list[dict]:
    """Run the binary K-sweep on a single (class_a, class_b) pair.

    Returns a list of result rows, one per successful ``K`` value, with
    keys ``K``, ``mean_acc``, ``std_acc``, ``mean_erank``, ``S``,
    ``marginal``. When ``track_intrinsic_dim`` is true the per-row dict
    additionally carries ``mean_stable_rank``, ``mean_two_nn``, ``mean_mle``,
    and the corresponding ``S_x`` saturation indices sampled at the
    smaller endpoint of each doubling pair.
    """
    mask = (y == class_a) | (y == class_b)
    X_pair = X[mask].astype(np.float64)
    y_pair = np.where(y[mask] == class_a, 0, 1)

    scaler = StandardScaler()
    X_pair = scaler.fit_transform(X_pair)
    if pca_dims is not None:
        X_pair = PCA(n_components=pca_dims).fit_transform(X_pair)

    results: list[dict] = []
    print(f"\n{'=' * 60}")
    print(f"{name} | {class_a} vs {class_b} | {n_trials} trials")
    print(f"{'=' * 60}")

    for K in Ks:
        n0 = (y_pair == 0).sum()
        n1 = (y_pair == 1).sum()
        if K > min(n0, n1) - test_size:
            print(f"  K={K:4d}: SKIPPED (insufficient data: {n0}/{n1} available)")
            continue

        accs: list[float] = []
        eranks: list[float] = []
        stable_ranks: list[float] = []
        two_nns: list[float] = []
        mles: list[float] = []

        for trial in range(n_trials):
            accuracy, summary = sample_and_evaluate(
                X_pair, y_pair, K=K, test_size=test_size, seed=trial, C=C,
                classifier=classifier, track_intrinsic_dim=track_intrinsic_dim,
            )
            accs.append(accuracy)
            eranks.append(summary["mean_erank"])
            if track_intrinsic_dim:
                stable_ranks.append(summary["stable_rank"])
                two_nns.append(summary["two_nn"])
                mles.append(summary["mle"])

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_erank = float(np.mean(eranks))
        S = saturation_index(mean_erank, K)
        marginal = mean_acc - results[-1]["mean_acc"] if results else 0.0

        row: dict = {
            "K": int(K),
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_erank": mean_erank,
            "S": S,
            "marginal": marginal,
        }
        if track_intrinsic_dim:
            mean_srank = float(np.mean(stable_ranks))
            mean_2nn = float(np.mean(two_nns))
            mean_mle = float(np.mean(mles))
            row.update({
                "mean_stable_rank": mean_srank,
                "mean_two_nn": mean_2nn,
                "mean_mle": mean_mle,
                "S_stable_rank": mean_srank / (2 * K),
                "S_two_nn": mean_2nn / (2 * K),
                "S_mle": mean_mle / (2 * K),
            })
        results.append(row)
        print(
            f"  K={K:4d}: acc={mean_acc:.4f}±{std_acc:.4f}, "
            f"erank={mean_erank:.2f}, S={S:.4f}, marginal={marginal:+.4f}"
        )

    return results


def run_nway_sweep(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    classes: list[int],
    Ks: list[int],
    test_size: int = 200,
    n_trials: int = 50,
    pca_dims: int | None = 50,
    classifier: str = "logistic",
) -> list[dict]:
    """Run the N-way K-sweep on the supplied list of ``classes``.

    Returns a list of result rows, one per successful ``K`` value, with
    keys ``K``, ``mean_acc``, ``std_acc``, ``mean_erank``, ``S``,
    ``marginal``, ``spearman_rho``, ``spearman_pval``. The Spearman pair
    is computed across the doubling-pair marginals ``ΔA(K) = A(2K) - A(K)``
    against ``S(K)`` at the smaller ``K``.
    """
    mask = np.isin(y, classes)
    X_sub = X[mask].astype(np.float64)
    y_sub = y[mask]
    y_remapped = np.empty_like(y_sub)
    for new_label, original in enumerate(classes):
        y_remapped[y_sub == original] = new_label
    classes_remapped = list(range(len(classes)))

    scaler = StandardScaler()
    X_sub = scaler.fit_transform(X_sub)
    if pca_dims is not None:
        X_sub = PCA(n_components=pca_dims).fit_transform(X_sub)

    results: list[dict] = []
    n_classes = len(classes)
    print(f"\n{'=' * 60}")
    print(f"{name} | {n_classes}-way: {classes} | {n_trials} trials")
    print(f"{'=' * 60}")

    for K in Ks:
        per_class_counts = [(y_remapped == c).sum() for c in classes_remapped]
        min_available = min(per_class_counts)
        if K > min_available - test_size:
            print(
                f"  K={K:4d}: SKIPPED "
                f"(min class has {min_available} samples, need {K + test_size})"
            )
            continue

        accs: list[float] = []
        eranks: list[float] = []
        for trial in range(n_trials):
            try:
                accuracy, erank = sample_and_evaluate_nway(
                    X_sub, y_remapped, classes_remapped,
                    K=K, test_size=test_size, seed=trial, classifier=classifier,
                )
                accs.append(accuracy)
                eranks.append(erank)
            except ValueError as exc:
                print(f"    trial {trial} skipped: {exc}")
                continue

        if not accs:
            continue

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_erank = float(np.mean(eranks))
        S = saturation_index(mean_erank, K)
        marginal = mean_acc - results[-1]["mean_acc"] if results else 0.0

        results.append({
            "K": int(K),
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_erank": mean_erank,
            "S": S,
            "marginal": float(marginal),
        })
        print(
            f"  K={K:4d}: acc={mean_acc:.4f}±{std_acc:.4f}, "
            f"erank={mean_erank:.2f}, S={S:.4f}, marginal={marginal:+.4f}"
        )

    if len(results) >= 2:
        S_vals = [r["S"] for r in results[1:]]
        marginals = [r["marginal"] for r in results[1:]]
        rho, pval = stats.spearmanr(S_vals, marginals)
        print(f"  Within-task Spearman ρ = {rho:.4f}  (p={pval:.4f})")
        for r in results:
            r["spearman_rho"] = float(rho)
            r["spearman_pval"] = float(pval)

    return results


# --------------------------------------------------------------------------
# JSON result I/O
# --------------------------------------------------------------------------
def save_results(results_dict: dict, filepath: str | Path) -> None:
    with open(filepath, "w") as f:
        json.dump(results_dict, f, indent=4)


def load_results(filepath: str | Path) -> dict | None:
    if Path(filepath).exists():
        with open(filepath) as f:
            return json.load(f)
    return None
