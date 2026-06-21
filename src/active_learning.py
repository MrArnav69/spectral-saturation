"""Active learning vs. saturation-gated stopping (§6.7).

Compares three strategies for annotation budget allocation in binary
few-shot classification:

1. **random** — random sampling (baseline)
2. **uncertainty** — HACohen-style margin-based uncertainty sampling
3. **joint** — uncertainty sampling gated by the saturation index ``S(K)``

For each task, the K-grid is walked in order; the *joint* strategy
halts as soon as ``S(K) ≤ τ`` for the first time and propagates the
final accuracy to every larger K.

Public functions
----------------

* :func:`compute_saturation_index` — per K, on the current support pool.
* :func:`run_trial` — single seed → curve over the ``target_Ks`` grid.
* :func:`run_task_experiment` — aggregates trials for one task.
* :func:`print_summary` — formatted ASCII summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.saturation import effective_rank

# Baseline constants used across the §6.7 surface.
default_tau: float = 0.3
default_k_grid: list[int] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
default_n_trials: int = 50
default_pca_dim: int = 50
default_test_size: int = 200

results_dir = Path(__file__).resolve().parent.parent / "results"


def compute_saturation_index(
    X_support: np.ndarray, y_support: np.ndarray, K: int
) -> float:
    """Return ``S(K) = erank(Σ_W^(K)) / K`` for the current support pool."""
    classes = np.unique(y_support)
    cov_W = np.zeros((X_support.shape[1], X_support.shape[1]))
    for c in classes:
        X_c = X_support[y_support == c]
        if len(X_c) > 1:
            cov_c = np.cov(X_c, rowvar=False, bias=True)
            cov_W += cov_c
    cov_W /= len(classes)
    cov_W += 1e-6 * np.eye(cov_W.shape[0])
    return float(effective_rank(cov_W) / K)


def run_trial(
    X_pair: np.ndarray,
    y_pair: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_Ks: list[int],
    strategy: str,
    tau: float = default_tau,
    seed: int = 0,
) -> dict[int, dict[str, Any]]:
    """Run one trial of a single sampling strategy.

    For fair comparison across strategies, the pool is pre-shuffled with
    ``rng.permutation`` so every strategy starts from the same initial
    ordering; random sampling then takes the first ``target_K`` per class,
    while uncertainty-based strategies maintain a separate growth list.
    """
    rng = np.random.default_rng(seed)
    n_pool = len(y_pair)

    perm = rng.permutation(n_pool)
    X_pool = X_pair[perm]
    y_pool = y_pair[perm]

    results: dict[int, dict[str, Any]] = {
        K: {"acc": None, "S": None, "stopped": False, "samples_used": None}
        for K in target_Ks
    }

    support_class_0: list[int] = []
    support_class_1: list[int] = []
    stopped_early = False
    stop_K: int | None = None

    for target_K in target_Ks:
        if stopped_early and stop_K is not None:
            results[target_K] = results[stop_K].copy()
            results[target_K]["stopped"] = True
            continue

        if strategy == "random":
            class_0_idxs = [i for i in range(n_pool) if y_pool[i] == 0]
            class_1_idxs = [i for i in range(n_pool) if y_pool[i] == 1]
            support_class_0 = class_0_idxs[:target_K]
            support_class_1 = class_1_idxs[:target_K]
        elif strategy in ("uncertainty", "joint"):
            if len(support_class_0) < 2:
                avail_0 = [
                    i for i in range(n_pool)
                    if y_pool[i] == 0 and i not in support_class_0
                ]
                if avail_0:
                    support_class_0.extend(avail_0[: min(2, len(avail_0))])
            if len(support_class_1) < 2:
                avail_1 = [
                    i for i in range(n_pool)
                    if y_pool[i] == 1 and i not in support_class_1
                ]
                if avail_1:
                    support_class_1.extend(avail_1[: min(2, len(avail_1))])

            needed = target_K - len(support_class_0)
            if needed > 0 and len(support_class_0) >= 2 and len(support_class_1) >= 2:
                support_indices = support_class_0 + support_class_1
                X_supp = X_pool[support_indices]
                y_supp = y_pool[support_indices]
                clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
                clf.fit(X_supp, y_supp)

                remaining = [i for i in range(n_pool) if i not in support_indices]
                if remaining:
                    X_rem = X_pool[remaining]
                    margins = np.abs(clf.decision_function(X_rem))

                    for c, support_list in (
                        (0, support_class_0),
                        (1, support_class_1),
                    ):
                        c_rem = [
                            remaining[i]
                            for i in range(len(remaining))
                            if y_pool[remaining[i]] == c
                        ]
                        if not c_rem:
                            continue
                        c_margins = margins[
                            [remaining.index(i) for i in c_rem]
                        ]
                        top_uncertain = np.argsort(c_margins)[
                            : min(needed, len(c_rem))
                        ]
                        selected = [c_rem[i] for i in top_uncertain]
                        if c == 0:
                            support_class_0.extend(selected)
                        else:
                            support_class_1.extend(selected)

        support_class_0 = support_class_0[:target_K]
        support_class_1 = support_class_1[:target_K]
        support_indices = support_class_0 + support_class_1

        if not support_class_0 or not support_class_1:
            results[target_K] = {
                "acc": 0.5, "S": 1.0, "stopped": False,
                "samples_used": len(support_indices),
            }
            continue

        X_supp = X_pool[support_indices]
        y_supp = y_pool[support_indices]
        S_K = compute_saturation_index(X_supp, y_supp, target_K)

        if strategy == "joint" and S_K <= tau:
            stopped_early = True
            stop_K = target_K

        X_supp_mean = X_supp.mean(axis=0)
        X_supp_centered = X_supp - X_supp_mean
        X_test_centered = X_test - X_supp_mean

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf.fit(X_supp_centered, y_supp)
        acc = clf.score(X_test_centered, y_test)

        results[target_K] = {
            "acc": float(acc),
            "S": float(S_K),
            "stopped": stopped_early,
            "samples_used": len(support_indices),
        }

    return results


def run_task_experiment(
    task_name: str,
    X: np.ndarray,
    y: np.ndarray,
    class_a: int,
    class_b: int,
    n_trials: int = default_n_trials,
    k_grid: list[int] = default_k_grid,
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, list[float]]]]]:
    """Run the full §6.7 experiment for one binary task."""
    print(f"\n[{task_name}] {class_a} vs {class_b}")

    mask = (y == class_a) | (y == class_b)
    X_bin = X[mask].astype(np.float64)
    y_bin = (y[mask] == class_b).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bin)
    pca = PCA(n_components=min(default_pca_dim, X_scaled.shape[1], len(y_bin) // 4))
    X_pca = pca.fit_transform(X_scaled)

    rng = np.random.default_rng(42)
    n_total = len(y_bin)
    test_idx = rng.choice(
        n_total, size=min(default_test_size, n_total // 4), replace=False
    )
    train_mask = ~np.isin(np.arange(n_total), test_idx)

    X_train, y_train = X_pca[train_mask], y_bin[train_mask]
    X_test, y_test = X_pca[~train_mask], y_bin[~train_mask]

    print(f"  Train: {len(y_train)}, Test: {len(y_test)}, PCA: {X_pca.shape[1]}D")

    strategies = ["random", "uncertainty", "joint"]
    all_results: dict[str, dict[int, dict[str, list[float]]]] = {
        s: {K: {"accs": [], "S_vals": [], "stop_flags": []} for K in k_grid}
        for s in strategies
    }

    for trial in range(n_trials):
        for strategy in strategies:
            trial_result = run_trial(
                X_train, y_train, X_test, y_test,
                target_Ks=k_grid, strategy=strategy, tau=default_tau,
                seed=trial * 100 + hash(strategy) % 1000,
            )
            for K in k_grid:
                all_results[strategy][K]["accs"].append(trial_result[K]["acc"])
                all_results[strategy][K]["S_vals"].append(trial_result[K]["S"])
                all_results[strategy][K]["stop_flags"].append(trial_result[K]["stopped"])
        if (trial + 1) % 10 == 0:
            print(f"    Trial {trial + 1}/{n_trials}")

    summary: dict[str, Any] = {
        "task": task_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "pca_dim": int(X_pca.shape[1]),
        "tau": default_tau,
        "n_trials": n_trials,
        "k_grid": k_grid,
    }
    for strategy in strategies:
        for K in k_grid:
            accs = all_results[strategy][K]["accs"]
            S_vals = all_results[strategy][K]["S_vals"]
            stops = all_results[strategy][K]["stop_flags"]
            summary[f"{strategy}_K{K}"] = {
                "mean_acc": float(np.mean(accs)),
                "std_acc": float(np.std(accs)),
                "median_S": float(np.median(S_vals)),
                "early_stop_rate": (
                    float(np.mean(stops)) if strategy == "joint" else 0.0
                ),
            }
    return summary, all_results


def print_summary(summaries: dict[str, dict[str, Any]], k_grid: list[int] = default_k_grid) -> None:
    """Format the per-task summary as ASCII tables."""
    print("\n" + "=" * 78)
    print("Summary: Mean Accuracy by Strategy and K")
    print("=" * 78)
    for task_name, summary in summaries.items():
        print(f"\n{task_name}:")
        print(f"{'K':>6} | {'Random':>12} | {'Uncertainty':>12} | "
              f"{'Joint':>12} | {'Early Stop %':>12}")
        print("-" * 70)
        for K in k_grid:
            print(
                f"{K:>6} | "
                f"{summary[f'random_K{K}']['mean_acc']:>10.3f}±{summary[f'random_K{K}']['std_acc']:.3f} | "
                f"{summary[f'uncertainty_K{K}']['mean_acc']:>10.3f}±{summary[f'uncertainty_K{K}']['std_acc']:.3f} | "
                f"{summary[f'joint_K{K}']['mean_acc']:>10.3f}±{summary[f'joint_K{K}']['std_acc']:.3f} | "
                f"{summary[f'joint_K{K}']['early_stop_rate'] * 100:>10.1f}%"
            )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(
    out_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the §6.7 active-learning study end-to-end and write JSON."""
    from src.datasets import load_all_datasets

    print("=" * 78)
    print("Phase B.7: Active Learning + Saturation Index Combined Experiment")
    print("=" * 78)
    print(f"Strategies: random, uncertainty, joint (tau={default_tau})")
    print(f"K-grid: {default_k_grid}")
    print(f"Trials: {default_n_trials}")
    print(f"PCA dim: {default_pca_dim}")

    print("\nLoading datasets...")
    datasets = load_all_datasets(data_dir=Path(__file__).resolve().parent.parent / "data")
    tasks = [
        ("MNIST_0v1", datasets["MNIST"], 0, 1),
        ("Fashion_3v5", datasets["Fashion"], 3, 5),
        ("BreastCancer", datasets["BreastCancer"], 0, 1),
    ]

    all_summaries: dict[str, dict[str, Any]] = {}
    for task_name, (X, y), ca, cb in tasks:
        summary, _ = run_task_experiment(task_name, X, y, ca, cb)
        all_summaries[task_name] = summary

    print_summary(all_summaries)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(all_summaries, indent=2))
        print(f"\nResults saved to: {out_path}")
    return all_summaries


if __name__ == "__main__":
    main(out_path=results_dir / "phase_b7_summary.json")
