"""K-sweeps in PCA space — the headline experiment.

Drives :func:`src.protocols.run_binary_sweep` and
:func:`src.protocols.run_nway_sweep` over the entire PCA battery:

* all 31 binary pairs (17 headline + 14 extras) on the dense 17-element
  K-grid;
* 5 N-way tasks (N=5) across MNIST, Fashion-MNIST, and CIFAR-10 on the
  15-element N-way K-grid.

Outputs
-------

* ``results/all_31_results.json`` — every binary pair keyed by task tag.
* ``results/nway_pca_results.json`` — every N-way task under the same
  keying convention.

Each row carries ``K``, ``mean_acc``, ``std_acc``, ``mean_erank``,
``S`` and ``marginal``; N-way rows additionally carry the within-task
Spearman ``spearman_rho`` / ``spearman_pval`` computed over the
doubling-pair marginals.

Both outputs are disk-cached: re-invoking this script reuses the prior
``results/`` JSON instead of recomputing.

Run::

    python analyses/pca_sweeps.py
"""

from __future__ import annotations

from pathlib import Path

from src.protocols import (
    load_results,
    run_binary_sweep,
    run_nway_sweep,
    save_results,
)

from analyses._shared import (
    bc_ks,
    dense_ks,
    dense_ks_small,
    load_all_with_cifar,
    nway_ks,
    resolve_dataset,
)

results_dir = Path("results")
data_dir = Path("data")

binary_results_path = results_dir / "all_31_results.json"
nway_results_path = results_dir / "nway_pca_results.json"


# ---------------------------------------------------------------------------
# 31 binary task specifications — (dataset_key, task_tag, a, b, pca_dims,
# test_size, k_grid).  pca_dims=None means no PCA (raw input space).
# ---------------------------------------------------------------------------
binary_tasks: list[tuple[str, str, int, int, int | None, int, list[int]]] = [
    # ---- 17 headline pairs -----------------------------------------
    ("MNIST",       "MNIST_0v1",       0, 1, 50, 200, dense_ks),
    ("MNIST",       "MNIST_3v8",       3, 8, 50, 200, dense_ks),
    ("MNIST",       "MNIST_4v9",       4, 9, 50, 200, dense_ks),
    ("MNIST",       "MNIST_1v7",       1, 7, 50, 200, dense_ks),
    ("MNIST",       "MNIST_2v7",       2, 7, 50, 200, dense_ks),
    ("MNIST",       "MNIST_4v7",       4, 7, 50, 200, dense_ks),
    ("MNIST",       "MNIST_5v8",       5, 8, 50, 200, dense_ks),
    ("Fashion",     "Fashion_0v1",     0, 1, 50, 200, dense_ks),
    ("Fashion",     "Fashion_2v6",     2, 6, 50, 200, dense_ks),
    ("Fashion",     "Fashion_3v5",     3, 5, 50, 200, dense_ks),
    ("Fashion",     "Fashion_4v6",     4, 6, 50, 200, dense_ks),
    ("Fashion",     "Fashion_5v7",     5, 7, 50, 200, dense_ks),
    ("Kuzushiji",   "Kuzushiji_0v9",   0, 9, 50, 200, dense_ks),
    ("USPS",        "USPS_1v2",        1, 2, 50, 100, dense_ks_small),
    ("CIFAR",       "CIFAR_0v1",       0, 1, 50, 200, dense_ks),
    ("CIFAR",       "CIFAR_3v5",       3, 5, 50, 200, dense_ks),
    ("BreastCancer", "BreastCancer",   0, 1, None, 50, bc_ks),
    # ---- 14 extras --------------------------------------------------
    ("MNIST",       "MNIST_0v6",       0, 6, 50, 200, dense_ks),
    ("MNIST",       "MNIST_2v3",       2, 3, 50, 200, dense_ks),
    ("MNIST",       "MNIST_5v6",       5, 6, 50, 200, dense_ks),
    ("MNIST",       "MNIST_8v9",       8, 9, 50, 200, dense_ks),
    ("Fashion",     "Fashion_0v2",     0, 2, 50, 200, dense_ks),
    ("Fashion",     "Fashion_0v4",     0, 4, 50, 200, dense_ks),
    ("Fashion",     "Fashion_1v3",     1, 3, 50, 200, dense_ks),
    ("CIFAR",       "CIFAR_2v3",       2, 3, 50, 200, dense_ks),
    ("CIFAR",       "CIFAR_3v7",       3, 7, 50, 200, dense_ks),
    ("CIFAR",       "CIFAR_5v7",       5, 7, 50, 200, dense_ks),
    ("CIFAR",       "CIFAR_4v9",       4, 9, 50, 200, dense_ks),
    ("Kuzushiji",   "Kuzushiji_2v6",   2, 6, 50, 200, dense_ks),
    ("Kuzushiji",   "Kuzushiji_3v4",   3, 4, 50, 200, dense_ks),
    ("USPS",        "USPS_4v9",        4, 9, 50, 200, dense_ks),
]


# ---------------------------------------------------------------------------
# 5 N-way task specifications.
# ---------------------------------------------------------------------------
nway_tasks: dict[str, dict] = {
    "MNIST_M5A_easy":    {"dataset": "MNIST",   "classes": [0, 1, 6, 7, 9]},
    "MNIST_M5B_hard":    {"dataset": "MNIST",   "classes": [3, 5, 8, 2, 4]},
    "Fashion_F5A_easy":  {"dataset": "Fashion", "classes": [0, 1, 5, 7, 9]},
    "Fashion_F5B_hard":  {"dataset": "Fashion", "classes": [2, 3, 4, 6, 8]},
    "CIFAR_C5A_animals": {"dataset": "CIFAR",   "classes": [2, 3, 4, 5, 7]},
}


def run_binary() -> dict:
    """Build the 31-binary-task result dict (cached on disk)."""
    if (cached := load_results(binary_results_path)) is not None:
        print(f"[pca_sweeps] loaded cached binary sweep from {binary_results_path}")
        return cached

    datasets, (X_cifar, y_cifar) = load_all_with_cifar(data_dir)
    results: dict[str, list[dict]] = {}

    for ds_name, tag, a, b, pca_dims, test_size, Ks in binary_tasks:
        X, y = resolve_dataset(ds_name, datasets, (X_cifar, y_cifar))
        if X is None:
            print(f"[pca_sweeps]   SKIP {tag}: dataset {ds_name} not loaded.")
            continue
        results[tag] = run_binary_sweep(
            tag, X, y, a, b,
            Ks=Ks, test_size=test_size, n_trials=50, pca_dims=pca_dims,
        )

    save_results(results, binary_results_path)
    print(f"[pca_sweeps] wrote {binary_results_path} ({len(results)} tasks).")
    return results


def run_nway() -> dict:
    """Build the 5-N-way task result dict (cached on disk)."""
    if (cached := load_results(nway_results_path)) is not None:
        print(f"[pca_sweeps] loaded cached N-way sweep from {nway_results_path}")
        return cached

    datasets, (X_cifar, y_cifar) = load_all_with_cifar(data_dir)
    results: dict[str, list[dict]] = {}

    for tag, cfg in nway_tasks.items():
        X, y = resolve_dataset(cfg["dataset"], datasets, (X_cifar, y_cifar))
        if X is None:
            print(f"[pca_sweeps]   SKIP {tag}.")
            continue
        results[tag] = run_nway_sweep(
            tag, X, y, classes=cfg["classes"], Ks=nway_ks,
            test_size=200, n_trials=50, pca_dims=50,
        )

    save_results(results, nway_results_path)
    print(f"[pca_sweeps] wrote {nway_results_path} ({len(results)} tasks).")
    return results


def main() -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    run_binary()
    run_nway()


if __name__ == "__main__":
    main()
