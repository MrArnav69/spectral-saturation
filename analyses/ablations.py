"""Ablations — PCA dim, regularisation, classifier-agnosticity.

Three independent ablations run in sequence; each writes its own JSON.

1. **PCA-dim ablation** — ``results/ablation_pca_results.json``.
   Re-runs three tasks (MNIST 0v1, USPS 1v2, Breast Cancer) at
   ``d ∈ {20, 50, 100}`` (or ``d ∈ {5, 10, 20}`` for Breast Cancer,
   which has only 30 raw features).
2. **Regularisation ablation** — ``results/ablation_reg_results.json``.
   Breast Cancer, USPS 1v2, and Fashion 0v1 at ``C ∈ {∞, 1.0, 0.1}``
   evaluated at each task's ``K_peak``.
3. **Classifier-agnostic check** —
   ``results/ablation_classifier_results.json``.  MNIST 3v8 at
   ``K = 4096`` across ``{logistic, nearest_centroid, linear_svm}``.

Each ablation runs 50 (or 20 for the PCA-dim) trials with the same
seed per cell so the numbers are bit-comparable across ``d`` / ``C``
/ classifier.

Run::

    python analyses/ablations.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.protocols import (
    load_results,
    run_binary_sweep,
    sample_and_evaluate,
    save_results,
)

from analyses._shared import (
    bc_ks,
    dense_ks_small,
    load_all_with_cifar,
    resolve_dataset,
)

results_dir = Path("results")
data_dir = Path("data")


# --------------------------------------------------------------------------
# Ablation 1: PCA dim
# --------------------------------------------------------------------------
def run_pca_dim_ablation() -> dict:
    """MNIST 0v1, USPS 1v2, Breast Cancer on multiple PCA dims."""
    out_path = results_dir / "ablation_pca_results.json"
    if (cached := load_results(out_path)) is not None:
        print(f"[ablations] loaded cached PCA-dim ablation from {out_path}")
        return cached

    dims_by_task = {
        "MNIST_0v1":    [20, 50, 100],
        "USPS_1v2":     [20, 50, 100],
        "BreastCancer": [5, 10, 20],
    }
    Ks_by_task = {
        "MNIST_0v1":    [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
                         128, 256, 512, 1024, 2048, 4096],
        "USPS_1v2":     dense_ks_small,
        "BreastCancer": bc_ks,
    }
    spec_by_task = {
        "MNIST_0v1":    {"dataset": "MNIST",        "a": 0, "b": 1},
        "USPS_1v2":     {"dataset": "USPS",         "a": 1, "b": 2},
        "BreastCancer": {"dataset": "BreastCancer", "a": 0, "b": 1},
    }

    datasets, (X_cifar, _) = load_all_with_cifar(data_dir)
    out: dict[str, dict[str, list[dict]]] = {}

    for task_name, dims in dims_by_task.items():
        cfg = spec_by_task[task_name]
        X_raw, y_raw = resolve_dataset(cfg["dataset"], datasets, X_cifar, None)
        if X_raw is None:
            continue
        mask = (y_raw == cfg["a"]) | (y_raw == cfg["b"])
        X_t = X_raw[mask].astype(np.float64)
        y_t = np.where(y_raw[mask] == cfg["a"], 0, 1)
        X_t = StandardScaler().fit_transform(X_t)
        out[task_name] = {}
        for d in dims:
            actual_d = min(d, X_t.shape[1] - 1)
            X_pca = PCA(n_components=actual_d).fit_transform(X_t)
            print(f"\n[ablations]   {task_name} | d={d} | PCA actual={actual_d}")
            out[task_name][str(d)] = run_binary_sweep(
                f"{task_name}_d{d}", X_pca, y_t, 0, 1,
                Ks=Ks_by_task[task_name], test_size=200, n_trials=20,
                pca_dims=None,
            )

    save_results(out, out_path)
    print(f"[ablations] wrote {out_path} ({len(out)} tasks).")
    return out


# --------------------------------------------------------------------------
# Ablation 2: regularisation
# --------------------------------------------------------------------------
def run_reg_ablation(binary_results_path=results_dir / "all_31_results.json") -> dict:
    out_path = results_dir / "ablation_reg_results.json"
    if (cached := load_results(out_path)) is not None:
        print(f"[ablations] loaded cached Reg ablation from {out_path}")
        return cached

    # Pull K_peak from the headline binary sweep if present.
    binary_results = load_results(binary_results_path) or {}
    if not binary_results:
        print("[ablations]   WARNING: no headline binary results found; "
              "defaulting K_peak to K=4096.")

    def k_peak(task_key: str) -> int:
        rows = binary_results.get(task_key, [])
        if not rows:
            return 4096
        return int(max(rows, key=lambda r: r["mean_acc"])["K"])

    spec = {
        "BreastCancer": {"dataset": "BreastCancer", "a": 0, "b": 1,
                          "test_size": 50, "pca_dims": None,
                          "K": lambda: k_peak("BreastCancer")},
        "USPS_1v2":     {"dataset": "USPS",         "a": 1, "b": 2,
                          "test_size": 100, "pca_dims": 50,
                          "K": lambda: k_peak("USPS_1v2")},
        "Fashion_0v1":  {"dataset": "Fashion",      "a": 0, "b": 1,
                          "test_size": 200, "pca_dims": 50,
                          "K": lambda: k_peak("Fashion_0v1")},
    }

    datasets, (X_cifar, _) = load_all_with_cifar(data_dir)
    out: dict[str, dict[str, dict[str, float]]] = {}

    for task_name, cfg in spec.items():
        X_raw, y_raw = resolve_dataset(cfg["dataset"], datasets, X_cifar, None)
        if X_raw is None:
            continue
        mask = (y_raw == cfg["a"]) | (y_raw == cfg["b"])
        X_t = X_raw[mask].astype(np.float64)
        y_t = np.where(y_raw[mask] == cfg["a"], 0, 1)
        X_t = StandardScaler().fit_transform(X_t)
        if cfg["pca_dims"] is not None:
            actual_d = min(cfg["pca_dims"], X_t.shape[1] - 1)
            X_t = PCA(n_components=actual_d).fit_transform(X_t)
        K = cfg["K"]()
        out[task_name] = {}
        for C_val in [np.inf, 1.0, 0.1]:
            c_str = "inf" if C_val == np.inf else str(C_val)
            accs = []
            for trial in range(50):
                acc, _ = sample_and_evaluate(
                    X_t, y_t, K=K, test_size=cfg["test_size"],
                    seed=trial, C=C_val, classifier="logistic",
                )
                accs.append(acc)
            out[task_name][c_str] = {
                "mean_acc": float(np.mean(accs)),
                "std_acc":  float(np.std(accs)),
                "raw_accs": [float(a) for a in accs],
            }
            print(f"[ablations]   {task_name} | C={c_str:>5s}: "
                  f"{np.mean(accs):.4f} ± {np.std(accs):.4f}")

    save_results(out, out_path)
    print(f"[ablations] wrote {out_path} ({len(out)} tasks).")
    return out


# --------------------------------------------------------------------------
# Ablation 3: classifier-agnostic
# --------------------------------------------------------------------------
def run_classifier_ablation() -> dict:
    out_path = results_dir / "ablation_classifier_results.json"
    if (cached := load_results(out_path)) is not None:
        print(f"[ablations] loaded cached Classifier ablation from {out_path}")
        return cached

    datasets, _ = load_all_with_cifar(data_dir)
    X_mnist, y_mnist = datasets["MNIST"]

    mask = (y_mnist == 3) | (y_mnist == 8)
    X_t = X_mnist[mask].astype(np.float64)
    y_t = np.where(y_mnist[mask] == 3, 0, 1)
    X_t = StandardScaler().fit_transform(X_t)
    X_t = PCA(n_components=50).fit_transform(X_t)

    print(f"\n[ablations]   MNIST 3v8 | K=4096 | diagram: all three classifiers")
    out: dict[str, dict[str, float]] = {}
    classifier_map = {
        "Logistic Regression": "logistic",
        "Nearest Centroid":    "nearest_centroid",
        "Linear SVM":          "svm",
    }
    for clf_name, clf_type in classifier_map.items():
        accs = []
        for trial in range(50):
            acc, _ = sample_and_evaluate(
                X_t, y_t, K=4096, test_size=200,
                seed=trial, classifier=clf_type,
            )
            accs.append(acc)
        out[clf_name] = {
            "mean_acc": float(np.mean(accs)),
            "std_acc":  float(np.std(accs)),
            "raw_accs": [float(a) for a in accs],
        }
        print(f"[ablations]     {clf_name:25s}: "
              f"{np.mean(accs):.4f} ± {np.std(accs):.4f}")

    save_results(out, out_path)
    print(f"[ablations] wrote {out_path}.")
    return out


def main() -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}\n# Ablation 1 — PCA dim\n{'=' * 70}")
    run_pca_dim_ablation()
    print(f"\n{'=' * 70}\n# Ablation 2 — Regularisation\n{'=' * 70}")
    run_reg_ablation()
    print(f"\n{'=' * 70}\n# Ablation 3 — Classifier\n{'=' * 70}")
    run_classifier_ablation()


if __name__ == "__main__":
    main()
