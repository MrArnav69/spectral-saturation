"""Active-learning comparison.

Wraps :func:`src.active_learning.run_task_experiment`: random baseline
versus uncertainty-only versus uncertainty-gated-by-S(K) on five binary
tasks covering all three representation-difficulty strata (MNIST,
Fashion-MNIST, CIFAR-10).

Output
------

* ``results/active_learning_results.json`` — per-task per-strategy
  per-K ``{mean_acc, std_acc, raw_accs}``.

Run::

    python analyses/active_learning.py           # defaults
    python analyses/active_learning.py --tau 0.5
    python analyses/active_learning.py --n-trials 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import active_learning as al

from analyses._shared import load_all_with_cifar, resolve_dataset

results_dir = Path("results")

# Five binary tasks spanning MNIST, Fashion-MNIST, and CIFAR-10 so all
# three representation-difficulty strata are visible.
default_tasks: list[dict] = [
    {"tag": "MNIST_0v1",   "dataset": "MNIST",   "a": 0, "b": 1, "pca": 50},
    {"tag": "MNIST_3v8",   "dataset": "MNIST",   "a": 3, "b": 8, "pca": 50},
    {"tag": "Fashion_0v1", "dataset": "Fashion", "a": 0, "b": 1, "pca": 50},
    {"tag": "Fashion_3v5", "dataset": "Fashion", "a": 3, "b": 5, "pca": 50},
    {"tag": "CIFAR_3v5",   "dataset": "CIFAR",   "a": 3, "b": 5, "pca": 50},
]


def _prepare(task: dict, datasets, X_cifar, y_cifar):
    X_raw, y_raw = resolve_dataset(task["dataset"], datasets, X_cifar, y_cifar)
    if X_raw is None:
        return None, None
    mask = (y_raw == task["a"]) | (y_raw == task["b"])
    X_t = X_raw[mask].astype(np.float64)
    y_t = np.where(y_raw[mask] == task["a"], 0, 1)
    X_t = StandardScaler().fit_transform(X_t)
    if task.get("pca"):
        actual_d = min(int(task["pca"]), X_t.shape[1] - 1)
        X_t = PCA(n_components=actual_d).fit_transform(X_t)
    return X_t, y_t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the active-learning comparison (random / "
                    "uncertainty / uncertainty-gated-by-S).",
    )
    parser.add_argument("--pca-dim",  type=int,   default=al.default_pca_dim)
    parser.add_argument("--n-trials", type=int,   default=al.default_n_trials)
    parser.add_argument("--k-grid",   type=int,   nargs="+", default=None)
    parser.add_argument("--tau",      type=float, default=al.default_tau)
    args = parser.parse_args()

    results_dir.mkdir(parents=True, exist_ok=True)
    datasets, (X_cifar, y_cifar) = load_all_with_cifar("data")

    k_grid = args.k_grid or al.default_k_grid
    summaries: dict[str, dict[str, dict]] = {}

    for task in default_tasks:
        X_t, y_t = _prepare(task, datasets, X_cifar, y_cifar)
        if X_t is None:
            print(f"[active_learning]   SKIP {task['tag']}: dataset unavailable.")
            continue
        summaries[task["tag"]] = al.run_task_experiment(
            name=task["tag"], X=X_t, y=y_t,
            k_grid=k_grid, n_trials=args.n_trials, tau=args.tau,
        )

    al.print_summary(summaries, k_grid=k_grid)

    out_path = results_dir / "active_learning_results.json"
    with open(out_path, "w") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"[active_learning] wrote {out_path}")


if __name__ == "__main__":
    main()
