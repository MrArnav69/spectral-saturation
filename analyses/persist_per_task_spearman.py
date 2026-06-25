"""Persist the per-task Spearman tables (Tab.~\\ref{tab:pertask_mnist} and
Tab.~\\ref{tab:pertask_other}) as JSON for byte-level reproducibility.

Reads ``results/all_results.json`` and ``results/extra_binary_results.json``,
computes per-task summaries (erank_inf, A_inf, K_sat at first S<=0.3 point,
within-task Spearman rho between log S and Delta A), and writes
``results/per_task_spearman.json``.

Released alongside the camera-ready version; corroborates the values printed in
Tables~\\ref{tab:pertask_mnist} and~\\ref{tab:pertask_other} of the manuscript.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PRIMARY_FILES = [
    RESULTS_DIR / "all_results.json",
    RESULTS_DIR / "extra_binary_results.json",
]
OUTPUT_PATH = RESULTS_DIR / "per_task_spearman.json"


def load_trials() -> dict:
    out = {}
    for path in PRIMARY_FILES:
        with open(path) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            for k, v in data.items():
                if k in out:
                    # merge overlapping entries
                    out[k].extend(v)
                else:
                    out[k] = list(v)
    return out


def per_task_summary(name: str, trials: list) -> dict:
    """Compute the per-task row printed in Tab.~pertask_mnist / pertask_other."""
    # Sort by K
    by_K = sorted(trials, key=lambda t: t["K"])

    K_grids = [t["K"] for t in by_K]
    eranks = [t["mean_erank"] for t in by_K]
    accs = [t["mean_acc"] for t in by_K]
    S_vals = [t["S"] for t in by_K]
    margins = [t["marginal"] for t in by_K]

    # Asymptotic (use largest K available)
    K_max = max(K_grids)
    erank_inf = float(np.mean([t["mean_erank"] for t in by_K if t["K"] == K_max]))
    A_inf = float(np.mean([t["mean_acc"] for t in by_K if t["K"] == K_max]))

    # K_sat = first K where S <= 0.3
    K_sat = None
    for k, s in zip(K_grids, S_vals):
        if s <= 0.3:
            K_sat = int(k)
            break
    if K_sat is None:
        K_sat = int(K_max)

    # Within-task Spearman rho between log S(K) and the per-doubling accuracy
    # marginal Delta A(K) = mean_acc_at_next_K_ge_2K - mean_acc_at_K.
    # The K grid is irregular (see configure_per_K_grid in analyses/) so we
    # match each K to the smallest later K that satisfies K' >= 2K.
    accs_arr = np.asarray(accs, dtype=float)
    S_arr = np.asarray(S_vals, dtype=float)
    log_S = []
    delta_A = []
    for i, (k, s) in enumerate(zip(K_grids, S_vals)):
        target = 2 * k
        # find minimum j > i such that K_grids[j] >= target
        j = next((jj for jj in range(i + 1, len(K_grids)) if K_grids[jj] >= target), None)
        if j is None:
            break
        log_S.append(np.log(max(S_arr[i], 1e-6)))
        delta_A.append(accs_arr[j] - accs_arr[i])
    rs, p = stats.spearmanr(log_S, delta_A)
    return {
        "task": name,
        "dataset": _dataset(name),
        "erank_inf": erank_inf,
        "A_inf": A_inf,
        "K_sat": K_sat,
        "rho": float(rs),
        "p_value": float(p),
        "K_grid": K_grids,
        "mean_acc": accs,
        "mean_erank": eranks,
        "S": S_vals,
        "marginal": margins,
    }


def _dataset(name: str) -> str:
    if name.startswith("MNIST"):
        return "MNIST"
    if name.startswith("Fashion"):
        return "Fashion-MNIST"
    if name.startswith("Kuzushiji"):
        return "Kuzushiji-MNIST"
    if name.startswith("USPS"):
        return "USPS"
    if name.startswith("CIFAR"):
        return "CIFAR-10"
    if name.startswith("BreastCancer"):
        return "Breast Cancer"
    return "Other"


def main() -> None:
    trials = load_trials()
    rows = [per_task_summary(name, t) for name, t in sorted(trials.items())]
    payload = {
        "schema_version": "1.0",
        "n_tasks": len(rows),
        "tau": 0.3,
        "notes": (
            "erank_inf = mean erank at largest K available; "
            "A_inf = mean accuracy at largest K; "
            "K_sat = first K where S <= 0.3 (or K_max if never reached); "
            "rho = within-task Spearman correlation between log S and the "
            "doubling-pair accuracy marginal Delta A."
        ),
        "tasks": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {OUTPUT_PATH} with {len(rows)} task rows.")


if __name__ == "__main__":
    main()
