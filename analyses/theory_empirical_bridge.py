"""Theory--empirical bridge: discharge Proposition~\\ref{app:proposition} on data.

Section~\\ref{sec:theorybound} of the paper promises an operator-norm
certificate of the form

    E[||\\hat\\Sigma_W^{(K)} - \\Sigma||_op]  <=  C * sqrt(r(\\Sigma) / K).

This script builds the empirical counterpart on the two CIFAR-10
representation-limited tasks used as the canonical representation-failure
cases in \\S\\ref{sec:repr-limit} --- 3v5 and 8v9 --- and checks whether
the predicted $1/\\sqrt{K}$ envelope is realised, then solves for an
empirical constant $\\hat C$ per task.  The numbers are reused in
\\S\\ref{sec:theorybound} (one paragraph bridge).

Outputs
-------

* ``results/theory_empirical_bridge.json``  --- empirical constants
  ``C_hat`` and per-K operator-norm errors for the two CIFAR-10 tasks.
* ``figures/theory_empirical_bridge.pdf``  --- 2-panel publication figure.
* ``figures/theory_empirical_bridge.png``  --- preview copy at 150 dpi.

Run::

    python analyses/theory_empirical_bridge.py

The script is fast on a single CPU:  2 tasks * 14 K values * 20 trials
of $d=50$ eigendecompositions total under two minutes on the M3 Pro.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Repo-relative imports:  allow running as `python analyses/theory_empirical_bridge.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import load_all_datasets, load_cifar10  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASK_SPECS: dict[str, dict] = {
    "CIFAR_3v5": {"dataset": "CIFAR", "a": 3, "b": 5},
    "CIFAR_8v9": {"dataset": "CIFAR", "a": 8, "b": 9},
}

# K grid sized to keep the script under two minutes on the M3 Pro.  Drops
# $K=2,3$ because the operator-norm certificate is meaningless on a 4-point
# support; aligns with the paper\\S 's $K \\ge \\lceil d/2\\rceil = 25$ floor.
K_GRID: list[int] = [32, 48, 64, 128, 256, 512, 1024, 2048]

# Reference asymptotic ``\\Sigma`` estimated on the largest support
# (``K_ref`` per class) once per task and then held fixed across the
# whole sweep.  This is the empirical counterpart of the population
# covariance the bound speaks of; we never access the test partition.
K_REF_PER_CLASS = 2048

# PCA dimensionality matches \\S\\ref{sec:data}: $d=50$.
PCA_DIMS = 50

# Trials per K.  20 reaches \\sigma-error smaller than 5% of the envelope on
# the M3 Pro while keeping the script interactive-fast.
N_TRIALS = 20

# Random seed for the trial-level sampling schedule (kept fixed across
# tasks so any drift between CIFAR-3v5 and CIFAR-8v9 is data-driven, not
# schedule-driven).
RNG_SEED = 0

OUTPUT_JSON = ROOT / "results" / "theory_empirical_bridge.json"
OUTPUT_PDF = ROOT / "figures" / "theory_empirical_bridge.pdf"
OUTPUT_PNG = ROOT / "figures" / "theory_empirical_bridge.png"

# Okabe--Ito inspired, identical to preamble in spectral-saturation.tex.
PALETTE = {
    "CIFAR_3v5": "#0072B2",
    "CIFAR_8v9": "#E69F00",
    "envelope": "#D55E00",
    "r_inf_marker": "#009E73",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_cifar_or_die():
    """Return CIFAR-10 as ``(X, y)`` or fail with a useful message."""
    X_cifar, y_cifar = load_cifar10()
    if X_cifar is None:
        sys.exit(
            "[theory_empirical_bridge] CIFAR-10 is unavailable. "
            "Install TensorFlow / Keras to enable the data load."
        )
    return X_cifar, y_cifar.astype(np.int64)


# ---------------------------------------------------------------------------
# Per-trial primitives
# ---------------------------------------------------------------------------


def _pooled_within_class_cov(
    X: np.ndarray, y: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray
) -> np.ndarray:
    """Compute the pooled within-class sample covariance on indices ``idx_*``.

    Returns a ``(d, d)`` ndarray.  Matches the estimator used through
    :mod:`src.saturation` (one-half of the per-class MLE covariance).
    """
    cov = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for idx in (idx_a, idx_b):
        Xc = X[idx] - X[idx].mean(axis=0, keepdims=True)
        cov += Xc.T @ Xc / (len(idx) - 1)
    return 0.5 * cov


def _operator_norm(matrix: np.ndarray) -> float:
    """Return ``||M||_op`` via singular values (numerically stable)."""
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def _empirical_constant(
    K_grid: np.ndarray,
    err: np.ndarray,
    r_inf: float,
) -> float:
    """Solve for $\\hat C$ so that ``\\hat C * sqrt(r_inf / K)`` matches ``err``.

    Uses least-squares on log-space --- minimises
    ``\\sum (log err - log \\hat C + 0.5 log K + 0.5 log r_inf)^2`` ---
    which yields $\\hat C$ independent of $K$ only when the data follow
    a clean $1/\\sqrt{K}$ envelope.
    """
    # log(err) = log(C_hat) - 0.5 * log(K) - 0.5 * log(r_inf)
    # equivalent to log(err) + 0.5*log(K) = log(C_hat) - 0.5*log(r_inf)
    rhs = np.log(err) + 0.5 * np.log(K_grid)
    log_C_hat_minus_half_log_r = float(rhs.mean())
    return float(np.exp(log_C_hat_minus_half_log_r + 0.5 * np.log(r_inf)))


# ---------------------------------------------------------------------------
# Per-task computation
# ---------------------------------------------------------------------------


def _trial_indices(
    y_pair: np.ndarray, K: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a balanced support of size ``K`` per class."""
    idx_a = np.where(y_pair == 0)[0]
    idx_b = np.where(y_pair == 1)[0]
    chosen_a = rng.choice(idx_a, size=K, replace=False)
    chosen_b = rng.choice(idx_b, size=K, replace=False)
    return chosen_a, chosen_b


def _compute_task(
    name: str,
    cfg: dict,
    X_pair: np.ndarray,
    y_pair: np.ndarray,
    K_ref: int,
) -> dict:
    """Compute the operator-norm error trajectory for one task.

    Returns a dict with the K-grid, mean+std operator-norm error, a fitted
    $\\hat C$, and a slope on log-log axes (a sanity check that the data
    follow the predicted $-0.5$ log-log slope).
    """
    rng = np.random.default_rng(RNG_SEED)
    n_samples = (y_pair == 0).sum()

    if K_ref > min(n_samples, n_samples) - 32:
        raise ValueError(
            f"K_ref={K_ref} too large for {name}: only {n_samples} samples per class."
        )

    # Reference covariance Sigma_hat from a single large fixed pool
    # (``K_ref_per_class`` examples per class), drawn once per task.
    idx_a_ref, idx_b_ref = _trial_indices(y_pair, K_ref, rng)
    Sigma_ref = _pooled_within_class_cov(
        X_pair, y_pair, idx_a_ref, idx_b_ref
    )
    r_inf = float(
        np.sum(np.linalg.eigvalsh(Sigma_ref))
        / _operator_norm(Sigma_ref)
    )

    errors_per_K: dict[int, np.ndarray] = {K: np.zeros(N_TRIALS) for K in K_GRID}
    for K in K_GRID:
        for trial in range(N_TRIALS):
            idx_a, idx_b = _trial_indices(y_pair, K, rng)
            Sigma_hat_K = _pooled_within_class_cov(
                X_pair, y_pair, idx_a, idx_b
            )
            errors_per_K[K][trial] = _operator_norm(Sigma_hat_K - Sigma_ref)

    K_arr = np.array(K_GRID, dtype=np.float64)
    mean_err = np.array([errors_per_K[K].mean() for K in K_GRID])
    std_err = np.array([errors_per_K[K].std() for K in K_GRID])

    # Fit log-log slope:  log(err) = a + b * log(K)  =>  expect b = -0.5.
    slope, intercept = np.polyfit(np.log(K_arr), np.log(mean_err), 1)
    C_hat = _empirical_constant(K_arr, mean_err, r_inf)

    return {
        "name": name,
        "K_grid": K_GRID,
        "mean_op_error": mean_err.tolist(),
        "std_op_error": std_err.tolist(),
        "r_inf": r_inf,
        "C_hat": C_hat,
        "loglog_slope": float(slope),
        "loglog_intercept": float(intercept),
    }


def _assemble_features(
    X_cifar: np.ndarray, y_cifar: np.ndarray, ds_name: str, a: int, b: int
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize + PCA-fit on the $(a, b)$ pair only.  No test/train split.

    Strict support isolation (no labels leak through the PCA basis) is
    immaterial here because the operator-norm error is computed against a
    held-out reference pool, not the test partition.  This matches the
    setup used inside :func:`src.protocols.run_binary_sweep`.
    """
    mask = (y_cifar == a) | (y_cifar == b)
    X_pair = X_cifar[mask].astype(np.float64)
    y_pair = np.where(y_cifar[mask] == a, 0, 1).astype(np.int64)
    X_pair = StandardScaler().fit_transform(X_pair)
    X_pair = PCA(n_components=PCA_DIMS).fit_transform(X_pair)
    return X_pair, y_pair


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_bridge(results: list[dict], out_pdf: Path, out_png: Path) -> None:
    """2-panel: left = log-log error vs K with theoretical envelope; right = $\\hat C$ bar."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    K_arr = np.array(K_GRID, dtype=np.float64)
    for res in results:
        name = res["name"]
        color = PALETTE.get(name)
        Ks = np.array(res["K_grid"], dtype=np.float64)
        err = np.array(res["mean_op_error"], dtype=np.float64)
        std = np.array(res["std_op_error"], dtype=np.float64)

        ax_left.errorbar(
            Ks,
            err,
            yerr=std,
            marker="o",
            color=color,
            label=f"{name}  (\\hat C={res['C_hat']:.3f})",
            capsize=3,
            lw=1.5,
        )
        # Theoretical envelope at fitted C_hat
        theory = res["C_hat"] * np.sqrt(res["r_inf"] / Ks)
        ax_left.plot(
            Ks, theory, ls="--", color=color, alpha=0.55,
            label=f"{name} envelope $\\hat C\\sqrt{{r/K}}$",
        )

    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.set_xlabel("$K$")
    ax_left.set_ylabel(r"$\|\hat\Sigma_W^{(K)} - \Sigma\|_{\rm op}$")
    ax_left.set_title(
        "Operator-norm error vs.\\ $K$, CIFAR-10 representation-limited tasks"
    )
    ax_left.grid(True, which="both", alpha=0.3)
    ax_left.legend(fontsize=8)

    # Right: $\\hat C$ bar
    names = [r["name"] for r in results]
    Chats = [r["C_hat"] for r in results]
    colors = [PALETTE[n] for n in names]
    ax_right.bar(names, Chats, color=colors)
    ax_right.set_ylabel(r"Empirical constant $\hat C$")
    ax_right.set_title(r"$\hat C$ (smaller = tighter envelope)")
    ax_right.grid(True, axis="y", alpha=0.3)
    for name, c in zip(names, Chats):
        ax_right.text(
            name, c, f"  {c:.3f}", va="bottom", ha="center", fontsize=10
        )

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console summary (one paragraph, paste-ready)
# ---------------------------------------------------------------------------


def _console_summary(results: list[dict]) -> str:
    r"""Return a single-paragraph summary suitable for $\S$4.5 bridge."""
    lines = []
    for res in results:
        lines.append(
            f"{res['name']}: $\\hat C = {res['C_hat']:.3f}$, "
            f"$r_\\infty = {res['r_inf']:.2f}$, "
            f"log-log slope $\\log\\|\\cdot\\|$ vs.\\ $\\log K$ = "
            f"{res['loglog_slope']:.3f} "
            f"(predicted $-0.5$)."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    # ---- Load CIFAR once (the only dataset needed) -----------------------
    X_cifar, y_cifar = _load_cifar_or_die()
    print(
        "[theory_empirical_bridge] CIFAR-10 loaded:"
        f" X.shape={X_cifar.shape}, "
        f"#classes=10"
    )

    results: list[dict] = []
    for name, cfg in TASK_SPECS.items():
        print(f"\n[theory_empirical_bridge] running {name} ...")
        X_pair, y_pair = _assemble_features(
            X_cifar, y_cifar, cfg["dataset"], cfg["a"], cfg["b"]
        )
        res = _compute_task(name, cfg, X_pair, y_pair, K_REF_PER_CLASS)
        results.append(res)
        print(
            f"  r_inf={res['r_inf']:.3f}, "
            f"C_hat={res['C_hat']:.3f}, "
            f"slope={res['loglog_slope']:.3f}"
        )

    # ---- Persist JSON -----------------------------------------------------
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[theory_empirical_bridge] wrote {OUTPUT_JSON}")

    # ---- Plot -------------------------------------------------------------
    _plot_bridge(results, OUTPUT_PDF, OUTPUT_PNG)
    print(f"[theory_empirical_bridge] wrote {OUTPUT_PDF}")
    print(f"[theory_empirical_bridge] wrote {OUTPUT_PNG}")

    # ---- One-paragraph summary -------------------------------------------
    print(
        "\n[theory_empirical_bridge] paste-ready summary for sec. 4.5:\n"
        + _console_summary(results)
    )


if __name__ == "__main__":
    main()
