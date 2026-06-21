"""Out-of-domain τ transferability across representations (§6.3).

This module implements the leave-one-representation-out (LODO) cross-
validation used to demonstrate that the saturation threshold ``τ``
transfers across radically different feature spaces. Concretely:

* Train the stopping rule on PCA features + CLIP-B/32 data.
* Test on CLIP-L/14 (held-out representation).
* Repeat for every combination.

The CV is computed over the doubling-pair-long table built from each
representation's K-sweep results. Each ``(task, K)`` observation
contributes a binary label ``saturated`` (``ΔA(K) < 0.005``) and the
saturation index ``S(K)`` at the smaller endpoint of the pair. The
scorer is signed so that higher score ⇒ more likely saturated.

Public functions
----------------

* :func:`build_dataset` — convert a result dict into a ``pandas.DataFrame``
  with one row per doubling-pair observation.
* :func:`evaluate_tau` — score a single ``τ`` on a held-out DataFrame.
* :func:`leave_one_rep_out_cv` — full LODO sweep across
  ``{PCA, CLIP-B/32, CLIP-L/14}``.
* :func:`print_transfer_report` — human-readable summary table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.statistics import doubling_pairs_multistat

results_dir = Path(__file__).resolve().parent.parent / "results"
tau_sweep: list[float] = [0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0]
"""Default τ sweep over the stopping-rule threshold."""


def build_dataset(all_results: dict, stat: str = "S") -> pd.DataFrame:
    """Long-format DataFrame; one row per ``(task, K)`` doubling-pair obs.

    Columns: ``task``, ``K``, ``S``, ``delta_acc``, ``saturated``.
    """
    rows = []
    for task, results in all_results.items():
        S_p, M_p, K_p = extract_doubling_pairs(results, stat=stat)
        for s, m, k in zip(S_p, M_p, K_p):
            rows.append({
                "task": task,
                "K": k,
                "S": s,
                "delta_acc": m,
                "saturated": int(m < 0.005),
            })
    return pd.DataFrame(rows)


def extract_doubling_pairs(
    results: list[dict], stat: str = "S"
) -> tuple[list[float], list[float], list[float]]:
    """Tuple ``(S_values, delta_acc, K_values)`` from a task's result rows.

    Drop-in replacement for :func:`scripts.phase_b8_tau_transfer.extract_doubling_pairs`,
    preserving identical convention so the cached JSON keeps working.
    """
    sp, mp = doubling_pairs_multistat(results, stat=stat)
    Ks = [r["K"] for r in results]
    K_pair = []
    for i, K in enumerate(Ks):
        if K % 2 == 0 and K > 2 and (K // 2) in Ks:
            K_pair.append(K // 2)
    return sp, mp, K_pair


def _youden_j(y_true: np.ndarray, scores: np.ndarray, tau: float) -> float:
    """Youden's J statistic at threshold τ on the *negated* scorer space."""
    y_pred = (scores > tau).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr - fpr


def find_optimal_tau(
    y_true: np.ndarray, scores: np.ndarray, tau_candidates: list[float]
) -> float:
    """Return ``τ*`` that maximizes Youden's J over the candidates."""
    best_j, best_tau = -float("inf"), tau_candidates[0]
    for tau in tau_candidates:
        j = _youden_j(y_true, scores, tau)
        if j > best_j:
            best_j, best_tau = j, tau
    return best_tau


def evaluate_tau(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tau: float,
    stat_name: str,
) -> dict[str, Any]:
    """Evaluate a single ``τ`` on ``test_df`` after training on ``train_df``."""
    train_scores = -train_df["S"].values
    test_scores = -test_df["S"].values
    y_test = test_df["saturated"].values

    auc = roc_auc_score(y_test, test_scores)

    # Negated-score space: predicting saturate ⇒ score > -τ ⇒ S < τ.
    y_pred = (test_scores > -tau).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_test == 1)))
    fp = int(np.sum((y_pred == 1) & (y_test == 0)))
    fn = int(np.sum((y_pred == 0) & (y_test == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    youden_j = _youden_j(y_test, test_scores, -tau)

    return {
        "tau": tau,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "youden_j": youden_j,
        "n_test": len(y_test),
        "n_pos_test": int(y_test.sum()),
        "stat": stat_name,
    }


def leave_one_rep_out_cv(
    pca_df: pd.DataFrame,
    clip_b_df: pd.DataFrame,
    clip_l_df: pd.DataFrame,
    tau_candidates: list[float],
    stat_name: str,
) -> list[dict[str, Any]]:
    """Perform a leave-one-representation-out CV for each ``τ`` candidate."""
    reps = {"PCA": pca_df, "CLIP-B/32": clip_b_df, "CLIP-L/14": clip_l_df}
    results: list[dict[str, Any]] = []

    for held_out_name, held_out_df in reps.items():
        train_dfs = [df for name, df in reps.items() if name != held_out_name]
        train_df = pd.concat(train_dfs, ignore_index=True)

        y_train = train_df["saturated"].values
        scores_train = -train_df["S"].values
        tau_star = find_optimal_tau(y_train, scores_train, tau_candidates)

        for tau in tau_candidates:
            res = evaluate_tau(train_df, held_out_df, tau, stat_name)
            res["train_reps"] = "+".join(
                sorted(n for n in reps.keys() if n != held_out_name)
            )
            res["test_rep"] = held_out_name
            res["tau_star_train"] = tau_star
            results.append(res)

    return results


def _within_rep_loto_auc(df: pd.DataFrame) -> float:
    """Leave-one-task-out AUC within a single representation."""
    aucs: list[float] = []
    for task in df["task"].unique():
        test = df[df["task"] == task]
        if len(test["saturated"].unique()) < 2:
            continue
        try:
            aucs.append(roc_auc_score(test["saturated"], -test["S"]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


def print_transfer_report(
    pca_results_path: str | Path = "all_results_multistat.json",
    clip_b_path: str | Path = "clip_vitb32_dense_results.json",
    clip_l_path: str | Path = "clip_vitl14_dense_results.json",
    tau_candidates: list[float] = tau_sweep,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full LODO transferability report and return the JSON payload."""
    pca_results = json.loads(Path(results_dir / pca_results_path).read_text())
    clip_b_results = json.loads(Path(results_dir / clip_b_path).read_text())
    clip_l_results = json.loads(Path(results_dir / clip_l_path).read_text())

    pca_df = build_dataset(pca_results, stat="S")
    clip_b_df = build_dataset(clip_b_results, stat="S")
    clip_l_df = build_dataset(clip_l_results, stat="S")

    print("\n" + "=" * 78)
    print("Leave-One-Representation-Out Cross-Validation (τ sweep)")
    print("=" * 78)
    cv_results = leave_one_rep_out_cv(
        pca_df, clip_b_df, clip_l_df, tau_candidates, stat_name="erank"
    )

    for result in cv_results:
        print(
            f"  Train: {result['train_reps']:<15} → Test: {result['test_rep']:<10} | "
            f"τ={result['tau']:<4} | AUC={result['auc']:.3f} | "
            f"F1={result['f1']:.3f} | J={result['youden_j']:+.3f}"
        )

    df_results = pd.DataFrame(cv_results)
    best_by_split = df_results.loc[
        df_results.groupby(["train_reps", "test_rep"])["youden_j"].idxmax()
    ]

    pca_in_domain = _within_rep_loto_auc(pca_df)
    clip_b_in_domain = _within_rep_loto_auc(clip_b_df)
    clip_l_in_domain = _within_rep_loto_auc(clip_l_df)

    payload = {
        "tau_sweep": tau_candidates,
        "cv_results": cv_results,
        "best_by_split": best_by_split.to_dict(orient="records"),
        "in_domain_aucs": {
            "PCA": pca_in_domain,
            "CLIP-B/32": clip_b_in_domain,
            "CLIP-L/14": clip_l_in_domain,
        },
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "tau_transferability_results.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to: {out_path / 'tau_transferability_results.json'}")

    return payload


if __name__ == "__main__":
    print_transfer_report()
