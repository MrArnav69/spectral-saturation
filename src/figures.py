"""Paper figures: every plot that appears in the manuscript is here.

The plotting pipeline is intentionally *function-driven*: every script
in ``analyses/`` imports a single function and writes a single PNG/SVG.
All figures share one font / dpi / gridstyle baseline (set via
``plt.rcParams`` at import time) so the paper reads as one document.

Problem-domain figures:

* :func:`plot_single_experiment` — one task: accuracy (left axis) and
  effective rank (right axis) versus K (log2 scale).
* :func:`plot_all_sweeps_grid` — every binary task in one grid.
* :func:`plot_decoupling_hypothesis` — ``erank_∞`` vs. peak acc scatter.
* :func:`plot_saturation_vs_marginal_gain` — phase-coloured scatter of
  ``S(K)`` against ``ΔA(K)`` (the centerpiece of §4 / Fig. 4).
* :func:`plot_nway_saturation` — same idea, N-way + binary overlay.
* :func:`plot_clip_vs_pca_comparison` — head-to-head per common task.
* :func:`plot_backbone_comparison` — ViT-B/32 vs ViT-L/14 panels.

Ablation figures:

* :func:`plot_pca_ablation` — effect of PCA dim on accuracy and erank.
* :func:`plot_reg_ablation` — effect of L2 strength at K_peak.
* :func:`plot_classifier_comparison` — logistic / centroid / linear-SVC
  accuracy at the largest K, with two-sample-error bars.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Shared baseline — every figure inherits this style without per-call repetition.
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
    "savefig.bbox": "tight",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Color palette — used consistently across the manuscript.
clr_acc = "#1f77b4"
clr_erank = "#d62728"
clr_decouple = "#2ca02c"
clr_pca = "#1f77b4"
clr_clip = "#ff7f0e"
phase_colors = {
    "Saturation": "#d62728",
    "Transition": "#ff7f0e",
    "Exploration": "#1f77b4",
}
phase_bins = [0, 0.3, 1.0, np.inf]
phase_labels = ["Saturation", "Transition", "Exploration"]


def _save(fig: plt.Figure, save_path: str | Path | None, dpi: int = 300) -> None:
    if save_path is None:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Per-task figures
# ---------------------------------------------------------------------------
def plot_single_experiment(
    results: list[dict],
    title: str,
    save_path: str | Path | None = None,
) -> None:
    """Accuracy + effective rank vs. K for one task."""
    Ks = [r["K"] for r in results]
    accs = [r["mean_acc"] for r in results]
    std_accs = [r["std_acc"] for r in results]
    eranks = [r["mean_erank"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xlabel("Training samples per class (K)", fontweight="bold")
    ax1.set_ylabel("Accuracy", color=clr_acc, fontweight="bold")
    ax1.errorbar(
        Ks, accs, yerr=std_accs, fmt="-o", color=clr_acc,
        linewidth=2, elinewidth=1.5, capsize=3, label="Accuracy",
    )
    ax1.tick_params(axis="y", labelcolor=clr_acc)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(Ks)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Effective Rank (erank)", color=clr_erank, fontweight="bold")
    ax2.plot(Ks, eranks, "-s", color=clr_erank, linewidth=2, label="erank")
    ax2.tick_params(axis="y", labelcolor=clr_erank)

    plt.title(f"Spectral Saturation Sweep: {title}", pad=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path, dpi=300)
    plt.close(fig)


def plot_all_sweeps_grid(
    all_results: Mapping[str, list[dict]],
    save_path: str | Path | None = None,
    n_cols: int = 3,
) -> None:
    """One panel per binary task with accuracy and erank overlaid."""
    tasks = list(all_results.keys())
    n_tasks = len(tasks)

    n_rows = int(np.ceil(n_tasks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, task_name in enumerate(tasks):
        ax1 = axes[i]
        results = all_results[task_name]
        Ks = [r["K"] for r in results]
        accs = [r["mean_acc"] for r in results]
        std_accs = [r["std_acc"] for r in results]
        eranks = [r["mean_erank"] for r in results]

        ax1.set_xlabel("K (log scale)", fontsize=9)
        ax1.set_ylabel("Accuracy", color=clr_acc, fontsize=9)
        ax1.errorbar(
            Ks, accs, yerr=std_accs, fmt="-o", color=clr_acc,
            linewidth=1.5, elinewidth=1, capsize=2,
        )
        ax1.tick_params(axis="y", labelcolor=clr_acc)
        ax1.set_xscale("log", base=2)
        tick_every = max(1, len(Ks) // 6)
        ax1.set_xticks(Ks[::tick_every])
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax1.grid(True)
        ax1.set_title(task_name, fontsize=10, fontweight="bold")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Effective Rank", color=clr_erank, fontsize=9)
        ax2.plot(Ks, eranks, "-s", color=clr_erank, linewidth=1.5, markersize=4)
        ax2.tick_params(axis="y", labelcolor=clr_erank)

    for j in range(n_tasks, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Spectral Saturation and Accuracy Sweeps Across All Tasks",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, save_path, dpi=200)
    plt.close(fig)


def plot_decoupling_hypothesis(
    all_results: Mapping[str, list[dict]],
    save_path: str | Path | None = None,
) -> None:
    """``erank_∞`` vs. peak-acc scatter with linear fit and per-point labels."""
    eranks_inf: list[float] = []
    peaks: list[float] = []
    task_names: list[str] = []
    for name, results in all_results.items():
        eranks_inf.append(results[-1]["mean_erank"])
        peaks.append(max(r["mean_acc"] for r in results))
        task_names.append(name)

    plt.figure(figsize=(7, 5.5))
    plt.scatter(
        eranks_inf, peaks, color=clr_decouple, s=100,
        edgecolors="black", zorder=3,
    )
    for name, x, y in zip(task_names, eranks_inf, peaks):
        plt.annotate(
            name, (x, y), textcoords="offset points",
            xytext=(0, 10), ha="center", fontsize=9, fontweight="bold",
        )

    m, b = np.polyfit(eranks_inf, peaks, 1)
    x_range = np.linspace(min(eranks_inf) - 2, max(eranks_inf) + 2, 100)
    plt.plot(x_range, m * x_range + b, color="gray", linestyle="--", alpha=0.7, label="Linear fit")

    plt.xlabel("Effective Rank Asymptote (erank_∞)", fontweight="bold")
    plt.ylabel("Peak Classification Accuracy", fontweight="bold")
    plt.title("Decoupling of Task Difficulty & Geometric Complexity", pad=15, fontweight="bold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300) if save_path else None
    plt.close()


def plot_saturation_vs_marginal_gain(
    all_results: Mapping[str, list[dict]],
    save_path: str | Path | None = None,
) -> None:
    """Phase-coloured scatter of ``S(K)`` against ``ΔA(K)``."""
    all_S: list[float] = []
    all_m: list[float] = []
    for results in all_results.values():
        for i, r in enumerate(results):
            if i == 0:
                continue
            m = r["marginal"]
            if m is None or np.isnan(m):
                continue
            all_S.append(r["S"])
            all_m.append(m)

    df = pd.DataFrame({"S": all_S, "marginal": all_m})
    df["phase"] = pd.cut(
        df["S"], bins=phase_bins, labels=phase_labels,
    )

    plt.figure(figsize=(8, 5))
    for phase, group in df.groupby("phase", observed=False):
        plt.scatter(
            group["S"], group["marginal"],
            label=phase, color=phase_colors[phase],
            alpha=0.7, edgecolors="none", s=60,
        )
    plt.axvline(x=0.3, color="gray", linestyle=":", alpha=0.5)
    plt.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)
    plt.xlabel("Saturation Index S(K) = erank / K", fontweight="bold")
    plt.ylabel("Marginal Accuracy Gain", fontweight="bold")
    plt.title(
        "Saturation Index predicts Marginal Accuracy Returns",
        pad=15, fontweight="bold",
    )
    plt.legend(title="Predicted Phase")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300) if save_path else None
    plt.close()


# ---------------------------------------------------------------------------
# Ablation panels
# ---------------------------------------------------------------------------
def plot_pca_ablation(
    ablation_results: Mapping[str, Mapping[str, list[dict]]],
    save_path: str | Path | None = None,
) -> None:
    """Each task has one panel pair (accuracy | erank); several lines for ``d``."""
    tasks = list(ablation_results.keys())
    n_tasks = len(tasks)
    color_map = {
        "5": "#9467bd", "10": "#e377c2", "20": "#2ca02c",
        "50": "#ff7f0e", "100": "#1f77b4",
    }

    fig, axes = plt.subplots(n_tasks, 2, figsize=(12, 4.5 * n_tasks), squeeze=False)

    for row, task_name in enumerate(tasks):
        task_results = ablation_results[task_name]
        dims = sorted(task_results.keys(), key=lambda x: int(x))

        ax_acc = axes[row, 0]
        ax_erank = axes[row, 1]
        for d in dims:
            results = task_results[d]
            Ks = [r["K"] for r in results]
            accs = [r["mean_acc"] for r in results]
            eranks = [r["mean_erank"] for r in results]
            clr = color_map.get(str(d))
            ax_acc.plot(Ks, accs, "-o", label=f"d={d}", color=clr, linewidth=1.8)
            ax_erank.plot(Ks, eranks, "-s", label=f"d={d}", color=clr, linewidth=1.8)

        ax_acc.set_xscale("log", base=2)
        ax_acc.set_xlabel("K", fontweight="bold")
        ax_acc.set_ylabel("Accuracy", fontweight="bold")
        ax_acc.set_title(f"{task_name} — Accuracy vs K", fontweight="bold")
        ax_acc.legend()
        ax_acc.grid(True)

        ax_erank.set_xscale("log", base=2)
        ax_erank.set_xlabel("K", fontweight="bold")
        ax_erank.set_ylabel("Effective Rank", fontweight="bold")
        ax_erank.set_title(f"{task_name} — erank vs K", fontweight="bold")
        ax_erank.legend()
        ax_erank.grid(True)

    plt.suptitle("Multi-Task PCA Dimensionality Ablation", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, save_path, dpi=200)
    plt.close(fig)


def plot_reg_ablation(
    ablation_reg_results: Mapping[str, Mapping[str, dict[str, float]]],
    save_path: str | Path | None = None,
) -> None:
    """Bar plot of mean accuracy at ``K_peak`` across three ``C`` values."""
    tasks = list(ablation_reg_results.keys())
    n_tasks = len(tasks)
    c_labels = ["inf", "1.0", "0.1"]
    c_display = ["C=∞", "C=1.0", "C=0.1"]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    fig, axes = plt.subplots(1, n_tasks, figsize=(5.5 * n_tasks, 4.5), squeeze=False)
    for col, task_name in enumerate(tasks):
        task_res = ablation_reg_results[task_name]
        ax = axes[0, col]
        means, stds, labels = [], [], []
        for c_str, c_disp in zip(c_labels, c_display):
            if c_str in task_res:
                means.append(task_res[c_str]["mean_acc"])
                stds.append(task_res[c_str]["std_acc"])
                labels.append(c_disp)
        x = np.arange(len(labels))
        bars = ax.bar(
            x, means, yerr=stds, capsize=5, color=colors[: len(labels)],
            edgecolor="black", linewidth=0.8, alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Mean Accuracy", fontweight="bold")
        ax.set_title(task_name, fontweight="bold")
        ax.set_ylim(max(0, min(means) - 0.05), min(1.0, max(means) + 0.05))
        ax.grid(axis="y", alpha=0.4)
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9,
            )
    plt.suptitle("Multi-Task Regularization Ablation (at K_peak)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path, dpi=200)
    plt.close(fig)


def plot_classifier_comparison(
    clf_results: Mapping[str, dict[str, float]],
    save_path: str | Path | None = None,
) -> None:
    """Logistic / centroid / linear-SVC accuracy at the largest K."""
    names = list(clf_results.keys())
    means = [clf_results[n]["mean_acc"] for n in names]
    stds = [clf_results[n]["std_acc"] for n in names]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    plt.figure(figsize=(7, 5))
    x = np.arange(len(names))
    bars = plt.bar(
        x, means, yerr=stds, capsize=8, color=colors[: len(names)],
        edgecolor="black", linewidth=0.9, alpha=0.88, width=0.5,
    )
    plt.xticks(x, names, fontsize=12, fontweight="bold")
    plt.ylabel("Mean Accuracy at K=4096", fontweight="bold", fontsize=12)
    plt.title("Classifier-Agnostic Check: MNIST 3 vs 8 (K=4096)", fontweight="bold", fontsize=13)
    plt.ylim(max(0, min(means) - 0.08), min(1.0, max(means) + 0.08))
    plt.grid(axis="y", alpha=0.4)
    for bar, m, s in zip(bars, means, stds):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
            f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200) if save_path else None
    plt.close()


# ---------------------------------------------------------------------------
# Representation / backbone comparisons
# ---------------------------------------------------------------------------
def plot_clip_vs_pca_comparison(
    clip_results: Mapping[str, list[dict]],
    pca_results: Mapping[str, list[dict]],
    save_path: str | Path | None = None,
) -> None:
    """One panel per common task; PCA dashed, CLIP solid; ratio annotation."""
    common_tasks = [t for t in clip_results if t in pca_results]
    if not common_tasks:
        print("plot_clip_vs_pca_comparison: no common tasks found, skipping.")
        return

    n_tasks = len(common_tasks)
    n_cols = min(3, n_tasks)
    n_rows = int(np.ceil(n_tasks / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    for i, task in enumerate(common_tasks):
        ax = axes_flat[i]
        ax2 = ax.twinx()
        for results, label, color, ls in [
            (pca_results[task], "PCA", clr_pca, "--"),
            (clip_results[task], "CLIP", clr_clip, "-"),
        ]:
            Ks = [r["K"] for r in results]
            accs = [r["mean_acc"] for r in results]
            stds = [r["std_acc"] for r in results]
            eranks = [r["mean_erank"] for r in results]
            ax.errorbar(
                Ks, accs, yerr=stds, fmt=f"{ls}o",
                color=color, linewidth=1.8, elinewidth=1,
                capsize=2, label=f"{label} acc", markersize=4,
            )
            ax2.plot(
                Ks, eranks, f"{ls}s", color=color,
                linewidth=1.2, alpha=0.6, markersize=3,
            )

        erank_pca_inf = pca_results[task][-1]["mean_erank"]
        erank_clip_inf = clip_results[task][-1]["mean_erank"]
        ratio = erank_pca_inf / max(erank_clip_inf, 1e-6)
        ax.text(
            0.97, 0.05,
            f"erank∞ ratio (PCA/CLIP): {ratio:.1f}×",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (log scale)", fontsize=9)
        ax.set_ylabel("Accuracy", color=clr_pca, fontsize=9)
        ax2.set_ylabel("Effective Rank", color="gray", fontsize=8)
        ax.tick_params(axis="y", labelcolor=clr_pca)
        ax.set_title(task, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    for j in range(n_tasks, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(
        "CLIP vs. PCA Space: Accuracy & Effective Rank Sweeps",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, save_path, dpi=200)
    plt.close(fig)


def plot_nway_saturation(
    nway_results: Mapping[str, list[dict]],
    binary_results: Mapping[str, list[dict]] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Phase scatter, N-way alone and overlaid with binary."""
    all_S, all_m, all_phase, all_source = [], [], [], []

    for task_name, results in nway_results.items():
        for i, r in enumerate(results):
            if i == 0:
                continue
            m = r["marginal"]
            if m is None or np.isnan(m):
                continue
            phase = phase_labels[int(np.digitize(r["S"], phase_bins[1:-1]))]
            all_S.append(r["S"])
            all_m.append(m)
            all_phase.append(phase)
            all_source.append("N-way")

    if binary_results is not None:
        for task_name, results in binary_results.items():
            for i, r in enumerate(results):
                if i == 0:
                    continue
                m = r["marginal"]
                if m is None or np.isnan(m):
                    continue
                phase = phase_labels[int(np.digitize(r["S"], phase_bins[1:-1]))]
                all_S.append(r["S"])
                all_m.append(m)
                all_phase.append(phase)
                all_source.append("Binary")

    if not all_S:
        print("plot_nway_saturation: no data, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, source_filter, title_suffix in [
        (axes[0], "N-way", "N-way Tasks (N=5)"),
        (axes[1], None, "N-way + Binary Overlay"),
    ]:
        for phase in phase_labels:
            mask = [
                p == phase and (source_filter is None or s == source_filter)
                for p, s in zip(all_phase, all_source)
            ]
            xs = [all_S[k] for k in range(len(all_S)) if mask[k]]
            ys = [all_m[k] for k in range(len(all_S)) if mask[k]]
            srcs = [all_source[k] for k in range(len(all_S)) if mask[k]]
            if not xs:
                continue

            for src, marker in (("N-way", "D"), ("Binary", "o")):
                xs2 = [x for x, s in zip(xs, srcs) if s == src]
                ys2 = [y for y, s in zip(ys, srcs) if s == src]
                if xs2:
                    ax.scatter(
                        xs2, ys2, color=phase_colors[phase], alpha=0.75,
                        edgecolors="none",
                        s=55 if src == "N-way" else 35,
                        marker=marker,
                        label=f"{phase} ({src})" if source_filter is None else phase,
                    )

        ax.axvline(x=0.3, color="gray", linestyle=":", alpha=0.5, label="τ=0.3")
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="τ=1.0")
        ax.set_xlabel("Saturation Index S(K) = erank / K", fontweight="bold")
        ax.set_ylabel("Marginal Accuracy Gain", fontweight="bold")
        ax.set_title(title_suffix, fontweight="bold")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Phase", fontsize=8)

    plt.suptitle(
        "Saturation Index Predicts Marginal Returns — Phase Boundary Transfer (τ=0.3, τ=1.0)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, save_path, dpi=200)
    plt.close(fig)


def plot_backbone_comparison(
    vit_b_results: Mapping[str, list[dict]],
    vit_l_results: Mapping[str, list[dict]],
    save_path: str | Path | None = None,
) -> None:
    """Side-by-side bar charts: ``erank_∞`` and peak accuracy per backbone."""
    common = [t for t in vit_b_results if t in vit_l_results]
    if not common:
        print("plot_backbone_comparison: no common tasks, skipping.")
        return

    erank_b = [vit_b_results[t][-1]["mean_erank"] for t in common]
    erank_l = [vit_l_results[t][-1]["mean_erank"] for t in common]
    peak_b = [max(r["mean_acc"] for r in vit_b_results[t]) for t in common]
    peak_l = [max(r["mean_acc"] for r in vit_l_results[t]) for t in common]

    x = np.arange(len(common))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, vals_b, vals_l, ylabel, title in [
        (axes[0], erank_b, erank_l, "Asymptotic erank (erank∞)", "Effective Rank Asymptote by Backbone"),
        (axes[1], peak_b, peak_l, "Peak Accuracy", "Peak Accuracy by Backbone"),
    ]:
        bars_b = ax.bar(
            x - width / 2, vals_b, width, label="ViT-B/32",
            color="#1f77b4", edgecolor="black", linewidth=0.7, alpha=0.85,
        )
        bars_l = ax.bar(
            x + width / 2, vals_l, width, label="ViT-L/14",
            color="#ff7f0e", edgecolor="black", linewidth=0.7, alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(common, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for bars, vals in [(bars_b, vals_b), (bars_l, vals_l)]:
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                )

    plt.suptitle(
        "ViT-B/32 vs. ViT-L/14: Spectral Geometry & Performance",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, save_path, dpi=200)
    plt.close(fig)
