#!/usr/bin/env python3
"""Orchestrator: run every analysis driver and emit the manuscript figures.

What it does
------------

1. Resolves ``data/`` and ``results/`` via ``analyses._shared``.

2. Runs every driver in dependency order so each driver's inputs are
   guaranteed on disk before the next starts:

   =====  =====================================  ==============================
   Stage  Module                                Output(s)
   =====  =====================================  ==============================
   pca_sweeps       All 31 binary + 5 N-way PCA tasks.
   clip_sweeps      14 CLIP-binary (B/32 + L/14) + 5 N-way.
   clip_dense       Densified 23-element K-grid (B/32 + L/14).
   ablations        PCA-dim, regularisation, classifier-agnosticity.
   multistat        S(erank) vs S(stable_rank) vs S(TwoNN) vs S(MLE).
   tau_transfer     Leave-one-representation-out CV on τ.
   active_learning  Random vs uncertainty vs gating by S(K).
   =====  =====================================  ==============================

   Every driver honours on-disk caching (re-runs are a no-op when the
   output JSON exists), so re-invoking this script is safe.

3. Calls ``src.figures`` to emit the manuscript figure set under
   ``figures/``: per-task K-sweeps grid, decoupling scatter, ablation
   comparisons, CLIP-vs-PCA overlay, N-way saturation, backbone
   comparison.

Usage::

    python run_all_experiments.py              # full pipeline
    python run_all_experiments.py --from 4     # resume from stage 4
    python run_all_experiments.py --to 5       # stop after stage 5
    python run_all_experiments.py --skip-figures
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import analyses._shared  # noqa: F401  -- ensures 'src' is importable
from analyses import (
    ablations,
    active_learning,
    clip_dense,
    clip_sweeps,
    multistat,
    pca_sweeps,
    tau_transfer,
)


STAGES: list[tuple[str, callable]] = [
    ("pca_sweeps",      pca_sweeps.main),
    ("clip_sweeps",     clip_sweeps.main),
    ("clip_dense",      clip_dense.main),
    ("ablations",       ablations.main),
    ("multistat",       multistat.main),
    ("tau_transfer",    tau_transfer.main),
    ("active_learning", active_learning.main),
]


def _emit_figures() -> None:
    """Drive the standard figure emitters via :mod:`src.figures`.

    Lazy import so the orchestrator stays fast on a cold install
    (matplotlib is only needed at figure-emission time).
    """
    from src import figures
    results_dir = Path("results")

    figures.plot_all_sweeps_grid(
        results_dir / "all_31_results.json",
        Path("figures/per_task"),
    )
    figures.plot_decoupling_hypothesis(
        results_dir / "all_31_results.json",
        Path("figures/decoupling.png"),
    )
    figures.plot_nway_saturation(
        results_dir / "nway_pca_results.json",
        Path("figures/nway_saturation.png"),
    )

    pca_b32 = results_dir / "clip_vitb32_binary_results.json"
    pca_l14 = results_dir / "clip_vitl14_binary_results.json"
    if pca_b32.exists():
        figures.plot_clip_vs_pca_comparison(
            results_dir / "all_31_results.json", pca_b32,
            Path("figures/clip_vs_pca_b32.png"),
        )
    if pca_l14.exists():
        figures.plot_clip_vs_pca_comparison(
            results_dir / "all_31_results.json", pca_l14,
            Path("figures/clip_vs_pca_l14.png"),
        )

    abl = results_dir / "ablation_pca_results.json"
    if abl.exists():
        figures.plot_pca_ablation(abl, Path("figures/ablation_pca.png"))
    reg = results_dir / "ablation_reg_results.json"
    if reg.exists():
        figures.plot_reg_ablation(reg, Path("figures/ablation_reg.png"))
    clf = results_dir / "ablation_classifier_results.json"
    if clf.exists():
        figures.plot_classifier_comparison(clf, Path("figures/ablation_classifier.png"))

    print("[run_all] figures emitted under figures/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run every analysis driver and emit manuscript figures.",
    )
    parser.add_argument(
        "--from", dest="from_stage", type=int, default=1,
        help="1-based index of the first stage to run (default: 1).",
    )
    parser.add_argument(
        "--to", dest="to_stage", type=int, default=len(STAGES),
        help="1-based index of the last stage to run (default: last).",
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="skip the matplotlib figure emission.",
    )
    parser.add_argument(
        "--only-figures", action="store_true",
        help="skip the analysis stages entirely; emit only the figures.",
    )
    args = parser.parse_args(argv)

    if args.from_stage < 1 or args.from_stage > len(STAGES):
        parser.error(f"--from must be in 1..{len(STAGES)}")
    if args.to_stage < 1 or args.to_stage > len(STAGES):
        parser.error(f"--to must be in 1..{len(STAGES)}")
    if args.from_stage > args.to_stage:
        parser.error("--from cannot exceed --to")

    print("=" * 78)
    print(" Spectral-saturation paper — full pipeline")
    print("=" * 78)
    print(f" stages {args.from_stage:02d}..{args.to_stage:02d}, "
          f"figures={'yes' if not args.skip_figures else 'no'}, "
          f"only_figures={args.only_figures}\n")

    if not args.only_figures:
        for idx, (name, fn) in enumerate(STAGES, start=1):
            if idx < args.from_stage or idx > args.to_stage:
                continue
            t0 = time.time()
            print(f"\n--- stage {idx:02d}  {name} ---")
            try:
                fn()
            except Exception as e:
                print(f"!! stage {idx:02d} failed: {type(e).__name__}: {e}")
                return 2
            print(f"   ({time.time() - t0:.1f}s)")
    else:
        print("[--only-figures] skipping every analysis stage.")

    if not args.skip_figures:
        print("\n--- figures ---")
        try:
            _emit_figures()
        except Exception as e:
            print(f"!! figures failed: {type(e).__name__}: {e}")
            return 3

    print("\nok — pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
