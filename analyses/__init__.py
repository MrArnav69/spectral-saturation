"""Single-purpose analysis drivers.

Each module in this package computes one piece of evidence and writes
one (or a few) JSON files under ``results/``.  Run them individually
for surgical updates or via ``run_all_experiments.py`` at the
repository root for the full pipeline.

* :mod:`analyses.pca_sweeps`     — 31 PCA binary + 5 N-way tasks.
* :mod:`analyses.clip_sweeps`    — 14 CLIP binary (B/32 + L/14) + 5 N-way.
* :mod:`analyses.clip_dense`     — densified 23-element K-grid for the
                                   B/32-vs-L/14 K_sat resolution.
* :mod:`analyses.ablations`      — PCA-dim, regularisation, classifier.
* :mod:`analyses.multistat`      — S(erank) vs S(stable_rank) vs S(TwoNN)
                                   vs S(MLE), with bootstrap and DeLong.
* :mod:`analyses.tau_transfer`   — leave-one-representation-out CV on τ.
* :mod:`analyses.active_learning` — random / uncertainty / uncertainty-
                                   gated-by-S(K).

Every runner honours on-disk caching: if its output JSON already
exists, it returns the cached dict instead of recomputing.  To force a
fresh run, delete the file before re-invoking the driver.
"""

from __future__ import annotations

__all__: list[str] = []
