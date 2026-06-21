"""Spectral saturation: a label-free diagnostic for few-shot annotation budgets.

This package implements the effective-rank saturation index and the
empirical-statistics pipeline that supports it. It is organized as:

* :mod:`src.saturation`   — the headline quantity ``erank`` and ``S(K) = erank/K``
* :mod:`src.geometry`     — alternative intrinsic-dimension estimators
                            (stable rank, TwoNN, MLE) used in the §6.5 head-to-head
* :mod:`src.datasets`     — loaders for MNIST, Fashion-MNIST, Kuzushiji-MNIST,
                            USPS, CIFAR-10, Breast Cancer
* :mod:`src.clip_features` — OpenCLIP ViT-B/32 and ViT-L/14 encoder for the
                              §6.6 representation transfer study
* :mod:`src.protocols`    — the K-sweep runner (binary and N-way) used by every
                            experiment
* :mod:`src.statistics`   — per-task Spearman, pooled Spearman, Mann-Whitney,
                            Wilcoxon, cluster-bootstrap, DeLong AUC, decoupling CI
* :mod:`src.tau_transfer` — leave-one-representation-out τ transferability
* :mod:`src.active_learning` — random vs uncertainty vs uncertainty-gated-by-S(K)
* :mod:`src.figures`      — paper figures

The empirical surface mirrors the manuscript: 31 binary tasks across 6 datasets
plus 2 CLIP backbones, 5 N-way tasks, 6 ablation studies, 1 active-learning
comparison, 1 τ transferability study.
"""

from __future__ import annotations

__all__ = [
    "saturation",
    "geometry",
    "datasets",
    "clip_features",
    "protocols",
    "statistics",
    "tau_transfer",
    "active_learning",
    "figures",
]
