"""Shared constants and helpers used by every module under ``analyses/``.

Centralises the conventions that previously lived inlined in each
runner: K-grid definitions, CLIP image reshape, dataset resolution
(handles the ``CIFAR`` exception), and the load-everything helper that
combines ``load_all_datasets`` with the ``load_cifar10`` retry path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.datasets import load_all_datasets, load_cifar10

# K-grids used across experiments.  Kept here so any future change is
# made in exactly one place rather than re-declared in each runner.
dense_ks = [
    2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
    128, 256, 512, 1024, 2048, 4096,
]
dense_ks_small = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512]
bc_ks = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100]
nway_ks = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024]

# 14 binary task specifications common to every CLIP backbone.  Both
# ``clip_sweeps`` and ``clip_dense`` read from this single canonical
# list so there is exactly one source of truth for which pairs are in
# the CLIP battery.
clip_binary_tasks: dict[str, dict[str, Any]] = {
    "CIFAR_3v5":   {"dataset": "CIFAR",   "a": 3, "b": 5},
    "CIFAR_0v1":   {"dataset": "CIFAR",   "a": 0, "b": 1},
    "MNIST_0v1":   {"dataset": "MNIST",   "a": 0, "b": 1},
    "MNIST_3v8":   {"dataset": "MNIST",   "a": 3, "b": 8},
    "MNIST_4v9":   {"dataset": "MNIST",   "a": 4, "b": 9},
    "MNIST_1v7":   {"dataset": "MNIST",   "a": 1, "b": 7},
    "MNIST_2v7":   {"dataset": "MNIST",   "a": 2, "b": 7},
    "MNIST_4v7":   {"dataset": "MNIST",   "a": 4, "b": 7},
    "MNIST_5v8":   {"dataset": "MNIST",   "a": 5, "b": 8},
    "Fashion_0v1": {"dataset": "Fashion", "a": 0, "b": 1},
    "Fashion_2v6": {"dataset": "Fashion", "a": 2, "b": 6},
    "Fashion_3v5": {"dataset": "Fashion", "a": 3, "b": 5},
    "Fashion_4v6": {"dataset": "Fashion", "a": 4, "b": 6},
    "Fashion_5v7": {"dataset": "Fashion", "a": 5, "b": 7},
}

# 5 N-way task specifications — same class indices as the PCA N-way.
nway_clip_tasks: dict[str, dict[str, Any]] = {
    "MNIST_M5A_easy":    {"dataset": "MNIST",   "classes": [0, 1, 6, 7, 9]},
    "MNIST_M5B_hard":    {"dataset": "MNIST",   "classes": [3, 5, 8, 2, 4]},
    "Fashion_F5A_easy":  {"dataset": "Fashion", "classes": [0, 1, 5, 7, 9]},
    "Fashion_F5B_hard":  {"dataset": "Fashion", "classes": [2, 3, 4, 6, 8]},
    "CIFAR_C5A_animals": {"dataset": "CIFAR",   "classes": [2, 3, 4, 5, 7]},
}


def resolve_dataset(name: str, datasets: dict, cifar_xy) -> tuple:
    """Return ``(X, y)`` for ``name``, falling back to ``(None, None)``.

    CIFAR-10 is loaded separately because TensorFlow is an optional
    dependency and is not bundled into :func:`load_all_datasets`.
    """
    if name == "CIFAR":
        if cifar_xy != (None, None):
            return cifar_xy
        return None, None
    if name in datasets:
        return datasets[name]
    return None, None


def load_all_with_cifar(data_dir: str | Path = "data") -> tuple:
    """Load MNIST-family datasets **and** CIFAR-10 if present.

    Returns ``(datasets, (X_cifar, y_cifar))``; both tuple slots fall
    back to ``(None, None)`` if CIFAR-10 is unavailable.
    """
    data_dir = Path(data_dir)
    datasets = load_all_datasets(data_dir=str(data_dir))
    X_cifar, y_cifar = load_cifar10()
    return datasets, (X_cifar, y_cifar)


def reshape_for_clip(X_raw: np.ndarray, ds_name: str) -> np.ndarray:
    """Reshape a tabular pixel array to ``(N, H, W[, 3])`` for CLIP.

    The ``.npz`` files store MNIST-family pixels as flat ``(N, 28*28)``
    or ``(N, 32*32*3)`` arrays; OpenCLIP's preprocessing wants the
    spatial layout.
    """
    if X_raw is None:
        return X_raw
    if ds_name == "CIFAR":
        return X_raw.reshape(-1, 32, 32, 3)
    if ds_name in ("MNIST", "Fashion", "Kuzushiji"):
        return X_raw.reshape(-1, 28, 28)
    return X_raw
