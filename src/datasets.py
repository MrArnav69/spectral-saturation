"""Public dataset loaders with on-disk caching.

Reads an ``.npz`` from ``data/`` if present; otherwise fetches the dataset
from OpenML (image) or ``scikit-learn`` (tabular) and caches it under the
same path. Public functions:

* :func:`load_mnist`, :func:`load_fashion_mnist`, :func:`load_kuzushiji`,
  :func:`load_usps`, :func:`load_breast_cancer`, :func:`load_cifar10`
* :func:`load_all_datasets` — convenience for the headline binary sweep
"""

from __future__ import annotations

import os

import numpy as np


def _download_and_cache(data_dir: str, filename: str) -> None:
    """Fetch a dataset from OpenML/scikit-learn and store it as ``.npz``."""
    path = os.path.join(data_dir, filename)
    os.makedirs(data_dir, exist_ok=True)
    print(f"\nDataset {filename} not found at '{path}'. Downloading from public source...")
    from sklearn.datasets import fetch_openml

    dispatch = {
        "mnist.npz":          ("mnist_784",         1),
        "fashion_mnist.npz":  ("Fashion-MNIST",     1),
        "kuzushiji.npz":      ("Kuzushiji-MNIST",   1),
        "usps.npz":           ("usps",              1),
    }
    if filename in dispatch:
        openml_id, version = dispatch[filename]
        data = fetch_openml(openml_id, version=version, parser="auto", as_frame=False)
        X = data.data.astype(np.float64)
        y = data.target.astype(np.int64)
    elif filename == "breast_cancer.npz":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = data.data.astype(np.float64)
        y = data.target.astype(np.int64)
    else:
        raise ValueError(f"Unknown dataset filename: {filename}")

    np.savez(path, X=X, y=y)
    print(f"Dataset {filename} successfully downloaded and cached at '{path}'.")


def _load_npz(data_dir: str, filename: str, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        _download_and_cache(data_dir, filename)
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    if normalize and filename in ("mnist.npz", "fashion_mnist.npz", "kuzushiji.npz"):
        X = X / 255.0
    return X, y


def load_mnist(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
    return _load_npz(data_dir, "mnist.npz", normalize=True)


def load_fashion_mnist(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
    return _load_npz(data_dir, "fashion_mnist.npz", normalize=True)


def load_kuzushiji(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
    return _load_npz(data_dir, "kuzushiji.npz", normalize=True)


def load_usps(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
    return _load_npz(data_dir, "usps.npz", normalize=False)


def load_breast_cancer(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
    return _load_npz(data_dir, "breast_cancer.npz", normalize=False)


def load_cifar10() -> tuple[np.ndarray, np.ndarray]:
    """Return CIFAR-10 as ``(X, y)`` with shape ``(N, 3072)`` and label ``y ∈ {0..9}``.

    Falls back to ``(None, None)`` if TensorFlow is not installed; the rest
    of the pipeline degrades gracefully (CIFAR tasks are skipped, all other
    surfaces still run).
    """
    try:
        from tensorflow.keras.datasets import cifar10
        (X, y), (_, _) = cifar10.load_data()
        X = X.reshape(-1, 3072).astype(np.float64) / 255.0
        y = y.flatten()
        return X, y
    except ImportError:
        print("Tensorflow not installed. Cannot load CIFAR-10.")
        return None, None  # type: ignore[return-value]


def load_all_datasets(data_dir: str = "data") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return every dataset (excluding CIFAR) keyed by short name."""
    return {
        "MNIST":        load_mnist(data_dir),
        "Fashion":      load_fashion_mnist(data_dir),
        "Kuzushiji":    load_kuzushiji(data_dir),
        "USPS":         load_usps(data_dir),
        "BreastCancer": load_breast_cancer(data_dir),
    }
