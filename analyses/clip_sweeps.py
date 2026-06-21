"""K-sweeps in CLIP space — binary (both backbones) and N-way (B/32).

Drives :func:`src.protocols.run_binary_sweep` and
:func:`src.protocols.run_nway_sweep` over OpenCLIP features at the
standard 17-element K-grid.  Embeddings are loaded once per
(dataset, backbone) pair through
:func:`src.clip_features.load_clip_features_for_dataset` and cached as
``data/clip_<dataset>_<backbone>.npz``.

Outputs
-------

* ``results/clip_vitb32_binary_results.json`` — 14 binary tasks on
  ViT-B/32.
* ``results/clip_vitl14_binary_results.json`` — 14 binary tasks on
  ViT-L/14.
* ``results/nway_clip_results.json`` — 5 N-way tasks on ViT-B/32.

The 14 binary tasks and 5 N-way tasks are sourced from
:mod:`analyses._shared` so this file and :mod:`analyses.clip_dense`
agree by construction.

Run::

    python analyses/clip_sweeps.py
"""

from __future__ import annotations

from pathlib import Path

from src.clip_features import load_clip_features_for_dataset
from src.protocols import (
    load_results,
    run_binary_sweep,
    run_nway_sweep,
    save_results,
)

from analyses._shared import (
    clip_binary_tasks,
    dense_ks,
    load_all_with_cifar,
    nway_clip_tasks,
    nway_ks,
    reshape_for_clip,
    resolve_dataset,
)

results_dir = Path("results")
data_dir = Path("data")

b32_binary_path = results_dir / "clip_vitb32_binary_results.json"
l14_binary_path = results_dir / "clip_vitl14_binary_results.json"
nway_clip_path = results_dir / "nway_clip_results.json"


def _run_clip_binary(model_name: str, out_path: Path,
                     datasets, X_cifar, y_cifar) -> dict:
    """One backbones's binary sweep; loaded from cache when available."""
    if (cached := load_results(out_path)) is not None:
        print(f"[clip_sweeps] loaded cached CLIP binary ({model_name}) "
              f"from {out_path}")
        return cached

    print(f"\n{'#' * 70}\n# CLIP Binary Sweeps — {model_name}\n{'#' * 70}")
    results: dict[str, list[dict]] = {}

    for tag, cfg in clip_binary_tasks.items():
        ds_name = cfg["dataset"]
        X_raw, y_raw = resolve_dataset(ds_name, datasets, X_cifar, y_cifar)
        if X_raw is None:
            print(f"[clip_sweeps]   {tag}: dataset unavailable, skipping.")
            continue
        images = reshape_for_clip(X_raw, ds_name)
        X_clip, y_clip = load_clip_features_for_dataset(
            dataset_name=ds_name, images_np=images, y=y_raw,
            model_name=model_name, data_dir=str(data_dir),
        )
        results[tag] = run_binary_sweep(
            tag, X_clip, y_clip, cfg["a"], cfg["b"],
            Ks=dense_ks, test_size=200, n_trials=50, pca_dims=None,
        )

    save_results(results, out_path)
    print(f"[clip_sweeps] wrote {out_path} ({len(results)} tasks).")
    return results


def _run_nway_clip(out_path: Path, datasets, X_cifar, y_cifar) -> dict:
    """ViT-B/32 N-way sweep (only one backbone is in scope)."""
    if (cached := load_results(out_path)) is not None:
        print(f"[clip_sweeps] loaded cached N-way CLIP results from {out_path}")
        return cached

    print(f"\n{'#' * 70}\n# 5-way K-sweeps — CLIP ViT-B/32\n{'#' * 70}")
    results: dict[str, list[dict]] = {}

    for tag, cfg in nway_clip_tasks.items():
        ds_name = cfg["dataset"]
        X_raw, y_raw = resolve_dataset(ds_name, datasets, X_cifar, y_cifar)
        if X_raw is None:
            print(f"[clip_sweeps]   {tag}: dataset unavailable, skipping.")
            continue
        images = reshape_for_clip(X_raw, ds_name)
        X_clip, y_clip = load_clip_features_for_dataset(
            dataset_name=ds_name, images_np=images, y=y_raw,
            model_name="ViT-B-32", data_dir=str(data_dir),
        )
        results[f"{tag}_CLIP"] = run_nway_sweep(
            f"{tag}_CLIP", X_clip, y_clip,
            classes=cfg["classes"], Ks=nway_ks,
            test_size=200, n_trials=50, pca_dims=None,
        )

    save_results(results, out_path)
    print(f"[clip_sweeps] wrote {out_path} ({len(results)} tasks).")
    return results


def main() -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    datasets, (X_cifar, y_cifar) = load_all_with_cifar(data_dir)

    _run_clip_binary("ViT-B-32", b32_binary_path, datasets, X_cifar, y_cifar)
    _run_clip_binary("ViT-L-14", l14_binary_path, datasets, X_cifar, y_cifar)
    _run_nway_clip(nway_clip_path, datasets, X_cifar, y_cifar)


if __name__ == "__main__":
    main()
