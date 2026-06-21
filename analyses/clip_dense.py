"""Densified CLIP K-grid — B/32 vs L/14 K_sat resolution.

The standard 17-element K-grid in :mod:`analyses.clip_sweeps` produces
identical ``K_sat`` values for many task pairs across the ViT-B/32 and
ViT-L/14 backbones — a mechanical artefact of grid quantisation rather
than a geometric identity.  This script adds six extra points
``{96, 192, 384, 768, 1536, 3072}`` to the standard grid (bound at
3072 to bound runtime) so ``K_sat`` can be located to the nearest
doubling step.

Outputs
-------

* ``results/clip_vitb32_dense_results.json``
* ``results/clip_vitl14_dense_results.json``

Each output mirrors the 14 standard binary tasks on the densified
23-element grid; the task list is read from
:mod:`analyses._shared.clip_binary_tasks` so this file stays consistent
with :mod:`analyses.clip_sweeps`.
"""

from __future__ import annotations

from pathlib import Path

from src.clip_features import load_clip_features_for_dataset
from src.protocols import (
    load_results,
    run_binary_sweep,
    save_results,
)

from analyses._shared import (
    clip_binary_tasks,
    load_all_with_cifar,
    reshape_for_clip,
    resolve_dataset,
)

results_dir = Path("results")
data_dir = Path("data")

# Standard 17-point grid + 6 densification steps = 23 elements, capped
# at 3072 to keep the extra per-trial cost balanced against the
# 14-task × 50-trial × 2-backbone fan-out.
dense_ks_full = [
    2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
    96, 128, 192, 256, 384, 512, 768,
    1024, 1536, 2048, 3072, 4096,
]
run_ks = [K for K in dense_ks_full if K <= 3072]


def _run_dense(model_name: str, out_path: Path,
               datasets, X_cifar, y_cifar) -> dict:
    if (cached := load_results(out_path)) is not None:
        print(f"[clip_dense] loaded cached dense CLIP ({model_name}) "
              f"from {out_path}")
        return cached

    print(f"\n{'#' * 70}\n# Densified CLIP K-grid — {model_name}\n{'#' * 70}")
    results: dict[str, list[dict]] = {}

    for tag, cfg in clip_binary_tasks.items():
        ds_name = cfg["dataset"]
        X_raw, y_raw = resolve_dataset(ds_name, datasets, X_cifar, y_cifar)
        if X_raw is None:
            print(f"[clip_dense]   SKIP {tag}: {ds_name} unavailable.")
            continue
        images = reshape_for_clip(X_raw, ds_name)
        X_clip, y_clip = load_clip_features_for_dataset(
            dataset_name=ds_name, images_np=images, y=y_raw,
            model_name=model_name, data_dir=str(data_dir),
        )
        results[tag] = run_binary_sweep(
            tag, X_clip, y_clip, cfg["a"], cfg["b"],
            Ks=run_ks, test_size=200, n_trials=50, pca_dims=None,
        )

    save_results(results, out_path)
    print(f"[clip_dense] wrote {out_path} ({len(results)} tasks).")
    return results


def main() -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    datasets, (X_cifar, y_cifar) = load_all_with_cifar(data_dir)
    _run_dense("ViT-B-32", results_dir / "clip_vitb32_dense_results.json",
               datasets, X_cifar, y_cifar)
    _run_dense("ViT-L-14", results_dir / "clip_vitl14_dense_results.json",
               datasets, X_cifar, y_cifar)


if __name__ == "__main__":
    main()
