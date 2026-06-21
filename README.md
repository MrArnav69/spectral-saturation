# Spectral Saturation in Few-Shot Linear Classification

Reproducibility package for the study of **spectral saturation** in
low-data binary classification regimes. The repository contains every
experiment, statistical analysis, and publication-quality figure
reported in the manuscript.

---

## Overview

The central object of study is the **effective rank** of the pooled
within-class sample covariance matrix $\widehat{\Sigma}_W(K)$ as a
function of the number of labeled examples per class $K$. The effective
rank is defined as the exponential Shannon entropy of the normalised
eigenvalue spectrum of the spectrum:

$$
\operatorname{erank}(\Sigma) = \exp\!\left(-\sum_i p_i \log p_i\right),
\qquad p_i = \frac{\lambda_i}{\sum_j \lambda_j}.
$$

The **saturation index** $S(K) = \operatorname{erank}\!\left(\widehat{\Sigma}_W(K)\right) / K$
is the label-free, classifier-free diagnostic that the paper studies.
It predicts when adding more support samples stops converting into
classification accuracy.

### Experimental scope

The headline result is computed over **31 binary classification tasks**
drawn from 6 datasets and 2 data modalities (image and tabular), with
5 additional **N-way (N=5)** tasks to extend the diagnostic beyond the
binary setting. Shot counts are swept over a geometric K-grid

$$
K \in \{2,\, 3,\, 4,\, 6,\, 8,\, 12,\, 16,\, 24,\, 32,\, 48,\, 64,\, 128,\, 256,\, 512,\, 1024,\, 2048,\, 4096\}.
$$

All trials are independent with fixed seed per cell: $T = 50$ for the
image tasks, $T = 50$ with a smaller grid for the tabular task.
OpenCLIP features are encoded once per (dataset, backbone) and cached
on disk; CLIP dimensionality is $512$ (ViT-B/32) or $768$ (ViT-L/14).

| Dataset | Source | Original dim. $d_0$ | Binary pairs | N-way |
|---|---|---|---|---|
| MNIST | OpenML | 784 | 11 (original 7 + 4 extras) | 2 |
| Fashion-MNIST | OpenML | 784 | 8 (original 5 + 3 extras) | 2 |
| Kuzushiji-MNIST | OpenML | 784 | 3 (original 1 + 2 extras) | — |
| USPS | OpenML | 256 | 2 | — |
| CIFAR-10 | Keras | 3072 | 6 (original 2 + 4 extras) | 1 |
| Breast Cancer Wisconsin | sklearn | 30 | 1 (benign vs. malignant) | — |

---

## Repository structure

```
spectral-saturation/
├── README.md                # this file
├── LICENSE                  # MIT license
├── pyproject.toml           # pip-installable metadata
├── environment.yml          # conda specification (alternative install)
├── Makefile                 # one-line targets (install / data / results / figures)
├── run_all_experiments.py   # orchestrator — runs every analysis driver
│
├── src/                     # core library (importable, no side effects)
│   ├── saturation.py        # effective rank, saturation index S(K)
│   ├── geometry.py          # stable rank, TwoNN (Facco 2017), MLE (Levina-Bickel 2004)
│   ├── datasets.py          # dataset loaders (mnist / fashion / kuzushiji / usps / cifar / breast_cancer)
│   ├── clip_features.py     # OpenCLIP encoding with on-disk caching
│   ├── protocols.py         # K-sweep machinery: sample_and_evaluate, run_binary_sweep, run_nway_sweep
│   ├── statistics.py        # pooled ρ, K_sat, decoupling, DeLong AUC, cluster bootstrap
│   ├── tau_transfer.py      # leave-one-representation-out CV on τ
│   ├── active_learning.py   # random / uncertainty / uncertainty-gated-by-S
│   └── figures.py           # matplotlib emitters for the paper figures
│
├── analyses/                # single-purpose drivers — one JSON output per file
│   ├── _shared.py           # K-grids, dataset resolution, CLIP image reshape
│   ├── pca_sweeps.py        # all 31 PCA-binary + 5 PCA-N-way tasks
│   ├── clip_sweeps.py       # CLIP-binary (B/32 + L/14) + 5 CLIP-N-way
│   ├── clip_dense.py        # densified 23-element K-grid for B/32 vs L/14
│   ├── ablations.py         # PCA-dim, regularisation, classifier-agnosticity
│   ├── multistat.py         # S(erank) vs S(stable_rank) vs S(TwoNN) vs S(MLE)
│   ├── tau_transfer.py      # LODO CV on the saturation threshold τ
│   └── active_learning.py   # active-learning strategy comparison
│
├── data/                    # cached datasets and CLIP embeddings (gitignored)
├── results/                 # canonical result artefacts (gitignored)
└── figures/                 # emitted figures (gitignored)
```

A separate `archive/` directory holds pre-refactor code and the
original exploratory notebook. It is gitignored — pre-refactor code is
preserved for traceability but not user-facing.

---

## Installation

### Option A — `pip` (minimal)

```bash
pip install -e .
```

The minimal install pulls in `numpy`, `scikit-learn`, `scipy`, and
`pandas`. Add `matplotlib` for figure emission and `open-clip-torch`
plus `torch` for the CLIP analyses.

### Option B — `conda` (everything bundled)

```bash
conda env create -f environment.yml
conda activate spectral-saturation
```

`environment.yml` pins every dependency at versions tested on
Apple Silicon (MPS path), including `tensorflow` (only used for
CIFAR-10 loading) and `open-clip-torch`.

> **CIFAR-10 fallback.** CIFAR-10 is loaded through `keras.datasets`
> which requires TensorFlow. If TensorFlow is unavailable, every
> non-CIFAR task still runs; the pipeline degrades gracefully and
> notes skipped tasks in the log output.

### Hardware

All experiments in the manuscript were run on an Apple M3 Pro, 18 GB
unified memory, macOS, MPS path. A single full pipeline run takes
roughly 8–12 hours of wall-clock time depending on K-grid density. The
on-disk cache eliminates redundant re-computation on re-run.

---

## Reproducing every result

Run the orchestrator end-to-end:

```bash
python run_all_experiments.py
```

It loads `data/` first, then runs every analyses driver in dependency
order, then emits the manuscript figure set:

| Stage | Driver | Output(s) |
|---|---|---|
| 1 | `pca_sweeps`     | `all_31_results.json`, `nway_pca_results.json` |
| 2 | `clip_sweeps`    | `clip_vitb32_binary_results.json`, `clip_vitl14_binary_results.json`, `nway_clip_results.json` |
| 3 | `clip_dense`     | `clip_vitb32_dense_results.json`, `clip_vitl14_dense_results.json` |
| 4 | `ablations`      | `ablation_pca_results.json`, `ablation_reg_results.json`, `ablation_classifier_results.json` |
| 5 | `multistat`      | `multistat_results.json` |
| 6 | `tau_transfer`   | `tau_transfer_report.json` |
| 7 | `active_learning` | `active_learning_results.json` |
| — | `src.figures`    | `figures/*.png` |

Or via the provided Makefile:

```bash
make install
make data        # downloads datasets + CLIP features (cached)
make results     # all 7 analysis stages
make figures     # emit manuscript figures
make all         # the full pipeline
```

### Running a single driver

Every driver is a self-contained CLI. To rerun only the multistat
analysis:

```bash
python -m analyses.multistat
```

Each driver checks for an existing output JSON before doing any work
— re-invocation is a no-op when the cache file is present. Force a
fresh run by deleting the relevant file first.

### Skipping parts of the pipeline

```bash
python run_all_experiments.py --from 4       # resume from stage 4
python run_all_experiments.py --to 5         # stop after stage 5
python run_all_experiments.py --skip-figures # skip the matplotlib step
```

---

## What every driver writes

| File | Content | Cached on |
|---|---|---|
| `results/all_31_results.json` | 31 binary PCA tasks, one row per `(K, trial)` | first run |
| `results/nway_pca_results.json` | 5 PCA N-way tasks | first run |
| `results/clip_vitb32_binary_results.json`   | 14 binary tasks, ViT-B/32 | first run |
| `results/clip_vitl14_binary_results.json`   | 14 binary tasks, ViT-L/14 | first run |
| `results/nway_clip_results.json`            | 5 N-way tasks, ViT-B/32 | first run |
| `results/clip_vitb32_dense_results.json`    | densified 23-element K-grid for B/32 | first run |
| `results/clip_vitl14_dense_results.json`    | densified 23-element K-grid for L/14 | first run |
| `results/ablation_pca_results.json`         | PCA-dim sweep | first run |
| `results/ablation_reg_results.json`         | regularisation C sweep | first run |
| `results/ablation_classifier_results.json`  | three classifiers at K_peak | first run |
| `results/multistat_results.json`            | S(erank) vs S(stable_rank) vs S(TwoNN) vs S(MLE) | first run |
| `results/tau_transfer_report.json`          | LODO CV report on τ | first run |
| `results/active_learning_results.json`      | active-learning strategy comparison | first run |

All of these are gitignored; they are recomputed deterministically by
the appropriate driver.

---

## Preprocessing protocol

Every preprocessing step is fitted **exclusively on the support set**
$\mathcal{S}_K$ to prevent test-set leakage. The fitted
transformations are then applied to the held-out test set at
evaluation time.

**Image datasets** (MNIST, Fashion-MNIST, Kuzushiji-MNIST, USPS, CIFAR-10):

1. Pixel normalisation to $[0, 1]$.
2. Standard scaling, fit on $\mathcal{S}_K$.
3. PCA to $d = 50$, fit on the standardised $\mathcal{S}_K$.
4. Subtract the support mean in the PCA-reduced space.

**Tabular dataset** (Breast Cancer Wisconsin, $d_0 = 30$): no PCA (raw
dimension is already below 50).

**OpenCLIP features** are kept in their native $512$- or $768$-dim
space — the protocol tests the saturation diagnostic in the raw
embedding space, not after a downstream PCA.

---

## Core API

The library is importable from anywhere inside the project:

```python
from src.saturation import effective_rank, saturation_index
from src.protocols import run_binary_sweep, run_nway_sweep
from src.statistics import (
    pooled_spearman,
    per_task_spearman,
    per_task_table,
    decoupling_corr,
    decoupling_ci,
    cluster_bootstrap_pooled_rho,
    binary_stopping_auc,
    multistat_summary_table,
)
from src.geometry import (
    effective_rank as effective_rank_negclip,
    stable_rank,
    two_nn_intrinsic_dim,
    mle_intrinsic_dim,
)
from src.tau_transfer import build_dataset, leave_one_rep_out_cv
from src.active_learning import run_task_experiment
from src.figures import (
    plot_single_experiment,
    plot_all_sweeps_grid,
    plot_decoupling_hypothesis,
    plot_clip_vs_pca_comparison,
    plot_nway_saturation,
    plot_pca_ablation,
    plot_reg_ablation,
    plot_classifier_comparison,
    plot_backbone_comparison,
)
```

---

## Citation

If you use this code or the saturation diagnostic in your work,
please cite the associated manuscript (forthcoming).

```
Gupta, A. Spectral saturation in few-shot linear classification.
Manuscript, 2026.
```

---

## License

Released under the MIT License — see `LICENSE` for details.

---

## Acknowledgements

Datasets: MNIST (LeCun & Cortes), Fashion-MNIST (Xiao et al. 2017),
Kuzushiji-MNIST (Clanuwat et al. 2018), USPS (Hull 1994), CIFAR-10
(Krizhevsky 2009), Breast Cancer Wisconsin (Street et al. 1993). CLIP
features via OpenCLIP (Cherti et al. 2022).
