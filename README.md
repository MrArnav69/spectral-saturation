# Spectral Saturation in Few-Shot Linear Classification

This repository provides the complete implementation and reproducibility package for the study of **spectral saturation** in low-data binary classification regimes. The codebase supports all experiments, ablation studies, and publication-quality figures reported in the associated work.

---

## Overview

The central object of study is the **effective rank** of the pooled within-class sample covariance matrix, $\widehat{\Sigma}_W(K)$, as a function of the number of labeled examples per class $K$. The effective rank is defined as the exponential Shannon entropy of the normalized eigenvalue spectrum:

$$\operatorname{erank}(\Sigma) = \exp\!\left(-\sum_i p_i \log p_i\right), \quad p_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

The **saturation index** $S(K) = \operatorname{erank}(\widehat{\Sigma}_W(K)) / K$ is the primary diagnostic quantity. It serves as a label-free, classifier-free predictor of marginal accuracy gain for a linear classifier trained on the support set $\mathcal{S}_K$.

### Experimental Scope

Experiments are conducted over **17 binary classification tasks** spanning **6 datasets** and **2 data modalities** (image and tabular). Shot counts are swept over a geometric grid

$$K \in \{2,\, 3,\, 4,\, 6,\, 8,\, 12,\, 16,\, 24,\, 32,\, 48,\, 64,\, 128,\, 256,\, 512,\, 1024,\, 2048,\, 4096\}$$

with $T = 50$ independent random trials per $(K, \text{task})$ pair ($T = 100$ for the tabular dataset). All datasets are publicly available and are fetched automatically from OpenML or `scikit-learn` if not present locally.

| Dataset | Source | Original Dim. $d_0$ | Tasks | $K_{\max}$ |
|---|---|---|---|---|
| MNIST | OpenML | 784 | 0v1, 1v7, 2v7, 3v8, 4v7, 4v9, 5v8 | 4096 |
| Fashion-MNIST | OpenML | 784 | 0v1, 2v6, 3v5, 4v6, 5v7 | 4096 |
| Kuzushiji-MNIST | OpenML | 784 | 0v1 | 4096 |
| USPS | OpenML | 256 | 1v2 | 512 |
| CIFAR-10 | Keras | 3072 | 0v1, 3v5 | 4096 |
| Breast Cancer Wisconsin | sklearn | 30 | malignant vs. benign | 100 |

---

## Repository Structure

```
spectral-saturation/
├── README.md                        # This file
├── environment.yml                  # Conda environment specification
├── run_all.py                       # End-to-end pipeline script
├── .gitignore
│
├── data/                            # Cached dataset files (auto-populated)
│   ├── mnist.npz
│   ├── fashion_mnist.npz
│   ├── kuzushiji.npz
│   ├── usps.npz
│   └── breast_cancer.npz
│
├── src/                             # Core library modules
│   ├── __init__.py
│   ├── metrics.py                   # Effective rank and saturation index (erank, S(K))
│   ├── datasets.py                  # Dataset loaders — local .npz and OpenML fallback
│   ├── experiment.py                # K-sweep runner, sample-and-evaluate, result I/O
│   └── visualization.py             # Publication-ready figure generation
│
├── notebooks/                       # Narrative Jupyter notebooks
│   ├── 01_experiments.ipynb         # K-sweep across all datasets; saves all_results.json
│   ├── 02_statistics.ipynb          # Wilcoxon, Spearman, and Pearson significance tests
│   └── 03_ablations.ipynb           # PCA target dimension and regularization ablations
│
├── results/                         # Cached JSON experiment outputs
│   ├── all_results.json
│   ├── ablation_pca_results.json
│   └── ablation_reg_results.json
│
└── figures/                         # Generated figures (populated by run_all.py)
    ├── all_sweeps.png               # Accuracy and erank grid across all tasks
    ├── decoupling.png               # erank∞ vs. peak accuracy scatter
    ├── saturation_vs_marginal.png   # S(K) vs. marginal accuracy gain ΔA(K)
    ├── pca_ablation.png             # Sensitivity to PCA target dimension d
    ├── reg_ablation.png             # Sensitivity to logistic regularization C
    └── classifier_comparison.png    # Classifier-agnostic verification
```

---

## Environment Setup

### Requirements

The project targets **Python 3.13**. All dependencies are pinned in `environment.yml`:

| Package | Version | Role |
|---|---|---|
| `numpy` | 2.4.6 | Numerical computation |
| `scikit-learn` | 1.9.0 | Classifiers, PCA, StandardScaler, datasets |
| `scipy` | 1.17.1 | Statistical tests (Wilcoxon, Spearman, Pearson) |
| `matplotlib` | 3.10.9 | Figure generation |
| `pandas` | 3.0.3 | Result aggregation |
| `tensorflow` | 2.21.0 | CIFAR-10 loading only (via `keras.datasets`) |
| `jupyter` / `ipykernel` | — | Notebook execution |

### Installation

```bash
# 1. Create and activate the Conda environment
conda env create -f environment.yml
conda activate spectral-saturation

# 2. Register the kernel with Jupyter
python -m ipykernel install --user \
    --name spectral-saturation \
    --display-name "Python (spectral-saturation)"
```

> **Note on TensorFlow.** TensorFlow is required only for CIFAR-10 loading (`keras.datasets.cifar10`). If TensorFlow installation fails on your platform, all non-CIFAR tasks remain fully functional; the pipeline degrades gracefully when CIFAR-10 is unavailable.

---

## Reproducing Results

Two fully equivalent pathways are provided. Both produce identical figures and statistical outputs.

### Pathway 1 — Command-Line Execution

Run the complete pipeline (dataset loading → K-sweeps → ablations → statistical tests → figure generation) with a single command:

```bash
python run_all.py
```

The script executes the following stages in order:

| Stage | Description | Output |
|---|---|---|
| 1 | Dataset loading | In-memory arrays |
| 2 | K-sweep experiments (17 tasks, 50 trials each) | `results/all_results.json` |
| 3 | PCA dimension ablation ($d \in \{20, 50, 100\}$) | `results/ablation_pca_results.json` |
| 4 | Regularization ablation ($C \in \{\infty, 1.0, 0.1\}$) | `results/ablation_reg_results.json` |
| 5 | Classifier-agnostic check (LR, Nearest Centroid, Linear SVM) | `results/ablation_classifier_results.json` |
| 6 | Figure generation | `figures/*.png` |

**Smart caching.** Each stage checks for an existing result file before running. If the file is present, the cached results are loaded immediately and computation is skipped. This allows figures to be regenerated from cache in seconds.

**Force full re-run.** To recompute everything from scratch:

```bash
rm results/*.json
python run_all.py
```

**Automatic dataset download.** If `.npz` files are absent from `data/`, the pipeline fetches them from OpenML and `scikit-learn` automatically and caches them for subsequent runs.

---

### Pathway 2 — Jupyter Notebooks

Open the notebooks sequentially in Jupyter Lab or Jupyter Notebook:

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and execute in order:

#### [`01_experiments.ipynb`](notebooks/01_experiments.ipynb)
Runs the K-sweep protocol across all 17 tasks. For each $(K, \text{task})$ pair:
- Draws $T = 50$ random support sets $\mathcal{S}_K$
- Applies the preprocessing pipeline (pixel normalization → StandardScaler → PCA to $d = 50$ → support centering; tabular datasets skip PCA)
- Trains an unregularized logistic regression classifier on $\mathcal{S}_K$
- Records test accuracy $A(K)$ and $\operatorname{erank}(\widehat{\Sigma}_W(K))$

Saves results to `results/all_results.json`.

#### [`02_statistics.ipynb`](notebooks/02_statistics.ipynb)
Performs the three primary statistical analyses:
- **Wilcoxon signed-rank test** — tests the hypothesis that $\Delta A(K) < 0$ in the deep-saturation sub-regime ($S(K) \leq 0.02$)
- **Spearman correlation** — quantifies the monotone relationship between $S(K)$ and $\Delta A(K)$ within tasks
- **Pearson correlation** — tests the decoupling hypothesis between asymptotic effective rank $\operatorname{erank}_\infty$ and peak classification accuracy $A_\infty$

#### [`03_ablations.ipynb`](notebooks/03_ablations.ipynb)
Assesses the robustness of the saturation signal:
- **PCA dimension ablation**: sweeps $d \in \{20, 50, 100\}$ for image tasks and $d \in \{5, 10, 20\}$ for the tabular task
- **Regularization ablation**: sweeps logistic regression inverse regularization strength $C \in \{\infty, 1.0, 0.1\}$ at each task's peak shot count $K_{\text{peak}}$

> **Caching.** All notebooks load from pre-computed `results/*.json` files when available, enabling instant figure rendering without re-running experiments. Deleting the cache files triggers automatic recomputation.

---

## Core API Reference

### `src/metrics.py`

```python
calculate_effective_rank(cov_matrix: np.ndarray) -> float
```
Computes $\operatorname{erank}(\Sigma) = \exp(-\sum_i p_i \log p_i)$ from a symmetric positive semi-definite covariance matrix. Eigenvalues are clipped to $10^{-12}$ for numerical stability.

```python
saturation_ratio(erank: float, K: int) -> float
```
Returns the saturation index $S(K) = \operatorname{erank} / K$. Returns `0.0` for $K \leq 0$.

### `src/experiment.py`

```python
run_experiment(tag, X, y, class_a, class_b, K_grid, test_size, n_trials, pca_dims) -> list[dict]
```
Executes the full K-sweep for a single binary task. Returns a list of result records, one per $(K, \text{trial})$ pair, each containing `K`, `mean_acc`, `std_acc`, `mean_erank`, and `mean_saturation`.

```python
sample_and_evaluate(X, y, K, test_size, seed, C, classifier) -> tuple[float, float]
```
Performs a single trial: draws a balanced support set of size $2K$, applies preprocessing, trains the specified classifier, and returns `(accuracy, erank)`.

### `src/visualization.py`

| Function | Output file | Description |
|---|---|---|
| `plot_all_sweeps_grid` | `all_sweeps.png` | Grid of $A(K)$ and $\operatorname{erank}(K)$ curves per task |
| `plot_decoupling_hypothesis` | `decoupling.png` | Scatter of $\operatorname{erank}_\infty$ vs. $A_\infty$ |
| `plot_saturation_vs_marginal_gain` | `saturation_vs_marginal.png` | $S(K)$ vs. $\Delta A(K)$ with phase boundary annotations |
| `plot_pca_ablation` | `pca_ablation.png` | $\operatorname{erank}(K)$ curves for $d \in \{20, 50, 100\}$ |
| `plot_reg_ablation` | `reg_ablation.png` | Accuracy comparison across regularization strengths |
| `plot_classifier_comparison` | `classifier_comparison.png` | Classifier-agnostic accuracy at saturation |

---

## Preprocessing Pipeline

All preprocessing is fitted **exclusively on the support set** $\mathcal{S}_K$ to prevent test-set leakage. The same fitted transformations are applied to the held-out test set at evaluation time.

**Image datasets** (MNIST, Fashion-MNIST, Kuzushiji-MNIST, USPS, CIFAR-10):
1. Pixel normalization: $x \leftarrow x / 255$
2. Feature standardization: `StandardScaler` fitted on $\mathcal{S}_K$
3. PCA to $d = 50$ components, fitted on standardized $\mathcal{S}_K$
4. Support centering: subtract the support mean in the PCA-reduced space

**Tabular dataset** (Breast Cancer Wisconsin, $d_0 = 30$):
1. Feature standardization: `StandardScaler` fitted on $\mathcal{S}_K$
2. Support centering: subtract the support mean

PCA is not applied to the tabular dataset as $d_0 = 30 < d = 50$.

---

## License

This repository is released for research and reproducibility purposes. See `LICENSE` for details.
