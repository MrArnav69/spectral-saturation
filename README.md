# Spectral Saturation in Few-Shot Linear Classification

This repository contains the official implementation and reproducibility package for the paper **"Spectral Saturation in Few-Shot Linear Classification"** (Anonymized for Review).

## Overview

We study the phenomenon of **spectral saturation** in low-data regimes. Specifically, we investigate how the effective rank ($\text{erank}$) of the pooled within-class covariance matrix behaves as the number of training samples per class ($K$) increases. 

### Key Findings
1. **Saturation Behavior**: The effective rank of the pooled within-class covariance matrix plateaus/saturates as $K$ increases.
2. **Predictive Capacity**: The saturation index $S(K) = \text{erank} / K$ accurately predicts the marginal accuracy gain of a linear classifier (logistic regression). When $S(K) < 0.3$, accuracy gains plateau or decay (indicating the *saturation phase*).
3. **Decoupling Hypothesis**: The geometric complexity of a task (represented by its asymptotic effective rank $\text{erank}_\infty$) is decoupled from its classification difficulty (peak accuracy).

---

## Directory Structure

```
spectral-saturation/
├── README.md                      # Paper overview & reproduction instructions
├── environment.yml                # Conda environment configuration
├── .gitignore                     # Git ignore rules
│
├── data/                          # Pre-downloaded npz dataset files
│   ├── mnist.npz
│   ├── fashion_mnist.npz
│   ├── kuzushiji.npz
│   ├── usps.npz
│   └── breast_cancer.npz
│
├── src/                           # Core implementation modules
│   ├── __init__.py
│   ├── metrics.py                 # Effective rank & saturation ratio formulas
│   ├── datasets.py                # Dataset loaders (local npz files & openml)
│   ├── experiment.py              # Sample evaluations & K-sweep runner
│   └── visualization.py           # Publication-ready plotting scripts
│
├── notebooks/                     # Step-by-step narrative notebooks
│   ├── 01_experiments.ipynb       # K-sweep sweeps across all datasets
│   ├── 02_statistics.ipynb        # Statistical validation tests
│   └── 03_ablations.ipynb         # PCA target and regularization ablations
│
├── results/                       # Cached JSON results for instant plotting
│   ├── all_results.json
│   ├── ablation_pca_results.json
│   └── ablation_reg_results.json
│
└── figures/                       # Generated publication-quality figures
    ├── all_sweeps.png             # Accuracy and erank grid sweep
    ├── decoupling.png             # Decoupling hypothesis scatter plot
    ├── saturation_vs_marginal.png # Saturation ratio vs marginal accuracy gain
    └── pca_ablation.png           # PCA target dimension curves
```

---

## Getting Started

### Prerequisites

Create and activate the conda environment using the provided environment file:

```bash
# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate spectral-saturation
```

Register the Conda environment's Python kernel with Jupyter so that notebooks run under the correct dependencies:

```bash
python -m ipykernel install --user --name spectral-saturation --display-name "Python (spectral-saturation)"
```

---

## Running the Project & Reproducing Results

We provide two pathways to reproduce the figures, statistical tests, and ablations in the paper.

### Pathway 1: Command Line Execution (Run All)

To run the entire pipeline (sweeps, statistical tests, ablations, and figures generation) from the command line:

```bash
python run_all.py
```

- **Automatic Downloading**: If the dataset `.npz` files are missing from the `data/` folder, the pipeline will automatically download them from OpenML and scikit-learn on-the-fly and cache them.
- **Smart Caching**: The script will check if results are already computed and cached in `results/`. If found, it loads the cached outputs to generate figures instantly.
- **Force Re-run**: To run all computations from scratch, delete the cached results in the `results/` folder:
  ```bash
  rm results/*.json
  python run_all.py
  ```

### Pathway 2: Interactive Jupyter Notebooks

You can open and run the numbered notebooks sequentially inside Jupyter Lab or Jupyter Notebook:

```bash
# Start the Jupyter server
jupyter notebook
```

Navigate to the `notebooks/` directory and open the notebooks in order:
1. **[`notebooks/01_experiments.ipynb`](file:///Users/mrarnav69/Documents/spectral-saturation/notebooks/01_experiments.ipynb)**: Runs K-sweep sweeps across all 6 datasets, tracking classification accuracy and effective rank. Saves standard sweeps to `results/all_results.json`.
2. **[`notebooks/02_statistics.ipynb`](file:///Users/mrarnav69/Documents/spectral-saturation/notebooks/02_statistics.ipynb)**: Performs the Wilcoxon signed-rank test for oversampling harm, Spearman correlation for $S(K)$, and Pearson correlation for the decoupling hypothesis.
3. **[`notebooks/03_ablations.ipynb`](file:///Users/mrarnav69/Documents/spectral-saturation/notebooks/03_ablations.ipynb)**: Performs PCA target dimensionality and inverse regularization $C$ parameter ablations.

*Note: All notebooks are designed to use cached results from `results/` to render figures and analysis instantly. If you delete the cache, the notebooks will automatically re-run the experiments.*

