import os
import json
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.datasets import load_all_datasets, load_cifar10
from src.experiment import run_experiment, save_results, load_results, sample_and_evaluate
from src.visualization import (
    plot_all_sweeps_grid,
    plot_decoupling_hypothesis,
    plot_saturation_vs_marginal_gain,
    plot_pca_ablation,
    plot_reg_ablation,
    plot_classifier_comparison,
)

# ---------------------------------------------------------------------------
# Dense K-grid (shared across all image/tabular tasks, capped per dataset)
# ---------------------------------------------------------------------------
DENSE_Ks        = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096]
DENSE_Ks_small  = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512]   # USPS / smaller sets
BC_Ks           = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100]             # Breast Cancer

def main():
    # Make sure output directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load datasets
    # -----------------------------------------------------------------------
    print("Loading datasets...")
    datasets = load_all_datasets(data_dir='data')

    X_mnist,     y_mnist     = datasets['MNIST']
    X_fashion,   y_fashion   = datasets['Fashion']
    X_kuzushiji, y_kuzushiji = datasets['Kuzushiji']
    X_usps,      y_usps      = datasets['USPS']
    X_bc,        y_bc        = datasets['BreastCancer']

    print("Loading CIFAR-10...")
    X_cifar, y_cifar = load_cifar10()

    # -----------------------------------------------------------------------
    # 2. Standard K-sweeps
    # -----------------------------------------------------------------------
    results_path = 'results/all_results.json'
    all_results = load_results(results_path)

    if all_results is not None:
        print("Loaded pre-computed sweeps results from results/all_results.json")
    else:
        all_results = {}

        # -- MNIST --
        for pair in [(0, 1), (3, 8), (4, 9), (1, 7), (2, 7), (4, 7), (5, 8)]:
            a, b = pair
            tag = f'MNIST_{a}v{b}'
            all_results[tag] = run_experiment(
                tag, X_mnist, y_mnist, a, b,
                DENSE_Ks, test_size=200, n_trials=50, pca_dims=50
            )

        # -- Fashion-MNIST --
        for pair in [(0, 1), (2, 6), (3, 5), (4, 6), (5, 7)]:
            a, b = pair
            tag = f'Fashion_{a}v{b}'
            all_results[tag] = run_experiment(
                tag, X_fashion, y_fashion, a, b,
                DENSE_Ks, test_size=200, n_trials=50, pca_dims=50
            )

        # -- Kuzushiji-MNIST --
        all_results['Kuzushiji_0v1'] = run_experiment(
            'Kuzushiji_0v1', X_kuzushiji, y_kuzushiji, 0, 1,
            DENSE_Ks, test_size=200, n_trials=50, pca_dims=50
        )

        # -- USPS --
        all_results['USPS_1v2'] = run_experiment(
            'USPS_1v2', X_usps, y_usps, 1, 2,
            DENSE_Ks_small, test_size=100, n_trials=50, pca_dims=50
        )

        # -- Breast Cancer --
        all_results['BreastCancer'] = run_experiment(
            'BreastCancer', X_bc, y_bc, 0, 1,
            BC_Ks, test_size=50, n_trials=100, pca_dims=None
        )

        # -- CIFAR-10 --
        if X_cifar is not None:
            all_results['CIFAR_0v1'] = run_experiment(
                'CIFAR_0v1', X_cifar, y_cifar, 0, 1,
                DENSE_Ks, test_size=200, n_trials=50, pca_dims=50
            )
            all_results['CIFAR_3v5'] = run_experiment(
                'CIFAR_3v5', X_cifar, y_cifar, 3, 5,
                DENSE_Ks, test_size=200, n_trials=50, pca_dims=50
            )

        print("Saving sweeps results...")
        save_results(all_results, results_path)

    # -----------------------------------------------------------------------
    # 3. Multi-task PCA Ablation   (d ∈ {20, 50, 100} or {5, 10, 20} for BC)
    # -----------------------------------------------------------------------
    pca_results_path = 'results/ablation_pca_results.json'
    ablation_pca_results = load_results(pca_results_path)

    if ablation_pca_results is not None:
        print("Loaded pre-computed PCA ablation results from results/ablation_pca_results.json")
    else:
        print("\nRunning Multi-Task PCA ablation...")
        ablation_pca_results = {}

        # ---- MNIST 0v1 ----
        mask = (y_mnist == 0) | (y_mnist == 1)
        X_m01, y_m01 = X_mnist[mask].astype(np.float64), np.where(y_mnist[mask] == 0, 0, 1)
        X_m01_sc = StandardScaler().fit_transform(X_m01)
        ablation_pca_results['MNIST_0v1'] = {}
        for d in [20, 50, 100]:
            X_pca = PCA(n_components=d).fit_transform(X_m01_sc)
            ablation_pca_results['MNIST_0v1'][str(d)] = run_experiment(
                f'MNIST_0v1_d{d}', X_pca, y_m01, 0, 1,
                DENSE_Ks, test_size=200, n_trials=20, pca_dims=None
            )

        # ---- USPS 1v2 ----
        mask = (y_usps == 1) | (y_usps == 2)
        X_u12, y_u12 = X_usps[mask].astype(np.float64), np.where(y_usps[mask] == 1, 0, 1)
        X_u12_sc = StandardScaler().fit_transform(X_u12)
        ablation_pca_results['USPS_1v2'] = {}
        for d in [20, 50, 100]:
            actual_d = min(d, X_u12_sc.shape[1] - 1)
            X_pca = PCA(n_components=actual_d).fit_transform(X_u12_sc)
            ablation_pca_results['USPS_1v2'][str(d)] = run_experiment(
                f'USPS_1v2_d{d}', X_pca, y_u12, 0, 1,
                DENSE_Ks_small, test_size=100, n_trials=20, pca_dims=None
            )

        # ---- Breast Cancer (use d ∈ {5, 10, 20}, only 30 features) ----
        mask = (y_bc == 0) | (y_bc == 1)
        X_bc2, y_bc2 = X_bc[mask].astype(np.float64), np.where(y_bc[mask] == 0, 0, 1)
        X_bc2_sc = StandardScaler().fit_transform(X_bc2)
        ablation_pca_results['BreastCancer'] = {}
        for d in [5, 10, 20]:
            X_pca = PCA(n_components=d).fit_transform(X_bc2_sc)
            ablation_pca_results['BreastCancer'][str(d)] = run_experiment(
                f'BreastCancer_d{d}', X_pca, y_bc2, 0, 1,
                BC_Ks, test_size=50, n_trials=20, pca_dims=None
            )

        save_results(ablation_pca_results, pca_results_path)

    # -----------------------------------------------------------------------
    # 4. Multi-task Regularization Ablation  (C ∈ {∞, 1.0, 0.1} at K_peak)
    # -----------------------------------------------------------------------
    reg_results_path = 'results/ablation_reg_results.json'
    ablation_reg_results = load_results(reg_results_path)

    if ablation_reg_results is not None:
        print("Loaded pre-computed Reg ablation results from results/ablation_reg_results.json")
    else:
        print("\nRunning Multi-Task Regularization ablation...")
        ablation_reg_results = {}

        # Helper: find K_peak for a task in all_results
        def get_k_peak(task_key):
            res = all_results.get(task_key, [])
            if not res:
                return DENSE_Ks[-1]
            return max(res, key=lambda r: r['mean_acc'])['K']

        # Tasks and their preprocessed arrays
        reg_tasks = {
            'BreastCancer': {
                'X': X_bc, 'y': y_bc, 'ca': 0, 'cb': 1,
                'test_size': 50, 'pca_dims': None, 'key': 'BreastCancer'
            },
            'USPS_1v2': {
                'X': X_usps, 'y': y_usps, 'ca': 1, 'cb': 2,
                'test_size': 100, 'pca_dims': 50, 'key': 'USPS_1v2'
            },
            'Fashion_0v1': {
                'X': X_fashion, 'y': y_fashion, 'ca': 0, 'cb': 1,
                'test_size': 200, 'pca_dims': 50, 'key': 'Fashion_0v1'
            },
        }

        for task_name, cfg in reg_tasks.items():
            K_peak = get_k_peak(cfg['key'])
            print(f"\n  {task_name}: K_peak={K_peak}")
            ablation_reg_results[task_name] = {}

            # Pre-process X
            mask = (cfg['y'] == cfg['ca']) | (cfg['y'] == cfg['cb'])
            X_t = cfg['X'][mask].astype(np.float64)
            y_t = np.where(cfg['y'][mask] == cfg['ca'], 0, 1)
            X_t = StandardScaler().fit_transform(X_t)
            if cfg['pca_dims'] is not None:
                actual_d = min(cfg['pca_dims'], X_t.shape[1] - 1)
                X_t = PCA(n_components=actual_d).fit_transform(X_t)

            for C_val in [np.inf, 1.0, 0.1]:
                C_str = 'inf' if C_val == np.inf else str(C_val)
                accs = []
                for trial in range(50):
                    acc, _ = sample_and_evaluate(
                        X_t, y_t, K=K_peak,
                        test_size=cfg['test_size'], seed=trial, C=C_val,
                        classifier='logistic'
                    )
                    accs.append(acc)
                ablation_reg_results[task_name][C_str] = {
                    'mean_acc': float(np.mean(accs)),
                    'std_acc':  float(np.std(accs)),
                    'raw_accs': [float(a) for a in accs]
                }
                print(f"    C={C_str:>6s}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}")

        save_results(ablation_reg_results, reg_results_path)

    # -----------------------------------------------------------------------
    # 5. Classifier-Agnostic Check  (LR, NearestCentroid, Linear SVM at K=4096)
    # -----------------------------------------------------------------------
    clf_results_path = 'results/ablation_classifier_results.json'
    ablation_clf_results = load_results(clf_results_path)

    if ablation_clf_results is not None:
        print("Loaded pre-computed classifier ablation results from results/ablation_classifier_results.json")
    else:
        print("\nRunning Classifier-Agnostic ablation (MNIST 3v8, K=4096)...")

        # Pre-process MNIST 3v8 with PCA-50
        mask_38 = (y_mnist == 3) | (y_mnist == 8)
        X_38 = X_mnist[mask_38].astype(np.float64)
        y_38 = np.where(y_mnist[mask_38] == 3, 0, 1)
        X_38 = StandardScaler().fit_transform(X_38)
        X_38 = PCA(n_components=50).fit_transform(X_38)

        ablation_clf_results = {}
        clf_map = {
            'Logistic Regression': 'logistic',
            'Nearest Centroid':    'nearest_centroid',
            'Linear SVM':          'svm',
        }
        for clf_name, clf_type in clf_map.items():
            accs = []
            for trial in range(50):
                acc, _ = sample_and_evaluate(
                    X_38, y_38, K=4096, test_size=200,
                    seed=trial, classifier=clf_type
                )
                accs.append(acc)
            ablation_clf_results[clf_name] = {
                'mean_acc': float(np.mean(accs)),
                'std_acc':  float(np.std(accs)),
                'raw_accs': [float(a) for a in accs]
            }
            print(f"  {clf_name:25s}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}")

        save_results(ablation_clf_results, clf_results_path)

    # -----------------------------------------------------------------------
    # 6. Generate Figures
    # -----------------------------------------------------------------------
    print("\nGenerating figures...")

    plot_all_sweeps_grid(all_results,            'figures/all_sweeps.png')
    plot_decoupling_hypothesis(all_results,      'figures/decoupling.png')
    plot_saturation_vs_marginal_gain(all_results,'figures/saturation_vs_marginal.png')
    plot_pca_ablation(ablation_pca_results,      'figures/pca_ablation.png')
    plot_reg_ablation(ablation_reg_results,      'figures/reg_ablation.png')
    plot_classifier_comparison(ablation_clf_results, 'figures/classifier_comparison.png')

    print("All figures saved to 'figures/' directory.")

if __name__ == '__main__':
    main()
