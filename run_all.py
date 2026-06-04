import os
import json
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.datasets import load_all_datasets, load_cifar10
from src.experiment import run_experiment, save_results, load_results
from src.visualization import (
    plot_all_sweeps_grid,
    plot_decoupling_hypothesis,
    plot_saturation_vs_marginal_gain,
    plot_pca_ablation
)

def main():
    # Make sure output directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # 1. Load datasets
    print("Loading datasets...")
    datasets = load_all_datasets(data_dir='data')
    
    X_mnist, y_mnist = datasets['MNIST']
    X_fashion, y_fashion = datasets['Fashion']
    X_kuzushiji, y_kuzushiji = datasets['Kuzushiji']
    X_usps, y_usps = datasets['USPS']
    X_bc, y_bc = datasets['BreastCancer']
    
    print("Loading CIFAR-10...")
    X_cifar, y_cifar = load_cifar10()
    
    # 2. Define experiment configs
    mnist_Ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    fashion_Ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    kuzushiji_Ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    usps_Ks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    bc_Ks = [2, 4, 8, 16, 32, 64, 100]
    cifar_Ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    results_path = 'results/all_results.json'
    all_results = load_results(results_path)
    
    if all_results is not None:
        print("Loaded pre-computed sweeps results from results/all_results.json")
    else:
        all_results = {}
        # Run sweeps
        # MNIST
        all_results['MNIST_0v1'] = run_experiment('MNIST_0v1', X_mnist, y_mnist, 0, 1, mnist_Ks, 200, n_trials=50, pca_dims=50)
        all_results['MNIST_3v8'] = run_experiment('MNIST_3v8', X_mnist, y_mnist, 3, 8, mnist_Ks, 200, n_trials=50, pca_dims=50)
        all_results['MNIST_4v9'] = run_experiment('MNIST_4v9', X_mnist, y_mnist, 4, 9, mnist_Ks, 200, n_trials=50, pca_dims=50)
        
        # Fashion
        all_results['Fashion_0v1'] = run_experiment('Fashion_0v1', X_fashion, y_fashion, 0, 1, fashion_Ks, 200, n_trials=50, pca_dims=50)
        all_results['Fashion_2v6'] = run_experiment('Fashion_2v6', X_fashion, y_fashion, 2, 6, fashion_Ks, 200, n_trials=50, pca_dims=50)
        
        # Kuzushiji
        all_results['Kuzushiji_0v1'] = run_experiment('Kuzushiji_0v1', X_kuzushiji, y_kuzushiji, 0, 1, kuzushiji_Ks, 200, n_trials=50, pca_dims=50)
        
        # USPS
        all_results['USPS_1v2'] = run_experiment('USPS_1v2', X_usps, y_usps, 1, 2, usps_Ks, 100, n_trials=50, pca_dims=50)
        
        # Breast Cancer
        all_results['BreastCancer'] = run_experiment('BreastCancer', X_bc, y_bc, 0, 1, bc_Ks, 50, n_trials=100, pca_dims=None)
        
        # CIFAR-10 (if loaded successfully)
        if X_cifar is not None:
            all_results['CIFAR_0v1'] = run_experiment('CIFAR_0v1', X_cifar, y_cifar, 0, 1, cifar_Ks, 200, n_trials=50, pca_dims=50)
            all_results['CIFAR_3v5'] = run_experiment('CIFAR_3v5', X_cifar, y_cifar, 3, 5, cifar_Ks, 200, n_trials=50, pca_dims=50)
            
        # Save standard results
        print("Saving sweeps results...")
        save_results(all_results, results_path)
    
    # 3. PCA Ablation
    mask_38 = (y_mnist == 3) | (y_mnist == 8)
    X_38 = X_mnist[mask_38]
    y_38 = y_mnist[mask_38]
    y_38 = np.where(y_38 == 3, 0, 1)
    
    scaler = StandardScaler()
    X_38_scaled = scaler.fit_transform(X_38)
    
    pca_results_path = 'results/ablation_pca_results.json'
    ablation_pca_results = load_results(pca_results_path)
    
    if ablation_pca_results is not None:
        print("Loaded pre-computed PCA ablation results from results/ablation_pca_results.json")
    else:
        print("\nRunning PCA ablation...")
        ablation_pca_results = {}
        for d in [20, 50, 100]:
            pca = PCA(n_components=d)
            X_pca_d = pca.fit_transform(X_38_scaled)
            ablation_pca_results[str(d)] = run_experiment(
                f'MNIST_3v8_d{d}', X_pca_d, y_38, 0, 1, mnist_Ks, 200, n_trials=20, pca_dims=None
            )
        save_results(ablation_pca_results, pca_results_path)
    
    # 4. Regularization Ablation
    print("\nRunning Regularization ablation...")
    # Use MNIST 3v8 with PCA 50
    pca_50 = PCA(n_components=50)
    X_pca_50 = pca_50.fit_transform(X_38_scaled)
    
    ablation_reg_results = {}
    for C_val in [np.inf, 1.0, 0.1]:
        C_str = 'inf' if C_val == np.inf else str(C_val)
        from src.experiment import sample_and_evaluate
        accs = []
        for trial in range(50):
            acc, _ = sample_and_evaluate(X_pca_50, y_38, K=4096, test_size=200, seed=trial, C=C_val)
            accs.append(acc)
        ablation_reg_results[C_str] = {
            'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs)),
            'raw_accs': [float(a) for a in accs]
        }
        print(f"C={C_str:>6s}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}")
    save_results(ablation_reg_results, 'results/ablation_reg_results.json')
    
    # 5. Generate Figures
    print("\nGenerating figures...")
    plot_all_sweeps_grid(all_results, 'figures/all_sweeps.png')
    plot_decoupling_hypothesis(all_results, 'figures/decoupling.png')
    plot_saturation_vs_marginal_gain(all_results, 'figures/saturation_vs_marginal.png')
    plot_pca_ablation(ablation_pca_results, 'figures/pca_ablation.png')
    print("All figures saved to 'figures/' directory.")
    
if __name__ == '__main__':
    main()
