import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set clean, publication-friendly plotting style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'savefig.bbox': 'tight',
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def plot_single_experiment(results, title, save_path=None):
    """Plot accuracy and effective rank on dual axes for a single task."""
    Ks = [r['K'] for r in results]
    accs = [r['mean_acc'] for r in results]
    std_accs = [r['std_acc'] for r in results]
    eranks = [r['mean_erank'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    
    # Left axis: Accuracy
    color = '#1f77b4'
    ax1.set_xlabel('Training samples per class (K)', fontweight='bold')
    ax1.set_ylabel('Accuracy', color=color, fontweight='bold')
    ax1.errorbar(Ks, accs, yerr=std_accs, fmt='-o', color=color, linewidth=2, elinewidth=1.5, capsize=3, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(Ks)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True)
    
    # Right axis: Effective Rank
    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Effective Rank (erank)', color=color, fontweight='bold')
    ax2.plot(Ks, eranks, '-s', color=color, linewidth=2, label='erank')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Spectral Saturation Sweep: {title}', pad=15, fontweight='bold')
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_all_sweeps_grid(all_results, save_path=None):
    """Plot accuracy and erank for all 8 standard tasks in a grid layout."""
    tasks = list(all_results.keys())
    n_tasks = len(tasks)
    
    # Grid size: 4 rows, 2 columns
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    
    for i, task_name in enumerate(tasks):
        if i >= len(axes):
            break
        ax1 = axes[i]
        results = all_results[task_name]
        Ks = [r['K'] for r in results]
        accs = [r['mean_acc'] for r in results]
        std_accs = [r['std_acc'] for r in results]
        eranks = [r['mean_erank'] for r in results]
        
        # Left axis: Accuracy
        color = '#1f77b4'
        ax1.set_xlabel('K (log scale)', fontsize=10)
        ax1.set_ylabel('Accuracy', color=color, fontsize=10)
        ax1.errorbar(Ks, accs, yerr=std_accs, fmt='-o', color=color, linewidth=1.5, elinewidth=1, capsize=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xscale('log', base=2)
        ax1.set_xticks(Ks)
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax1.grid(True)
        ax1.set_title(task_name, fontsize=11, fontweight='bold')
        
        # Right axis: erank
        ax2 = ax1.twinx()
        color = '#d62728'
        ax2.set_ylabel('Effective Rank', color=color, fontsize=10)
        ax2.plot(Ks, eranks, '-s', color=color, linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color)
        
    plt.suptitle('Spectral Saturation and Accuracy Sweeps Across All Tasks', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_decoupling_hypothesis(all_results, save_path=None):
    """Plot erank_inf vs peak accuracy to show decoupling of difficulty and complexity."""
    eranks_inf = []
    peaks = []
    task_names = []
    
    for name, results in all_results.items():
        erank_inf = results[-1]['mean_erank']
        peak_acc = max([r['mean_acc'] for r in results])
        eranks_inf.append(erank_inf)
        peaks.append(peak_acc)
        task_names.append(name)
        
    plt.figure(figsize=(7, 5.5))
    plt.scatter(eranks_inf, peaks, color='#2ca02c', s=100, edgecolors='black', zorder=3)
    
    # Annotate points
    for name, x, y in zip(task_names, eranks_inf, peaks):
        # Adjust label offset for readability
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
        
    # Fit line of best fit (optional, just to show weak correlation)
    m, b = np.polyfit(eranks_inf, peaks, 1)
    x_range = np.linspace(min(eranks_inf)-2, max(eranks_inf)+2, 100)
    plt.plot(x_range, m*x_range + b, color='gray', linestyle='--', alpha=0.7, label='Linear fit')
    
    plt.xlabel('Effective Rank Asymptote (erank_∞)', fontweight='bold')
    plt.ylabel('Peak Classification Accuracy', fontweight='bold')
    plt.title('Decoupling of Task Difficulty & Geometric Complexity', pad=15, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_saturation_vs_marginal_gain(all_results, save_path=None):
    """Plot Saturation ratio S(K) vs Marginal Accuracy Gain."""
    all_S = []
    all_marginal = []
    
    for task_name, results in all_results.items():
        for i, r in enumerate(results):
            if i == 0:
                continue
            marginal = r['marginal']
            S = r['S']
            if marginal is not None and not np.isnan(marginal):
                all_S.append(S)
                all_marginal.append(marginal)
                
    df = pd.DataFrame({'S': all_S, 'marginal': all_marginal})
    df['phase'] = pd.cut(df['S'], bins=[0, 0.3, 1.0, np.inf], 
                         labels=['Saturation', 'Transition', 'Exploration'])
    
    plt.figure(figsize=(8, 5))
    
    # Scatter plot colored by phase
    colors = {'Saturation': '#d62728', 'Transition': '#ff7f0e', 'Exploration': '#1f77b4'}
    for phase, group in df.groupby('phase', observed=False):
        plt.scatter(group['S'], group['marginal'], label=phase, color=colors[phase], alpha=0.7, edgecolors='none', s=60)
        
    plt.axvline(x=0.3, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
    
    plt.xlabel('Saturation Index S(K) = erank / K', fontweight='bold')
    plt.ylabel('Marginal Accuracy Gain', fontweight='bold')
    plt.title('Saturation Index predicts Marginal Accuracy Returns', pad=15, fontweight='bold')
    plt.legend(title='Predicted Phase')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pca_ablation(ablation_results, save_path=None):
    """Plot PCA dimensionality ablation (erank and accuracy curves for d=20, 50, 100)."""
    plt.figure(figsize=(10, 4.5))
    
    # Left subplot: Accuracy
    plt.subplot(1, 2, 1)
    colors = {'20': '#2ca02c', '50': '#ff7f0e', '100': '#1f77b4'}
    for d, results in ablation_results.items():
        Ks = [r['K'] for r in results]
        accs = [r['mean_acc'] for r in results]
        plt.plot(Ks, accs, '-o', label=f'd={d}', color=colors[d], linewidth=1.8)
    plt.xscale('log', base=2)
    plt.xlabel('K', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.title('Accuracy vs. K by PCA Dimension', fontweight='bold')
    plt.legend()
    plt.grid(True)
    
    # Right subplot: erank
    plt.subplot(1, 2, 2)
    for d, results in ablation_results.items():
        Ks = [r['K'] for r in results]
        eranks = [r['mean_erank'] for r in results]
        plt.plot(Ks, eranks, '-s', label=f'd={d}', color=colors[d], linewidth=1.8)
    plt.xscale('log', base=2)
    plt.xlabel('K', fontweight='bold')
    plt.ylabel('Effective Rank', fontweight='bold')
    plt.title('erank vs. K by PCA Dimension', fontweight='bold')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('PCA Dimensionality Ablation (MNIST 3v8)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
