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

def plot_all_sweeps_grid(all_results, save_path=None, n_cols=3):
    """Plot accuracy and erank for all standard tasks in a dynamic grid layout."""
    tasks = list(all_results.keys())
    n_tasks = len(tasks)
    
    # Dynamically compute grid size
    n_rows = int(np.ceil(n_tasks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()
    
    for i, task_name in enumerate(tasks):
        ax1 = axes[i]
        results = all_results[task_name]
        Ks = [r['K'] for r in results]
        accs = [r['mean_acc'] for r in results]
        std_accs = [r['std_acc'] for r in results]
        eranks = [r['mean_erank'] for r in results]
        
        # Left axis: Accuracy
        color = '#1f77b4'
        ax1.set_xlabel('K (log scale)', fontsize=9)
        ax1.set_ylabel('Accuracy', color=color, fontsize=9)
        ax1.errorbar(Ks, accs, yerr=std_accs, fmt='-o', color=color, linewidth=1.5, elinewidth=1, capsize=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xscale('log', base=2)
        # Reduce x-tick density for dense grids
        tick_every = max(1, len(Ks) // 6)
        ax1.set_xticks(Ks[::tick_every])
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax1.grid(True)
        ax1.set_title(task_name, fontsize=10, fontweight='bold')
        
        # Right axis: erank
        ax2 = ax1.twinx()
        color = '#d62728'
        ax2.set_ylabel('Effective Rank', color=color, fontsize=9)
        ax2.plot(Ks, eranks, '-s', color=color, linewidth=1.5, markersize=4)
        ax2.tick_params(axis='y', labelcolor=color)

    # Hide any unused subplots
    for j in range(n_tasks, len(axes)):
        axes[j].set_visible(False)
        
    plt.suptitle('Spectral Saturation and Accuracy Sweeps Across All Tasks', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
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
    """
    Plot PCA dimensionality ablation.
    Supports two formats:
      - Flat dict: {d: [results]} — single task (legacy)
      - Nested dict: {task_name: {d: [results]}} — multi-task grid
    """
    # Detect format: flat vs nested
    first_val = next(iter(ablation_results.values()))
    is_nested = isinstance(first_val, dict) and not isinstance(next(iter(first_val.values())), dict)

    if not is_nested:
        # Legacy flat format: treat as single task named 'MNIST_3v8'
        ablation_results = {'MNIST_3v8': ablation_results}

    tasks = list(ablation_results.keys())
    n_tasks = len(tasks)
    color_map = {'5': '#9467bd', '10': '#e377c2', '20': '#2ca02c', '50': '#ff7f0e', '100': '#1f77b4'}

    fig, axes = plt.subplots(n_tasks, 2, figsize=(12, 4.5 * n_tasks), squeeze=False)

    for row, task_name in enumerate(tasks):
        task_results = ablation_results[task_name]
        dims = sorted(task_results.keys(), key=lambda x: int(x))

        ax_acc = axes[row, 0]
        ax_erank = axes[row, 1]

        for d in dims:
            results = task_results[d]
            Ks = [r['K'] for r in results]
            accs = [r['mean_acc'] for r in results]
            eranks = [r['mean_erank'] for r in results]
            clr = color_map.get(str(d), None)
            ax_acc.plot(Ks, accs, '-o', label=f'd={d}', color=clr, linewidth=1.8)
            ax_erank.plot(Ks, eranks, '-s', label=f'd={d}', color=clr, linewidth=1.8)

        ax_acc.set_xscale('log', base=2)
        ax_acc.set_xlabel('K', fontweight='bold')
        ax_acc.set_ylabel('Accuracy', fontweight='bold')
        ax_acc.set_title(f'{task_name} — Accuracy vs K', fontweight='bold')
        ax_acc.legend()
        ax_acc.grid(True)

        ax_erank.set_xscale('log', base=2)
        ax_erank.set_xlabel('K', fontweight='bold')
        ax_erank.set_ylabel('Effective Rank', fontweight='bold')
        ax_erank.set_title(f'{task_name} — erank vs K', fontweight='bold')
        ax_erank.legend()
        ax_erank.grid(True)

    plt.suptitle('Multi-Task PCA Dimensionality Ablation', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_reg_ablation(ablation_reg_results, save_path=None):
    """
    Plot Regularization ablation.
    Supports:
      - Flat dict: {C_str: {mean_acc, std_acc}} — single task (legacy)
      - Nested dict: {task_name: {C_str: {mean_acc, std_acc}}} — multi-task
    """
    # Detect format
    first_val = next(iter(ablation_reg_results.values()))
    is_nested = isinstance(first_val, dict) and 'mean_acc' not in first_val

    if not is_nested:
        ablation_reg_results = {'MNIST_3v8': ablation_reg_results}

    tasks = list(ablation_reg_results.keys())
    n_tasks = len(tasks)
    c_labels = ['inf', '1.0', '0.1']
    c_display = ['C=∞', 'C=1.0', 'C=0.1']
    colors = ['#1f77b4', '#ff7f0e', '#d62728']

    fig, axes = plt.subplots(1, n_tasks, figsize=(5.5 * n_tasks, 4.5), squeeze=False)

    for col, task_name in enumerate(tasks):
        task_res = ablation_reg_results[task_name]
        ax = axes[0, col]
        means, stds, labels = [], [], []
        for c_str, c_disp in zip(c_labels, c_display):
            if c_str in task_res:
                means.append(task_res[c_str]['mean_acc'])
                stds.append(task_res[c_str]['std_acc'])
                labels.append(c_disp)
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(labels)],
                      edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Mean Accuracy', fontweight='bold')
        ax.set_title(task_name, fontweight='bold')
        ax.set_ylim(max(0, min(means) - 0.05), min(1.0, max(means) + 0.05))
        ax.grid(axis='y', alpha=0.4)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Multi-Task Regularization Ablation (at K_peak)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_classifier_comparison(clf_results, save_path=None):
    """
    Plot bar chart comparing Logistic Regression, Nearest Centroid, and Linear SVM
    accuracies at K=4096 on MNIST 3v8.
    clf_results: {clf_name: {mean_acc, std_acc}}
    """
    names = list(clf_results.keys())
    means = [clf_results[n]['mean_acc'] for n in names]
    stds = [clf_results[n]['std_acc'] for n in names]
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors[:len(names)],
                  edgecolor='black', linewidth=0.9, alpha=0.88, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Accuracy at K=4096', fontweight='bold', fontsize=12)
    ax.set_title('Classifier-Agnostic Check: MNIST 3 vs 8 (K=4096)', fontweight='bold', fontsize=13)
    ax.set_ylim(max(0, min(means) - 0.08), min(1.0, max(means) + 0.08))
    ax.grid(axis='y', alpha=0.4)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                f'{m:.3f}±{s:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
