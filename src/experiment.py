import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.metrics import calculate_effective_rank, saturation_ratio

def sample_and_evaluate(X, y, K, test_size=200, seed=None, C=np.inf):
    """
    Sample K examples per class for training, test_size for testing.
    Train logistic regression, return accuracy and effective rank.
    """
    rng = np.random.default_rng(seed)
    
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    
    # Safety check
    if K > len(idx_0) - test_size or K > len(idx_1) - test_size:
        raise ValueError(f"K={K} too large for available data")
    
    train_0 = rng.choice(idx_0, size=K, replace=False)
    train_1 = rng.choice(idx_1, size=K, replace=False)
    
    rem_0 = np.setdiff1d(idx_0, train_0)
    rem_1 = np.setdiff1d(idx_1, train_1)
    
    test_0 = rng.choice(rem_0, size=test_size, replace=False)
    test_1 = rng.choice(rem_1, size=test_size, replace=False)
    
    X_train = np.vstack([X[train_0], X[train_1]])
    y_train = np.array([0]*K + [1]*K)
    X_test = np.vstack([X[test_0], X[test_1]])
    y_test = np.array([0]*test_size + [1]*test_size)
    
    # Center training data
    X_train_mean = X_train.mean(axis=0)
    X_train_centered = X_train - X_train_mean
    X_test_centered = X_test - X_train_mean
    
    # Train
    clf = LogisticRegression(max_iter=5000, C=C)
    clf.fit(X_train_centered, y_train)
    acc = clf.score(X_test_centered, y_test)
    
    # Pooled within-class covariance
    cov_0 = np.cov(X_train_centered[:K], rowvar=False, bias=True)
    cov_1 = np.cov(X_train_centered[K:], rowvar=False, bias=True)
    cov_pooled = 0.5 * (cov_0 + cov_1)
    
    # Effective rank
    erank = calculate_effective_rank(cov_pooled)
    
    return acc, erank

def run_experiment(name, X, y, class_a, class_b, Ks, test_size, n_trials=50, pca_dims=50, C=np.inf):
    """
    Run full K-sweep for one dataset and one class pair.
    Returns list of dicts with K, mean_acc, std_acc, mean_erank, S.
    """
    # Filter and relabel
    mask = (y == class_a) | (y == class_b)
    X_pair = X[mask].astype(np.float64)
    y_pair = y[mask]
    y_pair = np.where(y_pair == class_a, 0, 1)
    
    # Preprocessing
    if pca_dims is not None:
        scaler = StandardScaler()
        X_pair = scaler.fit_transform(X_pair)
        pca = PCA(n_components=pca_dims)
        X_pair = pca.fit_transform(X_pair)
    else:
        scaler = StandardScaler()
        X_pair = scaler.fit_transform(X_pair)
    
    results = []
    print(f"\n{'='*60}")
    print(f"{name} | {class_a} vs {class_b} | {n_trials} trials")
    print(f"{'='*60}")
    
    for K in Ks:
        n0 = (y_pair == 0).sum()
        n1 = (y_pair == 1).sum()
        if K > min(n0, n1) - test_size:
            print(f"  K={K:4d}: SKIPPED (insufficient data: {n0}/{n1} available)")
            continue
            
        accs, eranks = [], []
        for trial in range(n_trials):
            acc, erank = sample_and_evaluate(X_pair, y_pair, K=K, test_size=test_size, seed=trial, C=C)
            accs.append(acc)
            eranks.append(erank)
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_erank = np.mean(eranks)
        S = saturation_ratio(mean_erank, K)
        
        marginal = mean_acc - results[-1]['mean_acc'] if results else 0.0
        
        results.append({
            'K': int(K), 
            'mean_acc': float(mean_acc), 
            'std_acc': float(std_acc),
            'mean_erank': float(mean_erank), 
            'S': float(S),
            'marginal': float(marginal)
        })
        print(f"  K={K:4d}: acc={mean_acc:.4f}±{std_acc:.4f}, "
              f"erank={mean_erank:.2f}, S={S:.4f}, marginal={marginal:+.4f}")
    
    return results

def save_results(results_dict, filepath):
    """Save results dictionary to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)

def load_results(filepath):
    """Load results dictionary from a JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None
