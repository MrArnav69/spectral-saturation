import numpy as np

def calculate_effective_rank(cov_matrix):
    eigvals = np.linalg.eigvalsh(cov_matrix)
    eigvals = np.maximum(eigvals, 1e-12)
    p = eigvals / eigvals.sum()
    erank = np.exp(-np.sum(p * np.log(p)))
    return erank

def saturation_ratio(erank, K):
    if K <= 0:
        return 0.0
    return erank / K
