import numpy as np

def calculate_effective_rank(cov_matrix):
    """
    Calculate the effective rank (erank) of a covariance matrix.
    erank = exp(H(p)) where p_i = lambda_i / sum(lambda_j) and H is entropy.
    """
    eigvals = np.linalg.eigvalsh(cov_matrix)
    # Clip to avoid division by zero or log of zero issues
    eigvals = np.maximum(eigvals, 1e-12)
    p = eigvals / eigvals.sum()
    erank = np.exp(-np.sum(p * np.log(p)))
    return erank

def saturation_ratio(erank, K):
    """
    Calculate the saturation index S = erank / K.
    """
    if K <= 0:
        return 0.0
    return erank / K
