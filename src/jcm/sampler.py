from .model import ClusterModel
import numpy as np
from tqdm import tqdm


def run_gibbs(model: ClusterModel, M: int, burnin: int, sequential: bool = False, track_likelihood: bool = False, show_progress: bool = False):
    """
    Run the collasped Gibbs sampler M iterations with a burn-in period.

    Args:
        model (ClusterModel): The cluster model to sample from.
        M (int): Total number of iterations to run.
        burnin (int): Number of burn-in iterations to discard.
        sequential (bool): If True, updates clusters sequentially; otherwise, updates in random order.
        track_likelihood (bool): If True, tracks log-likelihood during sampling.
        true_labels (np.ndarray, optional): True labels for the data, used for evaluation.

    Returns:
        samples (np.ndarray): Post burn-in samples of cluster assignments.
        log_likelihoods (np.ndarray, optional): Log-likelihoods of the model at each iteration if `track_likelihood` is True.
    """
    n = model.n
    samples = np.zeros((M, n), dtype=int)

    log_likelihoods = np.zeros(M + burnin) if track_likelihood else None

    # Run the Gibbs sampler for M + burnin iterations
    iterator = tqdm(range(M + burnin), total=M + burnin, desc="Gibbs Sampling") if show_progress else range(M + burnin)
    for j in iterator:
        for i in range(n) if sequential else np.random.permutation(n):
            model.update_cluster_assignment(i)

        # Store likelihood
        if track_likelihood:
            log_likelihoods[j] = model.log_marginal_likelihood()

        # Store post-burnin samples
        if j >= burnin:
            samples[j - burnin] = model.z.copy()
    
    return samples, log_likelihoods
