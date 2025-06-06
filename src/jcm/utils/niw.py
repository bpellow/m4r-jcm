import numpy as np
from scipy.special import gammaln


def log_marginal_likelihood(d, N, kappa, nu, S_logdet):
    """
    Computes the log marginal likelihood of N d-dimensional observations under
    a Normal-Inverse-Wishart (NIW) prior on the mean and covariance of a Gaussian.
    """

    return (
        -0.5 * N * d * np.log(np.pi)
        - 0.5 * d * np.log(kappa)
        - 0.5 * nu * S_logdet
        + np.sum(gammaln(0.5 * (nu + d - np.arange(1, d + 1))))
    )