import numpy as np
from scipy.stats import t
from scipy.special import gammaln
from numpy.linalg import slogdet, inv


def log_pdf(x, mu, Sigma, nu):
    """
    Compute the log-density of the multivariate Student's t-distribution.

    Parameters:
        x (np.ndarray): Point to evaluate, shape (d,)
        mu (np.ndarray): Mean vector, shape (d,)
        Sigma (np.ndarray): Scale matrix, shape (d, d)
        nu (float): Degrees of freedom

    Returns:
        float: Log-density at x
    """
    d = mu.shape[0]

    if d == 1:
        return t.logpdf(x, df=nu, loc=mu.item(), scale=np.sqrt(Sigma.item()))

    # Log determinant and inverse
    sign, logdet = slogdet(Sigma)
    if sign <= 0:
        raise ValueError("Sigma must be positive definite")

    x_centered = x - mu
    inv_Sigma = inv(Sigma)
    quad_term = x_centered @ inv_Sigma @ x_centered

    log_norm_const = gammaln(0.5 * (nu + d)) - gammaln(0.5 * nu)
    log_norm_const -= 0.5 * (d * (np.log(nu) + np.log(np.pi)) + logdet)
    log_kernel = -0.5 * (nu + d) * np.log(1 + quad_term / nu)

    return log_norm_const + log_kernel

def log_pdf_efficient(x, mu, Sigma_inv, Sigma_logdet, nu):
    """
    Efficiently compute the log-density of the multivariate Student's t-distribution,
    given the inverse and log-determinant of the scale matrix.

    Parameters:
        x (np.ndarray): Point to evaluate, shape (d,)
        mu (np.ndarray): Mean vector, shape (d,)
        Sigma_inv (np.ndarray): Inverse of scale matrix, shape (d, d)
        Sigma_logdet (float): Log-determinant of scale matrix
        nu (float): Degrees of freedom

    Returns:
        float: Log-density at x
    """
    d = mu.shape[0]

    if d == 1:
        # For univariate case, Sigma_inv = 1/var, so scale = sqrt(var)
        scale = 1.0 / np.sqrt(Sigma_inv.item())
        return t.logpdf(x, df=nu, loc=mu.item(), scale=scale)

    # Log normalization constant
    log_norm_const = gammaln(0.5 * (nu + d)) - gammaln(0.5 * nu)
    log_norm_const -= 0.5 * (d * (np.log(nu) + np.log(np.pi)) + Sigma_logdet)

    x_centered = x - mu
    quad_term = x_centered @ Sigma_inv @ x_centered

    log_kernel = -0.5 * (nu + d) * np.log(1 + quad_term)

    return log_norm_const + log_kernel