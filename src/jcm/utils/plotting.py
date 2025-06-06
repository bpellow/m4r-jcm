import matplotlib.pyplot as plt

def plot_log_likelihood(log_likelihoods, burnin):
    """
    Plot the log-likelihood over iterations with burnin shown in red.

    Args:
        log_likelihoods (np.ndarray): Array of log-likelihood values.
        burnin (int): Number of burn-in iterations.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(log_likelihoods)), log_likelihoods, color='blue', label='Post Burn-in')
    plt.plot(range(burnin), log_likelihoods[:burnin], color='red', label='Burn-in')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.legend()
    plt.tight_layout()
    plt.show()