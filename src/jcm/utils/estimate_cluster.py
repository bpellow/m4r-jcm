import numpy as np
from sklearn.cluster import AgglomerativeClustering


def get_psm(samples):
    """Compute the posterior similarity matrix from posterior samples."""
    M, n = samples.shape
    psm = np.zeros((n, n))
    for sample in samples:
        psm += np.equal.outer(sample, sample)

    return psm / M


def estimate_cluster(samples, K):
    """Estimate clusters using the posterior similarity matrix for a fixed number of clusters K."""
    psm = get_psm(samples)
    cluster_model = AgglomerativeClustering(
        n_clusters=K, metric='precomputed', linkage='average')
    return cluster_model.fit_predict(1 - psm)
