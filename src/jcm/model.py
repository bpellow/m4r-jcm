import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.sparse import csr_matrix
from .utils import mvt
from numpy.linalg import inv, slogdet


class ClusterModel:
    def __init__(self, X, W, K, z_init, m_0, S_0, kappa_0=1, nu_0=1, gamma=1, eta=1, weight_X=1.0, weight_W=1.0):
        """
        Initialises the ClusterModel with data, hyperparameters, and NIW prior parameters 
        and computes the initial cluster parameters.

        Args:
            X (np.ndarray): Data embeddings of shape (n, d).
            W (np.ndarray): Abstract words as count vectors of shape (n, V).
            K (int): Number of clusters.
            V (int): Vocabulary size.
            gamma (float): Hyperparameter for Dirichlet prior on cluster assignments.
            eta (float): Hyperparameter for Dirichlet prior on text likelihood.
            m_0 (np.ndarray): Prior mean for NIW.
            kappa_0 (float): Prior scaling factor for NIW.
            nu_0 (float): Prior degrees of freedom for NIW.
            S_0 (np.ndarray): Prior scatter matrix for NIW.
            z_init (np.ndarray): Initial cluster assignments.
            weight_X (float): Weighting factor for the embedding likelihood in the log-likelihood calculation.
            weight_W (float): Weighting factor for the text likelihood in the log-likelihood calculation.
        """
        # Data
        self.X = X
        self.n, self.d = X.shape
        self.W = csr_matrix(W) if isinstance(W, np.ndarray) else W
        self.K = K
        self.V = W.shape[1]

        # Dirichlet hyperparameters
        self.gamma = gamma
        self.eta = eta

        # NIW prior parameters
        self.m_0, self.kappa_0, self.nu_0, self.S_0 = m_0, kappa_0, nu_0, S_0

        sign_det, self.S_0_det = slogdet(S_0)
        if sign_det <= 0.0:
            raise ValueError("Prior scatter matrix S_0 is not positive definite.")

        self.z = z_init.copy()
        self.N = np.array([np.sum(self.z == k) for k in range(self.K)])

        self.kappa = self.kappa_0 + self.N
        self.nu = self.nu_0 + self.N

        self.prior_sum = self.kappa_0 * self.m_0
        self.prior_outer = self.kappa_0 * np.outer(self.m_0, self.m_0)

        self.sum_x = np.zeros((self.K, self.d))
        self.mean_k = np.zeros((self.K, self.d))
        self.squared_sum_x = np.zeros((self.K, self.d, self.d))

        self.S = np.zeros((self.K, self.d, self.d))
        self.S_inv = np.zeros((self.K, self.d, self.d))
        self.S_logdet = np.zeros(self.K)

        self.c = np.zeros((self.K, self.V), dtype=int)  # Word counts in each cluster
        self.C = np.zeros(self.K, dtype=int)  # Total word count in each cluster

        for k in range(K):
            # Embedding statistics
            X_k = self.X[self.z == k]
            self.sum_x[k] = X_k.sum(axis=0)
            self.mean_k[k] = (self.sum_x[k] + self.prior_sum) / (self.N[k] + self.kappa_0)
            self.squared_sum_x[k] = X_k.T @ X_k

            self.S[k] = self.S_0 + self.squared_sum_x[k] + self.prior_outer - self.kappa[k] * np.outer(self.mean_k[k], self.mean_k[k])
            self.S_inv[k] = self.kappa[k] / (self.kappa[k] + 1.0) * inv(self.S[k])
            sign_det, self.S_logdet[k] = slogdet(self.S[k])
            if sign_det <= 0.0:
                raise ValueError("Covariance matrix is negative definite.")

            # Text statistics
            W_k = self.W[self.z == k]
            self.c[k] = W_k.sum(axis=0)
            self.C[k] = self.c[k].sum()

        # Artificial weighting
        self.weight_X = weight_X
        self.weight_W = weight_W
            
    def update_cluster_assignment(self, i: int):
        zold = self.z[i]
        x_i = self.X[i]
        x_outer = np.outer(x_i, x_i)

        self.sum_x[zold] -= x_i
        self.squared_sum_x[zold] -= x_outer
        self.N[zold] -= 1
        self.nu[zold] -= 1
        self.kappa[zold] -= 1
        self.mean_k[zold] = (self.prior_sum + self.sum_x[zold]) / self.kappa[zold]
        
        # Update scatter matrix S_k (store the old value in case we need to revert)
        S_old = self.S[zold]
        S_inv_old = self.S_inv[zold]
        S_logdet_old = self.S_logdet[zold]

        self.S[zold] = self.S_0 + self.squared_sum_x[zold] + self.prior_outer - self.kappa[zold] * np.outer(self.mean_k[zold], self.mean_k[zold])
        self.S_inv[zold] = self.kappa[zold] / (self.kappa[zold] + 1) * inv(self.S[zold])
        sign_det, self.S_logdet[zold] = slogdet(self.S[zold])
        if sign_det <= 0.0:
            raise ValueError("Covariance matrix is negative definite.")
        
        # Update text statistics
        w_i = self.W[i].toarray().flatten()  # Convert sparse matrix row to dense array
        M_i = np.sum(w_i)
        self.c[zold] -= w_i
        self.C[zold] -= M_i

        # Compute the collapsed embedding likelihood for each cluster
        llh_Xk = np.array([mvt.log_pdf_efficient(x=x_i, mu=self.mean_k[k], Sigma_inv=self.S_inv[k], Sigma_logdet=self.d * np.log((self.kappa[k] + 1) / (self.kappa[k] * self.nu[k])) + self.S_logdet[k], nu=self.nu[k]) for k in range(self.K)])

        # Compute the collapsed text likelihood for each cluster
        nonzero_idx = w_i > 0  # filter for non-zero word counts
        llh_Wk = np.array([
            np.sum(
                gammaln(self.c[k, nonzero_idx] + w_i[nonzero_idx] + self.eta / self.V) -
                gammaln(self.c[k, nonzero_idx] + self.eta / self.V)
            ) - np.sum(np.log(self.C[k] + self.eta + np.arange(M_i)))
            for k in range(self.K)
        ])

        # Compute the prior likelihood for cluster assignments
        llh_z = np.log(self.N + self.gamma / self.K)

        log_p = self.weight_X * llh_Xk + self.weight_W * llh_Wk + llh_z
        p = np.exp(log_p - logsumexp(log_p))

        znew = np.random.choice(self.K, p=p)

        # Update embedding statistics
        self.z[i] = znew
        self.N[znew] += 1
        self.nu[znew] += 1
        self.kappa[znew] += 1
        self.sum_x[znew] += x_i
        self.squared_sum_x[znew] += x_outer
        self.mean_k[znew] = (self.prior_sum + self.sum_x[znew]) / self.kappa[znew]

        # Update text statistics
        self.c[znew] += w_i
        self.C[znew] += M_i

        # Use the stored scatter matrix if the cluster hasn't changed
        if zold != znew:
            self.S[znew] = self.S_0 + self.squared_sum_x[znew] + self.prior_outer - self.kappa[znew] * np.outer(self.mean_k[znew], self.mean_k[znew])
            self.S_inv[znew] = self.kappa[znew] / (self.kappa[znew] + 1) * inv(self.S[znew])
            sign_det, self.S_logdet[znew] = slogdet(self.S[znew])
            if sign_det <= 0.0:
                raise ValueError("Covariance matrix is negative definite.")
        else:
            self.S[znew] = S_old
            self.S_inv[znew] = S_inv_old
            self.S_logdet[znew] = S_logdet_old

    def log_marginal_likelihood(self):
        """
        Compute the log-marginal likelihood of the data and cluster assignments.
        """
        eta_over_V = self.eta / self.V
        gamma_over_K = self.gamma / self.K

        llh_X, llh_W, llh_z = 0.0, 0.0, 0.0
        llh_z += gammaln(self.gamma) - gammaln(self.n + self.gamma) - self.K * gammaln(gamma_over_K)

        for k in range(self.K):
            N_k = self.N[k]
            if N_k > 0:
                # Cluster assignment
                llh_z += gammaln(N_k + gamma_over_K) - gammaln(gamma_over_K)
                
                # Embedding likelihood
                llh_X -= 0.5 * N_k * self.d * np.log(np.pi)
                llh_X += 0.5 * self.d * (np.log(self.kappa_0) - np.log(self.kappa[k]))
                llh_X += 0.5 * ((self.nu_0 + self.d - 1) * self.S_0_det - (self.nu[k] + self.d - 1) * self.S_logdet[k])
                i = np.arange(1, self.d + 1)
                llh_X += np.sum(gammaln(0.5 * (self.nu[k] + self.d - i)) - gammaln(0.5 * (self.nu_0 + self.d - i)))

                # Text likelihood
                llh_W += gammaln(self.eta) - gammaln(self.C[k] + self.eta)
                llh_W += np.sum(gammaln(self.c[k] + eta_over_V) - gammaln(eta_over_V))

        return self.weight_X * llh_X + self.weight_W * llh_W + llh_z
