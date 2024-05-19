import torch
import torch.distributions as D
import numpy as np
from sklearn.mixture import GaussianMixture

import pdb


class GMM(object):
    """
    modified sklearn.mixture GaussianMixture class
    """
    def __init__(self, X, n_components=10, device="cpu") -> None:
        self.n_components = n_components
        self.device = device
        self.gmm = GaussianMixture(n_components=n_components)
        self.gmm.fit(X)  # fit via em

        self.weights = torch.tensor(self.gmm.weights_, dtype=torch.float32, device=self.device)
        self.means = torch.tensor(self.gmm.means_, dtype=torch.float32, device=self.device)
        self.covariances = torch.tensor(self.gmm.covariances_, dtype=torch.float32, device=self.device)
        self.dist = self._get_dist()
    
    def _get_dist(self):
        """
        Returns a torch.Distribution for given parameters of this distribution.
        :return :
        """
        mix = D.Categorical(self.weights)
        comp = D.Independent(D.MultivariateNormal(self.means, self.covariances), 0)
        return D.MixtureSameFamily(mix, comp)
        
    # def fit(self, X):
    #     self.gmm.fit(X)

    def sample_(self, n_samples=1):
        X, y = self.gmm.sample(n_samples)
        return X

    def predict_log_prob_(self, X):
        # self.gmm.predict_proba(X)

        """
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        log_prob_norm, log_responsibilities = self.gmm._estimate_log_prob_resp(X)

        return log_prob_norm

    def predict_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return log p(x) [bs, 1]
        """
        log_prob = self.dist.log_prob(X)
        return log_prob.clamp(-1e2, 0).unsqueeze(1)

    def score(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return dlogp(x)/dx
        :params x: Batch of input, np.ndarray or torch.Tersor, (B, D)
        :return score: Batch of score, torch.Tensor, (B, D)
        """
        X = X.clone().detach().requires_grad_(True)
        log_prob = self.dist.log_prob(X).clamp(-1e2, 0)
        log_prob.backward(torch.ones_like(log_prob))
        score = X.grad
        return score
    
    def sample(self, n_samples=1):
        """
        :params n_samples: Number of sample size
        :return : Batch of samples. torch.Tensor, (num_samples, D)
        """
        return self.dist.sample((n_samples,))