import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
import torch.autograd as autograd
import torch.optim
import os
import pdb


class GMM(nn.Module):
    def __init__(self, input_shape, n_components):
        super().__init__()
        self.n_components = n_components
        self.dim = np.prod(input_shape)
        self.mean = torch.randn(self.n_components, self.dim)
        self.mean = nn.Parameter(self.mean)
        self.log_std = nn.Parameter(torch.randn(self.n_components, self.dim))
        self.mix_logits = nn.Parameter(torch.randn(self.n_components))

        self._mean = None
        self._std = None

    def forward(self, X: torch.Tensor):
        energy = (X.unsqueeze(1) - self.mean) ** 2 / (2 * (2 * self.log_std).exp()) + np.log(2 * np.pi) / 2. + self.log_std
        log_prob = -energy.sum(dim=-1)
        mix_probs = F.log_softmax(self.mix_logits)
        log_prob += mix_probs
        log_prob = torch.logsumexp(log_prob, dim=-1)
        pdb.set_trace()
        return log_prob  # TODO check dim [B,]
    
    def log_prob(self, x: torch.Tensor):
        """
        :param x: [B, D]
        :return logp: [B, 1]
        """
        return self.forward(x).unsqueeze(1)
    
    def scaled_log_prob(self, x: torch.Tensor):
        """
        :param x: [B, D]
        :return scaled logp: [B, 1]
        """
        assert (self._mean and self._std), "Run prepare first!"
        logp = self.log_prob(x)
        scaled_logp = (logp - self._mean) / self._std
        return scaled_logp
    
    def score(self, X: torch.Tensor):
        X.requires_grad_(True)
        logp = self.log_prob(X)
        grad = autograd.grad(logp.sum(), X)[0]
        return grad # [B, D]
    
    def scaled_score(self, X: torch.Tensor):
        X.requires_grad_(True)
        logp = self.scaled_log_prob(X)
        grad = autograd.grad(logp.sum(), X)[0]
        return grad # [B, D]

    def sample(self, n_samples):
        mix_idx = torch.multinomial(F.log_softmax(self.mix_logits), n_samples, replacement=True)
        means = self.mean[mix_idx]
        log_stds = self.log_std[mix_idx]
        return torch.randn_like(means) * torch.exp(log_stds) + means
    
    def prepare(self, X: torch.Tensor):
        """
        Initialize the mean and std, prepare for scaling log_prob
        """
        n_samples = min(X.shape[0], 4096)
        random_state = np.random.get_state()
        np.random.seed(2024)
        n_indices = np.random.choice(np.arange(X.shape[0]), size=n_samples, replace=False)
        np.random.set_state(random_state)

        X = X[n_indices]
        self._mean = self.log_prob(X).mean()
        self._std = self.log_prob(X).std()


class GMMTrainer():
    def __init__(self,
                 gmm: torch.nn.Module,
                 optim=torch.optim.Adam,
                 lr=1e-4,
                 weight_decay=0.000,
                 model_load=False,
                 model_dir="checkpoints/fisher") -> None:
        self.gmm = gmm
        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_load = model_load
        self.model_dir = model_dir

    def launch(self,
               train_data,
               validate_data,
               logger,
               epochs,
               snapshot_freq=None):
        
        test_loader = validate_data
        test_iter = iter(test_loader)

        gmm = self.gmm
        optimizer = self.optim(gmm.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.model_load:
            states = torch.load(os.path.join(self.model_dir, 'gmm_checkpoint.pth'))
            gmm.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        for epoch in range(epochs):
            for i, (X, y) in enumerate(train_data):
                step += 1

                nll = -gmm(X).sum()
                optimizer.zero_grad()
                nll.backward()
                optimizer.step()
                logger.logger.info("epoch: {}, step: {}, total_step: {}, loss: {}".format(epoch, i, step, nll.item()))
                # print("epoch: {}, step: {}, total_step: {}, loss: {}".format(epoch, i, step, nll.item()))

                if step % 100 == 0:
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)
                    # test_X has been to device

                    test_nll = -gmm(test_X).sum()

                    logger.logger.info("epoch: {}, step: {}, total_step: {}, [test_loss]: {}".format(epoch, i, step, test_nll.item()))
                    # print("epoch: {}, step: {}, total_step: {}, test_loss: {}".format(epoch, i, step, test_nll.item()))

                if snapshot_freq and step % snapshot_freq == 0:
                    states = [
                        gmm.state_dict(),
                        optimizer.state_dict()
                    ]
                    torch.save(states, os.path.join(self.model_dir, 'gmm_checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.model_dir, 'gmm_checkpoint.pth'))
    
