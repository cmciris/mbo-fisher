from models.forward_model import build_forward_model
from models.gradient_ascent import GradientAscent
from models.vae_model import SequentialVAE, VAETrainer
import torch
import torch.distributions as D

from scipy.stats import spearmanr
import numpy as np
import os
import json
import time
from utils.logger import Logger
from utils.data import StaticGraphTask, build_pipeline

import pdb

# from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    # 'task': 'HopperController-Exact-v0',
    # 'task': 'DKittyMorphology-Exact-v0',
    'task': 'TFBind8-Exact-v0',
    'task_relabel': False,
    'task_max_samples': None,
    'task_distribution': None,
    'normalize_ys': True,
    'normalize_xs': True,
    'in_latent_space': False,
}
    

def dataset(**config):
    task_name=config.get('task')
    task_relabel=config.get('task_relabel', False)
    task_max_samples=config.get('task_max_samples', None)
    task_distribution=config.get('task_distribution', None)
    normalize_ys=config.get('normalize_ys')
    normalize_xs=config.get('normalize_xs')
    in_latent_space=config.get('in_latent_space', False)

    print(task_name, task_relabel, task_max_samples, task_distribution, normalize_ys, normalize_xs)

    # create a model-based optimization task
    task = StaticGraphTask(task_name,
                           relabel=task_relabel,
                           dataset_kwargs=dict(max_samples=task_max_samples, distribution=task_distribution))

    if normalize_ys:
        task.map_normalize_y()
    if task.is_discrete and not in_latent_space:
        task.map_to_logits()
    if normalize_xs:
        task.map_normalize_x()

    # save the initial dataset statistics for safe keeping
    # x = torch.tensor(task.x, dtype=torch.float32, device=device)
    # y = torch.tensor(task.y, dtype=torch.float32, device=device)
    x = task.x
    y = task.y
    if task.is_discrete:
        x = np.reshape(x, [x.shape[0], -1])
    
    print('is_discrete:', task.is_discrete, 'x:', x.shape, 'y:', y.shape)

    return x, y


def pca_fit(data_to_fit, n_components=2):
    pca_fitted = PCA(n_components=n_components)
    pca_fitted.fit(data_to_fit)
    print(pca_fitted.explained_variance_ratio_)
    print(pca_fitted.explained_variance_)

    return pca_fitted


def pca_transform(pca_fitted: PCA, data_to_transform):
    pca_result = pca_fitted.transform(data_to_transform)
    
    return pca_result


def dist_scatter(file_name, data, n_components=2):
    if n_components == 2:
        sns.scatterplot(x=data[:, 0], y=data[:, 1], markers='o', size=1, alpha=0.6, legend=False)
        # plt.scatter(X[:, 0], X[:, 1], marker='o')
        # plt.xlabel('pca 1st component')
        # plt.ylabel('pca 2nd component')
        plt.savefig('./figs/{}'.format(file_name))
        plt.close()
        plt.cla()
    elif n_components == 3:
        fig = plt.figure()
        ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
        # ax = fig.gca(projection='3d')
        sns.scatterplot(data[:, 0], data[:, 1], data[:, 2], markers='o', size=1, alpha=0.6, legend=False, color="blue")
        plt.savefig('./figs/{}'.format(file_name))
        plt.close()
        plt.cla()
    else:
        print('Check n_components: {}'.format(n_components))


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

    def predict_log_prob(self, X):
        """
        Return log p(x)
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        log_prob = self.dist.log_prob(X)
        return log_prob.detach().cpu().numpy()

    def score(self, X):
        """
        Return dlogp(x)/dx
        :params x: Batch of input, np.ndarray or torch.Tersor, (B, D)
        :return score: Batch of score, torch.Tensor, (B, D)
        """
        X = torch.tensor(X, dtype=torch.float32, requires_grad=True, device=self.device)
        # X.clone().detach().requires_grad_(True)
        log_prob = self.dist.log_prob(X)
        log_prob.backward(torch.ones_like(log_prob))
        score = X.grad
        return score.detach().cpu().numpy()
    
    def sample(self, n_samples=1):
        """
        :params n_samples: Number of sample size
        :return : Batch of samples. torch.Tensor, (num_samples, D)
        """
        return self.dist.sample((n_samples,)).detach().cpu().numpy()
    
    
def gmm_fit_sample(data):
    """
    Naive GMM model implemented in sklearn package with EM
    """
    gmm = GMM(data, n_components=10, device=device)
    n_samples = min(data.shape[0], 10000)
    gmm_gen_samples = gmm.sample(n_samples=n_samples)
    gmm_gen_probs = np.exp(gmm.predict_log_prob(gmm_gen_samples))
    gmm_gen_scores = gmm.score(gmm_gen_samples)

    print(gmm_gen_samples[:10], gmm_gen_probs[:10], gmm_gen_scores[:10])
    print(gmm_gen_samples.shape, gmm_gen_probs.shape, gmm_gen_scores.shape)

    return gmm_gen_samples, gmm_gen_probs, gmm_gen_scores


def ebm_fit_sample(data):
    """
    A energy-based model (EBM) perform non-normalized log density estimation and directly parameterize the energy function with neural networks implemented in "Sliced score matching: A scalable approach to density and score estimation",
    use Sliced Score Matching to reduce computational expensitivity.
    https://github.com/ermongroup/ncsn
    """

    def sample():
        pass

    pass


def made_fit_sample(daata):
    """
    A gaussian mixture variant of Masked Autoencoders for Density Estimation (MADE - Made: Masked autoencoder for distribution estimation) implemented in "Masked autoregressive flow for density estimation",trained via MLE.
    https://github.com/kamenbliznashki/normalizing_flows  
    """
    def sample():
        pass

    pass


def diff_fit_sample(data):

    def sample():
        pass

    pass


if __name__ == "__main__":
    # for task in ['AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0', 'HopperController-Exact-v0', 'Superconductor-RandomForest-v0', 'TFBind8-Exact-v0', 'TFBind10-Exact-v0', 'CIFARNAS-Exact-v0']:
    for task in ['AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0', 'Superconductor-RandomForest-v0', 'TFBind8-Exact-v0', 'TFBind10-Exact-v0']:
    # for task in ['AntMorphology-Exact-v0']:
        print(task)
        config['task'] = task
        x, y = dataset(**config)

        # gmm fit and samples
        gmm_samples, _, _ = gmm_fit_sample(x)
        for n_components in [2, 3]:
        # for n_components in [3]:
            pca = pca_fit(x, n_components)
            origin = pca_transform(pca, x)
            gmm = pca_transform(pca, gmm_samples)
            
            # sns.set_theme(style="ticks")
            # sns.set_theme(style="white")
            # sns.set_theme(style="dark")
            sns.set_theme(style="whitegrid")
            # sns.set_theme(style="darkgrid")
            dist_scatter('{}_baseline_pca_{}d.jpg'.format(config.get('task'), n_components), origin, n_components)
            dist_scatter('{}_gmm_pca_{}d.jpg'.format(config.get('task'), n_components), gmm, n_components)