import torch
import torch.distributions as D

from scipy.stats import spearmanr
import numpy as np
import os
import json
import time
from utils.logger import Logger
from utils.data import BraninTask, build_pipeline

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
    'task_relabel': False,
    'task_max_samples': None,
    'task_distribution': None,
    'task_sample_method': 'normal',
    'task_subset': None,
    'normalize_ys': True,
    'normalize_xs': True,
    'in_latent_space': False,
}

def branin_dataset(**config):
    task_relabel=config.get('task_relabel', False)
    task_max_samples=config.get('task_max_samples', None)
    task_distribution=config.get('task_distribution', None)
    task_sample_method=config.get('task_sample_method', 'normal')
    task_subset=config.get('task_subset', None)
    normalize_ys=config.get('normalize_ys')
    normalize_xs=config.get('normalize_xs')
    in_latent_space=config.get('in_latent_space', False)

    print(task_relabel, task_max_samples, task_distribution, task_sample_method, task_subset, normalize_ys, normalize_xs)

    # create a model-based optimization task
    task = BraninTask(data_dir='data/datasets',
                      relabel=task_relabel,
                      dataset_kwargs=dict(max_samples=task_max_samples, distribution=task_distribution, sample_method=task_sample_method, subset=task_subset))

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
    
    
def gmm_fit_sample(data):
    """
    Naive GMM model implemented in sklearn package with EM
    """
    pass


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
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make task
    x, y = branin_dataset(**config)  # numpy [B, 2], [B, 1]

    pdb.set_trace()

    # gmm fit and samples
    gmm_samples, _, _ = gmm_fit_sample(x)