import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import pdb

sys.path.append("../")
# print(sys.path)
from utils.branin import branin, branin_hoo


def load_numpy(load_path: str):
    data = np.load(load_path)
    return data

def kdeplot(x: np.ndarray, save_path):
    fig = sns.kdeplot(x=x[:,0], y=x[:,1], cmap='Blues', shade=True, thresh=0, levels=200)
    fig.set_xlim(-5.1, 10.1)
    fig.set_ylim(-0.1, 15.1)
    kde_fig = fig.get_figure()
    kde_fig.savefig(save_path, dpi=400)
    fig.cla()

def scatterplot(x: np.ndarray, save_path):
    fig = sns.scatterplot(x=x[:,0], y=x[:,1], linewidth=0, s=1, alpha=0.6)
    fig.set_xlim(-5.1, 10.1)
    fig.set_ylim(-0.1, 15.1)
    scatter_fig = fig.get_figure()
    scatter_fig.savefig(save_path, dpi=400)
    fig.cla()

def plot_background(fig_dir, is_subfigure=True):
    fig_save_path = os.path.join(fig_dir, "manifold.jpg")
    x = np.linspace(-5, 10, 150)
    y = np.linspace(0, 15, 150)
    # x = np.linspace(0, 1, 150)
    # y = np.linspace(0, 1, 150)
    x, y = np.meshgrid(x, y)
    z = branin(x, y)

    # plt.contourf(x, y, z, 100, cmap="RdGy")
    plt.contourf(x, y, z, 200, cmap="rainbow")
    # plt.imshow(z, extent=[-5.0, 10.0, 0.0, 15.0], origin='lower', cmap='rainbow')
    # plt.colorbar()
    
    if not is_subfigure:
        plt.savefig(fig_save_path, dpi=400)
        plt.cla()

def plot(data_dir, fig_dir, show='scatter' , method="random", task=None):
    if task:
        x_load_path = os.path.join(data_dir, f"{method}_x_{task}.npy")
        y_load_path = os.path.join(data_dir, f"{method}_y_{task}.npy")
    else:
        x_load_path = os.path.join(data_dir, f"{method}_x.npy")
        y_load_path = os.path.join(data_dir, f"{method}_y.npy")
    x, y = load_numpy(x_load_path), load_numpy(y_load_path)
    if task:
        fig_save_path = os.path.join(fig_dir, f"{method}_x_{task}_{show}.jpg")
    else:
        fig_save_path = os.path.join(fig_dir, f"{method}_x_{show}.jpg")
    
    if show == 'scatter':
        scatterplot(x, fig_save_path)
    elif show == 'kde':
        kdeplot(x, fig_save_path)
    else:
        raise NotImplementedError("Unknown visualization method")

def main():
    data_dir = "../utils/datasets"
    fig_dir = "./figs"
    os.makedirs(fig_dir, exist_ok=True)
    
    plot_background(fig_dir, is_subfigure=False)
    for method in ["random", "normal"]:
        for task in [None, "easy", "medium", "hard"]:
            plot(data_dir, fig_dir, method, task)

if __name__ == "__main__":
    main()