import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import pdb

sys.path.append("../")
# print(sys.path)
from data.branin import branin, branin_hoo


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

def plot_background_2d(fig_dir, is_subfigure=True):
    fig_save_path = os.path.join(fig_dir, "manifold_2d.jpg")
    x = np.linspace(-5, 10, 150)
    y = np.linspace(0, 15, 150)
    x, y = np.meshgrid(x, y)
    z = branin(x, y)

    # plt.contourf(x, y, z, 100, cmap="RdGy")
    plt.contourf(x, y, z, 200, cmap="jet")
    # plt.imshow(z, extent=[-5.0, 10.0, 0.0, 15.0], origin='lower', cmap='jet')
    # plt.colorbar()
    
    if not is_subfigure:
        plt.savefig(fig_save_path, dpi=400)
        plt.clf()
    else:
        plt.cla()

def plot_background_3d(fig_dir, is_subfigure=True):
    fig_save_path = os.path.join(fig_dir, "manifold_3d.jpg")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.linspace(-5, 10, 150)
    y = np.linspace(0, 15, 150)
    x, y = np.meshgrid(x, y)
    z = branin(x, y)
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    ax.set_zlim3d(-350, 0)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.colorbar(surf, shrink=0.6)
    
    if not is_subfigure:
        plt.savefig(fig_save_path, dpi=400)
        plt.clf()
    else:
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
    data_dir = "../data/datasets"
    fig_dir = "./figs"
    os.makedirs(fig_dir, exist_ok=True)
    
    plot_background_3d(fig_dir, is_subfigure=False)
    # plot_background_2d(fig_dir, is_subfigure=False)
    for method in ["random", "normal"]:
        for task in [None, "easy", "medium", "hard"]:
            plot(data_dir, fig_dir, method=method, task=task)

if __name__ == "__main__":
    main()