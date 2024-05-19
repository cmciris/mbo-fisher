import os
import numpy as np
import matplotlib.pyplot as plt

def main(path, stride=5):
    os.makedirs(path, exist_ok=True)

    if os.getcwd().endswith("/visualization"):
        logs_dir = os.path.join("../logs", path)
        x = np.load("../data/x.npy")
        y = np.load("../data/y.npy")
    else:
        logs_dir = os.path.join("./logs", path)
        x = np.load("./data/x.npy")
        y = np.load("./data/y.npy")

    plt.figure(dpi=800)
    tmp = x[(y<-255).reshape(-1)]
    plt.scatter(tmp[:,0], tmp[:,1], c=[[0.3, 0, 0]], s=0.3)
    for i in range(-255, 0):
        tmp = x[((y>i) & (y<i+1)).reshape(-1)]
        r = 0.6 * (-i) ** 0.1 / 255 ** 0.1
        g = (-i) ** 0.5 / 255 ** 0.5
        b = (-i) ** 0.5 / 255 ** 0.5
        r = 1 - r
        g = 1 - g
        b = 1 - b
        plt.scatter(tmp[:,0], tmp[:,1], c=[[r, g, b]], s=0.3)
    figname = os.path.join(path, "data.png")
    plt.savefig(figname)
    plt.close()

    for root, _, files in os.walk(logs_dir):
        if root == logs_dir: break

    for file in files:
        if file.endswith("_solutions.npy"):
            plt.figure(dpi=800)
            tmp = x[(y<-255).reshape(-1)]
            plt.scatter(tmp[:,0], tmp[:,1], c=[[0.3, 0, 0]], s=0.3)
            for i in range(-255, 0):
                tmp = x[((y>i) & (y<i+1)).reshape(-1)]
                r = 0.6 * (-i) ** 0.1 / 255 ** 0.1
                g = (-i) ** 0.5 / 255 ** 0.5
                b = (-i) ** 0.5 / 255 ** 0.5
                r = 1 - r
                g = 1 - g
                b = 1 - b
                plt.scatter(tmp[:,0], tmp[:,1], c=[[r, g, b]], s=0.3)

            solutions = np.load(os.path.join(logs_dir, file))
            steps = solutions.shape[1]
            for point in range(len(solutions)):
                plt.plot(solutions[point,0:steps:stride,0], solutions[point,0:steps:stride,1], c="k", linewidth=0.5, zorder=1)
            plt.scatter(solutions[:,0:steps:stride,0], solutions[:,0:steps:stride,1], c="k", s=0.5, zorder=1)
            plt.scatter(solutions[:,0,0], solutions[:,0,1], c="b", s=0.5, zorder=2)
            plt.scatter(solutions[:,-1,0], solutions[:,-1,1], c="r", s=0.5, zorder=3)

            figname = os.path.join(path, file[:-14] + ".png")
            plt.savefig(figname)
            plt.close()

        if file.endswith("_pred_x.npy"):
            pred_x = np.load(os.path.join(logs_dir, file))
            pred_y = np.load(os.path.join(logs_dir, file[:-11] + "_pred_y.npy"))
            plt.figure(dpi=800)
            tmp = pred_x[(pred_y<-255).reshape(-1)]
            plt.scatter(tmp[:,0], tmp[:,1], c=[[0.3, 0, 0]], s=0.3)
            for i in range(-255, 0):
                tmp = pred_x[((pred_y>i) & (pred_y<i+1)).reshape(-1)]
                r = 0.6 * (-i) ** 0.1 / 255 ** 0.1
                g = (-i) ** 0.5 / 255 ** 0.5
                b = (-i) ** 0.5 / 255 ** 0.5
                r = 1 - r
                g = 1 - g
                b = 1 - b
                plt.scatter(tmp[:,0], tmp[:,1], c=[[r, g, b]], s=0.3)

            figname = os.path.join(path, "pred_" + file[:-11] + ".png")
            plt.savefig(figname)
            plt.close()


if __name__ == "__main__":
    # main(path="rim/", stride=2)
    # main(path="gradient_ascent/", stride=2)
    main(path="rim/train5k_110_0.2", stride=2)
    main(path="rim/1dim_train5k_011", stride=2)
    main(path="rim/dim_train5k_011", stride=2)
    main(path="gradient_ascent/train8k", stride=2)
    main(path="gradient_ascent/1dim_train8k", stride=2)
