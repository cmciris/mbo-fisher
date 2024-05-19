import numpy as np
import os
import torch
import pdb
from typing import Tuple

def branin(x0, x1):
    # b,c,t = 5.1/(4.*(pi)**2), 5./pi, 1./(8.*pi)
    b, c, t = 0.12918450914398066, 1.5915494309189535, 0.039788735772973836
    u = (x1 + 5) - b * x0 ** 2 + c * x0 - 6
    r = 10. * (1. - t) * np.cos(x0) + 10
    y = u ** 2 + r
    return -y

def create_mesh(n_points=10):
    x0 = np.random.random(n_points) * 15 - 5
    x1 = np.random.random(n_points) * 15 - 5
    y = branin(x0, x1)
    return x0, x1, y

def init_data(n_points=100):
    x0, x1, y = create_mesh(n_points)
    data = {}
    data["x"] = np.concatenate([x0.reshape(-1, 1), x1.reshape(-1, 1)], axis=-1)
    data["y"] = y.reshape(-1, 1)
    return data

def main():
    data = init_data(n_points=20000)
    print(data['x'].shape, data['y'].shape)
    np.save('./x.npy', data['x'])
    np.save('./y.npy', data['y'])

if __name__ == "__main__":
    main()