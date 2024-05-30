import numpy as np
import os
import torch
import pdb
from typing import Tuple, List, Dict
import pdb


def branin(x0, x1):
    """
    :param x0: B,
    :param x1: B,
    :return f(x0,x1): B,
    minimize, three global minima
    f(x) = a(x1-bx0^2) + cx0 - r)^2 + s(1-t)cos(x1) + s
    a = 1, b = 5.1/(4pi^2), c = 5/pi, r = 6, s = 10, t = 1/(8pi)
    x0 in [-5, 10], x1 in [0, 15]
    """
    # b, c, t = 0.12918450914398066, 1.5915494309189535, 0.039788735772973836
    # pdb.set_trace()
    b, c, t = 5.1/(4. * np.pi**2), 5./np.pi, 1/(8. * np.pi)
    u = x1 - b * (x0 ** 2) + (c * x0) - 6
    r = 10. * (1. - t) * np.cos(x0) + 10.
    y = u ** 2 + r
    return -y

def branin_hoo(x0, x1):
    """
    :param x0: B,
    :param x1: B,
    :return f(x0,x1): B,
    modifications and alternate forms
    Picheny et al. (2012) on [0, 1]^2
    https://www.sfu.ca/~ssurjano/branin.html
    """
    _x0 = 15. * x0 - 5.
    _x1 = 15. * x1
    u = _x1 - (5.1 * _x0**2)/(4. * np.pi**2) + (5. * _x0)/np.pi - 6
    r = (10. - 10./(8. * np.pi)) * np.cos(_x0) - 44.81
    y = (1/51.95) * (u ** 2 + r)
    return -y

def create_uniform(x0_d, x1_d, n, alter=False):
    func = lambda x0, x1: branin_hoo(x0, x1) if alter else branin(x0, x1)
    x0 = np.random.uniform(x0_d[0], x0_d[1], n).clip(-5, 10) # B,
    x1 = np.random.uniform(x1_d[0], x1_d[1], n).clip(0, 15)
    y = func(x0, x1)

    x = np.concatenate([x0.reshape(-1, 1), x1.reshape(-1, 1)], axis=-1).reshape(-1, 2)
    y = y.reshape(-1, 1)
    return x, y

def create_multivariate_normal(x0, x1, cov, n, alter=False):
    func = lambda x0, x1: branin_hoo(x0, x1) if alter else branin(x0, x1)
    x = np.random.multivariate_normal([x0, x1], cov, n)
    x0 = x[:,0].clip(-5, 10)
    x1 = x[:,1].clip(0, 15)
    y = func(x0, x1)

    x = np.concatenate([x0.reshape(-1, 1), x1.reshape(-1, 1)], axis=-1).reshape(-1, 2)
    y = y.reshape(-1, 1)
    return x, y

def random_sample(x0_domains: List[Tuple],
                  x1_domains: List[Tuple],
                  n_samples: List[int],
                  alter=False):
    assert len(x0_domains) == len(x1_domains) == len(n_samples), "Check the shape of input"
    
    xs, ys = [], []
    for x0_d, x1_d, n in zip(x0_domains, x1_domains, n_samples):
        x, y = create_uniform(x0_d, x1_d, n, alter)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

def normal_sample(means: List[Tuple],
                  covs: List[List],
                  n_samples: List[int],
                  alter=False):
    assert len(means) == len(covs) == len(n_samples), "Check the shape of input"

    xs, ys = [], []
    for mu, cov, n in zip(means, covs, n_samples):
        x, y = create_multivariate_normal(mu[0], mu[1], cov, n, alter)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

def make(x, y, task='hard', ratio=0.5):
    keep = int(ratio * x.shape[0])
    
    # sort in ascending-y order
    ascend_indices = np.argsort(y, axis=0).squeeze()
    ascend_x, ascend_y = x[ascend_indices], y[ascend_indices]

    if task == 'easy':
        x, y = ascend_x[-keep:], ascend_y[-keep:]
    elif task == 'medium':
        fore = int(keep // 2)
        back = int(keep - fore)
        middle = int(x.shape[0] // 2)
        x, y = ascend_x[middle - fore:middle + back], ascend_y[middle - fore:middle + back]
    elif task == 'hard':
        x, y = ascend_x[:keep], ascend_y[:keep]
    else:
        raise NotImplementedError('Unknown task')
    
    return shuffle_numpy(x, y)

def shuffle_numpy(x: np.ndarray, y: np.ndarray, seed=2024) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(x.shape[0])
    random_state = np.random.get_state()
    np.random.seed(seed)
    np.random.shuffle(indices)
    np.random.set_state(random_state)

    return x[indices], y[indices]  

def main():
    os.makedirs('./datasets', exist_ok=True)

    # x, y = random_sample([(-3, 3), (0, 15), (2, 10), (-4, 0)], [(1, 7), (5, 15), (0, 4), (5, 15)], [5000, 5000, 5000, 5000])
    x0_domains, x1_domains = [(-4, 1), (2, 10)], [(0, 13), (2, 15)]
    # x, y = normal_sample([(0, 4), (5, 10), (6, 2), (-2, 10)], [[[2., 0.], [0., 2.]], [[4., 0.], [0., 9.]], [[9., 0.], [0., 1.]], [[1., 0.], [0., 9.]]], [5000, 5000, 5000, 5000])
    means, covs = [(0, 4), (5, 10)], [[[2., 0.], [0., 2.]], [[4., 0.], [0., 9.]]]
    n_samples = [6000, 6000]

    for method in ['random', 'normal']:
        if method == 'random':
            x, y = random_sample(x0_domains, x1_domains, n_samples)
        else:
            x, y = normal_sample(means, covs, n_samples)
        x, y = shuffle_numpy(x, y, seed=2024)

        # save
        print(x.shape, y.shape)
        np.save(f'./datasets/{method}_x.npy', x)
        np.save(f'./datasets/{method}_y.npy', y)

        for task in ['easy', 'medium', 'hard']:
            task_x, task_y = make(x, y, task)
            # save
            print(task_x.shape, task_y.shape)
            np.save(f'./datasets/{method}_x_{task}.npy', task_x)
            np.save(f'./datasets/{method}_y_{task}.npy', task_y)
    
    # full screen manifold
    # x0_domains, x1_domains = [(-5, 10)], [(0, 15)]
    # n_samples = [20000]
    # x, y = random_sample(x0_domains, x1_domains, n_samples)
    # print(x.shape, y.shape)
    # np.save(f'./datasets/x.npy', x)
    # np.save(f'./datasets/y.npy', y)

if __name__ == "__main__":
    main()