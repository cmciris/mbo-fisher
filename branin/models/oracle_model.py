import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from scipy.stats import spearmanr
import pdb
import itertools
import numpy as np
import os
from models.forward_model import TanhMultiplier

torch.autograd.set_detect_anomaly(True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def BraninOracle(x):
    # b,c,t = 5.1/(4.*(pi)**2), 5./pi, 1./(8.*pi)
    b, c, t = 0.12918450914398066, 1.5915494309189535, 0.039788735772973836
    u = x[..., 1] + 5 - b * x[..., 0] ** 2 + c * x[..., 0] - 6
    r = 10. * (1. - t) * torch.cos(x[..., 0]) + 10
    y = u ** 2 + r
    return -y.reshape(-1, 1)