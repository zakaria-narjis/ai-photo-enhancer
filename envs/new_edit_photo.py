import torch
from torch import sigmoid
import numpy as np
import torch.nn.functional as F


def sigmoid_inverses(y):
    epsilon = 10**(-3)
    y = F.relu(y-epsilon)+epsilon
    y = 1-epsilon-F.relu((1-epsilon)-y)
    return -torch.log(1/(y)-1)