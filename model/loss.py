import torch.nn.functional as F
import numpy as np


def nll_loss(output, target):
    return F.nll_loss(output, target)


def MSELoss(output, target):
    assert np.array_equal(output.size(), target.size())
    return F.mse_loss(output, target)
