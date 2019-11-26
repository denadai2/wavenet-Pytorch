import torch
import torch.nn.functional as F
import numpy as np


def rmse_loss(output, target):
    assert np.array_equal(output.size(), target.size())
    with torch.no_grad():
        error = torch.sqrt(F.mse_loss(output, target))
    return error

