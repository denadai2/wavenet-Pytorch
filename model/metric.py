import torch
import torch.nn.functional as F
import numpy as np


def rmse_loss(output, target):
    with torch.no_grad():
        error = torch.sqrt(F.mse_loss(output, target))
    return error


def haversine(output, target):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert to numpy
    output = output.data.numpy()
    target = target.data.numpy()

    # convert decimal degrees to radians
    output = np.deg2rad(output)
    target = np.deg2rad(target)

    # approximate radius of earth in km
    R = 6373.0

    s_lat = target[:, 1, :]
    e_lat = output[:, 1, :]
    s_lng = target[:, 0, :]
    e_lng = output[:, 0, :]

    d = np.sin((e_lat - s_lat) / 2) ** 2 + np.cos(s_lat) * np.cos(e_lat) * np.sin((e_lng - s_lng) / 2) ** 2

    return np.mean(2 * R * np.arcsin(np.sqrt(d)))
