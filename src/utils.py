import torch

def KL(_mean, _logvar):
    return -0.5 * torch.sum(1 + _logvar - _mean.pow(2) - _logvar.exp())