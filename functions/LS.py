import torch

dtype = torch.complex64


def Project(x, c):
    ones = torch.tensor(1, dtype=x.real.dtype, device=x.device)
    x_max = torch.maximum(torch.abs(x) / c, ones).to(dtype)
    return x / x_max


def Wxs(x):
    n = x.shape[1]
    temp_x = torch.zeros_like(x, dtype=dtype)
    temp_x[:, : n - 2] = x[:, 1: n - 1]
    return temp_x - x


def Wtxs(x):
    n = x.shape[1]
    temp_x = torch.zeros_like(x, dtype=dtype)
    temp_x[:, 1: n - 1] = x[:, : n - 2]
    res = temp_x - x
    res[:, 0] = -x[:, 0]
    res[:, n - 1] = x[:, n - 2]
    return res
