import torch
import numpy as np

from math import sqrt


def add_gaussian_noise(kdata, mask, mean_dim, noise_var=0.05, seed=0):
    """
    add gaussian noise with chosen variance to data.
    (can also be image )

    N.B. z = x + i*y in C^N ~ N(0,sigma**2 * Id )  is equivalent to
                x ~ N(0, sigma**2 / 2 * Id) and y ~ N(0, sigma**2 / 2 * Id)
    """

    if mask is not None:
        assert (
            kdata.shape == mask.shape
        ), "k-space data and mask must have the same shape."

    def my_mean(x, mask, dim):
        if mask is not None:
            return torch.sum(x, dim=dim, keepdim=True) / torch.sum(
                mask, dim=dim, keepdim=True
            )
        else:
            return torch.mean(x, dim=dim, keepdim=True)

    def my_std(x, mask, dim):
        if mask is not None:
            return torch.sqrt(
                torch.sum((x - my_mean(x, mask, dim)) ** 2, dim=dim, keepdim=True)
                / torch.sum(mask, dim=dim, keepdim=True)
            )
        else:
            return torch.std(x, dim=dim, keepdim=True)

    # compute mean and std
    mu_r, std_r = my_mean(kdata.real, mask, dim=mean_dim), my_std(
        kdata.real, mask, dim=mean_dim
    )
    mu_i, std_i = my_mean(kdata.imag, mask, dim=mean_dim), my_std(
        kdata.imag, mask, dim=mean_dim
    )

    # center k-space data
    kdata_r = (kdata.real - mu_r) / std_r
    kdata_i = (kdata.imag - mu_i) / std_i

    torch.manual_seed(seed)
    np.random.seed(seed)

    noise_r = torch.randn_like(kdata_r)
    noise_i = torch.randn_like(kdata_i)
    noise = noise_r + 1j * noise_i

    kdata_r = kdata_r + sqrt(noise_var / 2) * noise_r
    kdata_i = kdata_i + sqrt(noise_var / 2) * noise_i

    kdata = (mu_r + std_r * kdata_r) + 1j * (mu_i + std_i * kdata_i)

    if mask is not None:
        kdata = kdata * mask

    return kdata, noise
