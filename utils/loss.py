import torch


def mse2psnr(mse):
    return -10.0 * torch.log10(mse)

