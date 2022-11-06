import torch
import torch.nn.functional as F


def calc_mse(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    return torch.mean((x - y) ** 2)


def calc_psnr(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = calc_mse(x, y)
    psnr = -10.0 * torch.log10(mse)
    return psnr


def calc_ssim(img0,
              img1,
              max_val=1.0,
              filter_size=11,
              filter_sigma=1.5,
              k1=0.01,
              k2=0.03,
              return_map=False):
    """Computes SSIM from two images.

This function was modeled after tf.image.ssim, and should produce comparable output.

Args:
  img0: array. An image of size [..., num_channels, height, width].
  img1: array. An image of size [..., num_channels, height, width].
  max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
  filter_size: int >= 1. Window size.
  filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
  k1: float > 0. One of the SSIM dampening parameters.
  k2: float > 0. One of the SSIM dampening parameters.
  return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned

Returns:
  Each image's mean SSIM, or a tensor of individual values if `return_map`.
"""
    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=img0.device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)
    # print(filt.shape)
    filt = torch.matmul(filt[:, None], filt[None, :])
    # print(filt.shape)
    filt = filt[None, None, ...].repeat(3, 1, 1, 1)
    # print(filt.shape)
    batch, channel, height, width = img0.shape

    # print(img0.dtype, filt.dtype)
    mu0 = F.conv2d(img0, filt, padding=0, groups=channel)
    mu1 = F.conv2d(img1, filt, padding=0, groups=channel)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = F.conv2d(img0 ** 2, filt, padding=0, groups=channel) - mu00
    sigma11 = F.conv2d(img1 ** 2, filt, padding=0, groups=channel) - mu11
    sigma01 = F.conv2d(img0 * img1, filt, padding=0, groups=channel) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.maximum(torch.zeros_like(sigma00), sigma00)
    sigma11 = torch.maximum(torch.zeros_like(sigma11), sigma11)
    sigma01 = torch.sign(sigma01) * torch.minimum(torch.sqrt(sigma00 * sigma11), torch.abs(sigma01))

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    num_dims = len(img0.shape)
    ssim = torch.mean(ssim_map, list(range(num_dims - 3, num_dims)))
    return ssim_map if return_map else ssim
