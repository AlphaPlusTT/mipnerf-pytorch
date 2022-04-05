import torch
from einops import rearrange
import numpy as np


def lift_gaussian(directions, t_mean, t_var, r_var, diagonal):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = directions[..., None, :] * t_mean[..., None]
    # d_mag_sq = jnp.maximum(1e-10, torch.sum(directions ** 2, dim=-1, keepdim=True))
    d_norm_denominator = torch.sum(directions ** 2, dim=-1, keepdim=True)
    # TODO: is dtype and device necessary
    # min_denominator = torch.full_like(d_norm_denominator, 1e-10, dtype=d_norm_denominator.dtype,
    #                                   device=d_norm_denominator.device)
    min_denominator = torch.full_like(d_norm_denominator, 1e-10)
    d_norm_denominator = torch.maximum(min_denominator, d_norm_denominator)
    if diagonal:
        d_outer_diag = directions ** 2
        null_outer_diag = 1 - d_outer_diag / d_norm_denominator
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = directions[..., :, None] * directions[..., None, :]
        # eye = torch.eye(directions.shape[-1], dtype=directions.dtype, device=directions.device)
        eye = torch.eye(directions.shape[-1])
        # TODO: directions / torch.sqrt(d_norm_denominator) ?
        null_outer = eye - directions[..., :, None] * (directions / d_norm_denominator)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(directions, t0, t1, base_radius, diagonal, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `directions` is normalized.
    Args:
        directions: jnp.float32 3-vector, the axis of the cone
        t0: float, the starting distance of the frustum.
        t1: float, the ending distance of the frustum.
        base_radius: float, the scale of the radius as a function of distance.
        diagonal: boolean, whether or the Gaussian will be diagonal or full-covariance.
        stable: boolean, whether or not to use the stable computation described in
        the paper (setting this to False will cause catastrophic failure).
    Returns:
        a Gaussian (mean and covariance).
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
        t_var = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) /
                                            (3 * mu ** 2 + hw ** 2) ** 2)
        r_var = base_radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 *
                                    (hw ** 4) / (3 * mu ** 2 + hw ** 2))
    else:
        t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
        r_var = base_radius ** 2 * (3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3))
        t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_var = t_mosq - t_mean ** 2
    return lift_gaussian(directions, t_mean, t_var, r_var, diagonal)


def cast_rays(t_samples, origins, directions, radii, ray_shape, diagonal=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
    Args:
        t_samples: float array, the "fencepost" distances along the ray.
        origins: float array, the ray origin coordinates.
        directions: float array, the ray direction vectors.
        radii: float array, the radii (base radii for cones) of the rays.
        ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
        diagonal: boolean, whether or not the covariance matrices should be diagonal.
    Returns:
        a tuple of arrays of means and covariances.
    """
    t0 = t_samples[..., :-1]
    t1 = t_samples[..., 1:]
    if ray_shape == 'cone':
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == 'cylinder':
        raise NotImplementedError
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii, diagonal)
    means = means + origins[..., None, :]
    return means, covs


def sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, disparity, ray_shape):
    """
    Stratified sampling along the rays.
    Args:
        origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
        directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
        radii: jnp.ndarray(float32), [batch_size, 3], ray radii.
        num_samples: int.
        near: jnp.ndarray, [batch_size, 1], near clip.
        far: jnp.ndarray, [batch_size, 1], far clip.
        randomized: bool, use randomized stratified sampling.
        disparity: bool, sampling linearly in disparity rather than depth.
        ray_shape: string, which shape ray to assume.
    Returns:
    t_samples: jnp.ndarray, [batch_size, num_samples], sampled z values.
    means: jnp.ndarray, [batch_size, num_samples, 3], sampled means.
    covs: jnp.ndarray, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    batch_size = origins.shape[0]

    t_samples = torch.linspace(0., 1., num_samples + 1)
    if disparity:
        t_samples = 1. / (1. / near * (1. - t_samples) + 1. / far * t_samples)
    else:
        # t_samples = near * (1. - t_samples) + far * t_samples
        t_samples = near + (far - near) * t_samples

    if randomized:
        mids = 0.5 * (t_samples[..., 1:] + t_samples[..., :-1])
        upper = torch.cat([mids, t_samples[..., -1:]], -1)
        lower = torch.cat([t_samples[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1)
        t_samples = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_samples to make the returned shape consistent.
        # t_samples = jnp.broadcast_to(t_samples, [batch_size, num_samples + 1])
        raise NotImplementedError
    means, covs = cast_rays(t_samples, origins, directions, radii, ray_shape)
    return t_samples, (means, covs)


def expected_sin(x, x_var):
    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    # When the variance is wide, shrink sin towards zero.
    y = torch.exp(-0.5 * x_var) * torch.sin(x)
    y_var = 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y ** 2
    y_var = torch.maximum(torch.zeros_like(y_var), y_var)
    return y, y_var


def integrated_pos_enc(means_covs, min_deg, max_deg, diagonal=True):
    """Encode `means` with sinusoids scaled by 2^[min_deg:max_deg-1].
    Args:
        means_covs: a tuple containing: means, jnp.ndarray, variables to be encoded. Should
            be in [-pi, pi]. covs, jnp.ndarray, covariance matrices.
        min_deg: int, the min degree of the encoding.
        max_deg: int, the max degree of the encoding.
        diagonal: bool, if true, expects input covariances to be diagonal (full otherwise).
    Returns:
        encoded: jnp.ndarray, encoded variables.
    """
    if diagonal:
        means, covs_diag = means_covs
        scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)])
        # shape = list(means.shape[:-1]) + [-1]
        # y = torch.reshape(means[..., None, :] * scales[:, None], shape)
        y = rearrange(means[..., None, :] * scales[:, None],
                      'batch sample scale_dim mean_dim -> batch sample (scale_dim mean_dim)')
        # y_var = torch.reshape(covs_diag[..., None, :] * scales[:, None] ** 2, shape)
        y_var = rearrange(covs_diag[..., None, :] * scales[:, None] ** 2,
                          'batch sample scale_dim cov_dim -> batch sample (scale_dim cov_dim)')
    else:
        means, x_cov = means_covs
        num_dims = means.shape[-1]
        basis = torch.cat([2 ** i * torch.eye(num_dims) for i in range(min_deg, max_deg)], 1)
        y = torch.matmul(means, basis)
        # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
        # to jax.vmap(jnp.diagonal)((basis.T @ covs) @ basis).
        y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)

    return expected_sin(torch.cat([y, y + 0.5 * torch.tensor(np.pi)], dim=-1), torch.cat([y_var] * 2, dim=-1))[0]


def pos_enc(x, min_deg, max_deg, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)])
    # xb = jnp.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    xb = rearrange(x[..., None, :] * scales[:, None],
                   'batch scale_dim x_dim -> batch (scale_dim x_dim)')
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.tensor(np.pi)], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)
    else:
        return four_feat


def volumetric_rendering(rgb, density, t_samples, dirs, white_bkgd):
    """Volumetric Rendering Function.
    Args:
        rgb: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
        density: jnp.ndarray(float32), density, [batch_size, num_samples, 1].
        t_samples: jnp.ndarray(float32), [batch_size, num_samples].
        dirs: jnp.ndarray(float32), [batch_size, 3].
        white_bkgd: bool.
    Returns:
        comp_rgb: jnp.ndarray(float32), [batch_size, 3].
        disp: jnp.ndarray(float32), [batch_size].
        acc: jnp.ndarray(float32), [batch_size].
        weights: jnp.ndarray(float32), [batch_size, num_samples]
    """
    t_mids = 0.5 * (t_samples[..., :-1] + t_samples[..., 1:])
    t_interval = t_samples[..., 1:] - t_samples[..., :-1]
    delta = t_interval * torch.linalg.norm(dirs[..., None, :], dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ],
        dim=-1))
    weights = alpha * trans

    comp_rgb = (weights[..., None] * rgb).sum(axis=-2)
    acc = weights.sum(axis=-1)
    distance = (weights * t_mids).sum(axis=-1) / acc
    distance = torch.clamp(torch.nan_to_num(distance), t_samples[:, 0], t_samples[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])
    return comp_rgb, distance, acc, weights
