import torch
from einops import repeat
from .mip import sample_along_rays, integrated_pos_enc, pos_enc, volumetric_rendering, resample_along_rays
from collections import namedtuple


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)


class MLP(torch.nn.Module):
    """
    A simple MLP.
    """

    def __init__(self, net_depth: int, net_width: int, net_depth_condition: int, net_width_condition: int,
                 skip_index: int, num_rgb_channels: int, num_density_channels: int, activation: str,
                 xyz_dim: int, view_dim: int):
        super(MLP, self).__init__()
        self.skip_index: int = skip_index  # Add a skip connection to the output of every N layers.
        layers = []
        for i in range(net_depth):
            if i == 0:
                dim_in = xyz_dim
                dim_out = net_width
            elif (i - 1) % skip_index == 0 and i > 1:
                dim_in = net_width + xyz_dim
                dim_out = net_width
            else:
                dim_in = net_width
                dim_out = net_width
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            if activation == 'relu':
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            else:
                raise NotImplementedError
        self.layers = torch.nn.ModuleList(layers)
        del layers
        self.density_layer = torch.nn.Linear(net_width, num_density_channels)
        _xavier_init(self.density_layer)
        self.extra_layer = torch.nn.Linear(net_width, net_width)
        _xavier_init(self.extra_layer)
        layers = []
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = net_width + view_dim
                dim_out = net_width_condition
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            if activation == 'relu':
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            else:
                raise NotImplementedError
        self.view_layers = torch.nn.Sequential(*layers)
        del layers
        self.color_layer = torch.nn.Linear(net_width_condition, num_rgb_channels)

    def forward(self, x, view_direction=None):
        """Evaluate the MLP.

        Args:
            x: torch.Tensor(float32), [batch, num_samples, feature], points.
            view_direction: torch.Tensor(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
            raw_rgb: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_rgb_channels].
            raw_density: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_density_channels].
        """
        num_samples = x.shape[1]
        inputs = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % self.skip_index == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_density = self.density_layer(x)
        if view_direction is not None:
            # Output of the first part of MLP.
            bottleneck = self.extra_layer(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            view_direction = repeat(view_direction, 'batch feature -> batch sample feature', sample=num_samples)
            x = torch.cat([bottleneck, view_direction], dim=-1)
            # Here use 1 extra layer to align with the original nerf model.
            x = self.view_layers(x)
        raw_rgb = self.color_layer(x)
        return raw_rgb, raw_density


class MipNerf(torch.nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""
    def __init__(self, num_samples: int, num_levels: int, resample_padding: float, stop_resample_grad: bool,
                 use_viewdirs: bool, disparity: bool, ray_shape: str, min_deg_point: int, max_deg_point: int,
                 deg_view: int, density_activation: str, density_noise: float, density_bias: float,
                 rgb_activation: str, rgb_padding: float, disable_integration: bool, append_identity: bool,
                 mlp_net_depth: int, mlp_net_width: int, mlp_net_depth_condition: int, mlp_net_width_condition: int,
                 mlp_skip_index: int, mlp_num_rgb_channels: int, mlp_num_density_channels: int,
                 mlp_net_activation: str):
        super(MipNerf, self).__init__()
        self.num_levels = num_levels
        self.num_samples = num_samples
        self.disparity = disparity
        self.ray_shape = ray_shape
        self.disable_integration = disable_integration
        self.min_deg_point = min_deg_point
        self.max_deg_point = max_deg_point
        self.use_viewdirs = use_viewdirs
        self.deg_view = deg_view
        self.density_noise = density_noise
        mlp_xyz_dim = (max_deg_point - min_deg_point) * 3 * 2
        mlp_view_dim = deg_view * 3 * 2
        mlp_view_dim = mlp_view_dim + 3 if append_identity else mlp_view_dim
        self.mlp = MLP(mlp_net_depth, mlp_net_width, mlp_net_depth_condition, mlp_net_width_condition,
                       mlp_skip_index, mlp_num_rgb_channels, mlp_num_density_channels, mlp_net_activation,
                       mlp_xyz_dim, mlp_view_dim)
        if rgb_activation == 'sigmoid':
            # TODO: torch.sigmoid and torch.nn.Sigmoid?
            self.rgb_activation = torch.nn.Sigmoid()
        else:
            raise NotImplementedError
        self.rgb_padding = rgb_padding
        if density_activation == 'softplus':
            self.density_activation = torch.nn.Softplus()
        else:
            raise NotImplementedError
        self.density_bias = density_bias
        self.resample_padding = resample_padding
        self.stop_resample_grad = stop_resample_grad

    def forward(self, rays: namedtuple, randomized: bool, white_bkgd: bool):
        """The mip-NeRF Model.
        Args:
            rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
            randomized: bool, use randomized stratified sampling.
            white_bkgd: bool, if True, use white as the background (black o.w.).
        Returns:
            ret: list, [*(rgb, distance, acc)]
        """
        # Construct the MLP.
        # mlp = MLP()

        ret = []
        t_samples, weights = None, None
        for i_level in range(self.num_levels):
            # key, rng = random.split(rng)
            if i_level == 0:
                # Stratified sampling along rays
                t_samples, means_covs = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.disparity,
                    self.ray_shape,
                )
            else:
                t_samples, means_covs = resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_samples,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_resample_grad,
                    resample_padding=self.resample_padding,
                )
            if self.disable_integration:
                means_covs = (means_covs[0], torch.zeros_like(means_covs[1]))
            samples_enc = integrated_pos_enc(
                means_covs,
                self.min_deg_point,
                self.max_deg_point,
            )

            # Point attribute predictions
            if self.use_viewdirs:
                viewdirs_enc = pos_enc(
                    rays.viewdirs,
                    min_deg=0,
                    max_deg=self.deg_view,
                    append_identity=True,
                )
                raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)
            else:
                raw_rgb, raw_density = self.mlp(samples_enc)

            # Add noise to regularize the density predictions if needed.
            if randomized and (self.density_noise > 0):
                raw_density += self.density_noise * torch.randn(raw_density.shape, dtype=raw_density.dtype)

            # Volumetric rendering.
            rgb = self.rgb_activation(raw_rgb)
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_density + self.density_bias)
            comp_rgb, distance, acc, weights = volumetric_rendering(
                rgb,
                density,
                t_samples,
                rays.directions,
                white_bkgd=white_bkgd,
            )
            ret.append((comp_rgb, distance, acc))

        return ret


if __name__ == '__main__':
    # mlp = MLP(8, 256, 2, 128, 4, 3, 1, 'relu', 96, 27)
    # xyz_feature = torch.randn((2, 128, 96))
    # view_feature = torch.randn((2, 27))
    # out = mlp(xyz_feature, view_feature)
    # print(out[0].shape, out[1].shape)

    import collections
    Rays = collections.namedtuple(
        'Rays',
        ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))
    randn3 = torch.randn(64, 3)
    randn1 = torch.randn(64, 1)
    rayss = Rays(*([randn3] * 3), *([randn1] * 4))
    model = MipNerf(128, 2, 0., True, True, False, 'cone', 0, 16, 4, 'softplus', 0., -1., 'sigmoid', 0.001, False, True,
                    8, 256, 2, 128, 4, 3, 1, 'relu')
    out = model(rayss, True, True)
    print(out[0][0].shape)


