import torch
from typing import Any, Callable
from einops import rearrange, repeat


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
        # self.net_depth: int = cfg.mlp.net_depth  # The depth of the first part of MLP.
        # self.net_width: int = cfg.mlp.net_width  # The width of the first part of MLP.
        # self.net_depth_condition: int = cfg.mlp.net_depth_condition  # The depth of the second part of MLP.
        # self.net_width_condition: int = cfg.mlp.net_width_condition  # The width of the second part of MLP.
        # # if cfg.mlp.activation == 'relu':
        # #     self.net_activation: Callable[..., Any] = torch.nn.ReLU  # The activation function.
        # # else:
        # #     raise NotImplementedError
        self.skip_index: int = skip_index  # Add a skip connection to the output of every N layers.
        # self.num_rgb_channels: int = cfg.mlp.num_rgb_channels  # The number of RGB channels.
        # self.num_density_channels: int = cfg.mlp.num_density_channels  # The number of density channels.
        # self.layer = torch.nn.Linear(xyz_dim, self.net_width)
        layers = []
        for i in range(net_depth):
            if i == 0:
                dim_in = xyz_dim
                dim_out = net_width
            elif (i-1) % skip_index == 0 and i > 1:
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
        # self.view_layers = torch.nn.ModuleList(layers)
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
            raw_rgb: jnp.ndarray(float32), with a shape of
                [batch, num_samples, num_rgb_channels].
            raw_density: jnp.ndarray(float32), with a shape of
                [batch, num_samples, num_density_channels].
        """
        # feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        # x = x.reshape([-1, feature_dim])
        # x = rearrange(x, 'batch sample dim -> (batch sample) dim')
        # dense_layer = functools.partial(
        #     nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
        inputs = x
        # for i in range(net_depth):
        #     samples_enc = dense_layer(net_width)(samples_enc)
        #     samples_enc = net_activation(samples_enc)
        #     if i % skip_layer == 0 and i > 0:
        #         samples_enc = jnp.concatenate([samples_enc, inputs], axis=-1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % self.skip_index == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)

        # raw_density = dense_layer(self.num_density_channels)(x).reshape(
        #     [-1, num_samples, self.num_density_channels])
        raw_density = self.density_layer(x)
        # raw_density = rearrange(raw_density, 'batch_sample dim -> batch sample dim', sample=num_samples)

        if view_direction is not None:
            # Output of the first part of MLP.
            # bottleneck = dense_layer(self.net_width)(x)
            bottleneck = self.extra_layer(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            # condition = jnp.tile(condition[:, None, :], (1, num_samples, 1))
            view_direction = repeat(view_direction, 'batch feature -> batch sample feature', sample=num_samples)
            # Collapse the [batch, num_samples, feature] tensor to
            # [batch * num_samples, feature] so that it can be fed into nn.Dense.
            # condition = condition.reshape([-1, condition.shape[-1]])
            x = torch.cat([bottleneck, view_direction], dim=-1)
            # Here use 1 extra layer to align with the original nerf model.
            # for i in range(self.net_depth_condition):
            #     x = dense_layer(self.net_width_condition)(x)
            #     x = self.net_activation(x)
            # print(x.shape)
            x = self.view_layers(x)
            # for view_layer in self.view_layers:
            #     x = view_layer(x)
        # raw_rgb = dense_layer(self.num_rgb_channels)(x).reshape(
        #     [-1, num_samples, self.num_rgb_channels])
        raw_rgb = self.color_layer(x)
        return raw_rgb, raw_density


if __name__ == '__main__':
    mlp = MLP(8, 256, 2, 128, 4, 3, 1, 'relu', 96, 27)
    xyz_feature = torch.randn((2, 128, 96))
    view_feature = torch.randn((2, 27))
    out = mlp(xyz_feature, view_feature)
    print(out[0].shape, out[1].shape)






































