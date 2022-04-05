import os
import warnings
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from dataset.dataset import get_dataset
from torch.utils.data import DataLoader
from models.mip_nerf import MipNerf

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="lego")
def main(cfg: DictConfig):
    # Set the relevant seeds for reproducibility.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            + "the training is unlikely to finish in reasonable time."
        )
        device = "cpu"

    train_dateset = get_dataset(cfg.data.path, 'train', cfg)
    test_dataset = get_dataset(cfg.data.path, 'test', cfg)
    train_dataloder = DataLoader(train_dateset,
                                 batch_size=cfg.train.batch_size,
                                 shuffle=True,
                                 num_workers=cfg.train.num_work)
    test_dataloder = DataLoader(test_dataset,
                                batch_size=cfg.test.batch_size,
                                shuffle=False,
                                num_workers=cfg.test.num_work)

    # Initialize the Radiance Field model.
    model = MipNerf(
        num_samples=cfg.nerf.num_samples,
        num_levels=cfg.nerf.num_levels,
        resample_padding=cfg.nerf.resample_padding,
        stop_resample_grad=cfg.nerf.stop_resample_grad,
        use_viewdirs=cfg.nerf.use_viewdirs,
        disparity=cfg.nerf.disparity,
        ray_shape=cfg.nerf.ray_shape,
        min_deg_point=cfg.nerf.min_deg_point,
        max_deg_point=cfg.nerf.max_deg_point,
        deg_view=cfg.nerf.deg_view,
        density_activation=cfg.nerf.density_activation,
        density_noise=cfg.nerf.density_noise,
        density_bias=cfg.nerf.density_bias,
        rgb_activation=cfg.nerf.rgb_activation,
        rgb_padding=cfg.nerf.rgb_padding,
        disable_integration=cfg.disable_integration,
        append_identity=cfg.nerf.append_identity,
        mlp_net_depth=cfg.nerf.mlp.net_depth,
        mlp_net_width=cfg.nerf.mlp.net_width,
        mlp_net_depth_condition=cfg.nerf.mlp.net_depth_condition,
        mlp_net_width_condition=cfg.nerf.mlp.net_width_condition,
        mlp_skip_index=cfg.nerf.mlp.skip_index,
        mlp_num_rgb_channels=cfg.nerf.mlp.num_rgb_channels,
        mlp_num_density_channels=cfg.nerf.mlp.num_density_channels,
        mlp_net_activation=cfg.nerf.mlp.net_activation
    )


if __name__ == '__main__':
    main()
