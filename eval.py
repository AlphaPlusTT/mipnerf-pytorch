import os
import warnings
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from dataset.dataset import get_dataset
from torch.utils.data import DataLoader
from models.mip_nerf import MipNerf
from utils.stats import Stats
from utils.lr_schedule import MipLRDecay
from dataset.dataset import Rays_keys, Rays
from utils.loss import calc_psnr, calc_ssim
from visdom import Visdom
from utils.vis import visualize_nerf_outputs, save_image_tensor
import pickle
import logging
import pdb

# import warnings
# warnings.simplefilter('error')

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
log = logging.getLogger(__name__)


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

    # if 'batch_type' is 'single_image', make sure the 'batch_size' is 1
    test_dataset = get_dataset(cfg.data.name, cfg.data.path, 'test', cfg.val.white_bkgd, cfg.val.batch_type)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=cfg.val.batch_size,
                                 shuffle=False,
                                 num_workers=cfg.val.num_work)

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
        disable_integration=cfg.nerf.disable_integration,
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

    # Move the model to the relevant device.
    model.load_state_dict(torch.load('/home/zed/project/nerf/mipnerf-pytorch-swap/checkpoints/lego-ep26.pth')['model'])
    model.to(device)
    # Set the model to the eval mode.
    model.eval()

    # Init the visualization visdom env.
    if cfg.visualization.visdom:
        viz = Visdom(
            server=cfg.visualization.visdom_server,
            port=cfg.visualization.visdom_port,
            use_incoming_socket=False,
        )
    else:
        viz = None

    cur_step = 0
    total_step = len(test_dataset)
    test_image_folder = os.path.join(os.getcwd(), cfg.visualization.test_image_path)
    print('test_image_folder: ', test_image_folder)
    for p in [test_image_folder]:
        os.makedirs(p)
    chunk_size = cfg.val.chunk_size
    psnr_list = []
    ssim_list = []
    # for idx in range(800):
    for single_image_rays, single_image_pixels in test_dataloader:
        cur_step += 1
        if cur_step > 800:
            break
        print(f'Evaluating {cur_step}/{total_step}')
        # Validation
        # single_image_rays, single_image_pixels = next(iter(test_dataloader))
        _, height, width, _ = single_image_pixels.shape  # N H W C
        # pdb.set_trace()
        rgb_gt = single_image_pixels[..., :3]
        rgb_gt = rgb_gt.to(device)

        # change Rays to list: [origins, directions, viewdirs, radii, lossmult, near, far]
        single_image_rays = [getattr(single_image_rays, key) for key in Rays_keys]
        val_mask = single_image_rays[-3].to(device)
        # flatten each Rays attribute and put on device
        single_image_rays = [rays_attr.reshape(-1, rays_attr.shape[-1]).to(device) for rays_attr in
                             single_image_rays]
        # get the amount of full rays of an image
        length = single_image_rays[0].shape[0]
        # divide each Rays attr into N groups according to chunk_size,
        # the length of the last group <= chunk_size
        single_image_rays = [[rays_attr[i:i + chunk_size] for i in range(0, length, chunk_size)] for rays_attr
                             in single_image_rays]
        # get N, the N for each Rays attr is the same
        length = len(single_image_rays[0])
        # generate N Rays instances
        single_image_rays = [Rays(*[rays_attr[i] for rays_attr in single_image_rays]) for i in range(length)]

        corse_rgb, fine_rgb = [], []
        with torch.no_grad():
            for batch_rays in single_image_rays:
                (c_rgb, _, _), (f_rgb, _, _) = model(batch_rays, cfg.val.randomized, cfg.val.white_bkgd)
                corse_rgb.append(c_rgb)
                fine_rgb.append(f_rgb)
        corse_rgb = torch.cat(corse_rgb, dim=0)
        fine_rgb = torch.cat(fine_rgb, dim=0)
        corse_rgb = corse_rgb.reshape(1, height, width, corse_rgb.shape[-1])  # N H W C
        fine_rgb = fine_rgb.reshape(1, height, width, fine_rgb.shape[-1])  # N H W C
        mse_corse = (val_mask * (corse_rgb - rgb_gt) ** 2).sum() / val_mask.sum()
        mse_fine = (val_mask * (fine_rgb - rgb_gt) ** 2).sum() / val_mask.sum()

        psnr_corse = calc_psnr(corse_rgb, rgb_gt)
        psnr_fine = calc_psnr(fine_rgb, rgb_gt)
        ssim_fine = calc_ssim(fine_rgb.permute(0, 3, 1, 2), rgb_gt.permute(0, 3, 1, 2))
        psnr_list.append(psnr_fine.cpu())
        ssim_list.append(ssim_fine[0].cpu())
        for rgb, name in zip([corse_rgb, fine_rgb, rgb_gt], ['coarse', 'fine', 'gt']):
            save_path = os.path.join(test_image_folder, '{:d}_{:s}.png'.format(cur_step, name))
            save_image_tensor(rgb, height, width, save_path, nhwc=True)

    psnr_multi_scale = [np.mean(np.array(p)) for p in [psnr_list[i::4] for i in range(4)]]
    psnr_d0, psnr_d1, psnr_d2, psnr_d3 = psnr_multi_scale
    average = np.mean(np.array(psnr_multi_scale))
    print(psnr_d0, psnr_d1, psnr_d2, psnr_d3, average)

    # pdb.set_trace()
    ssim_multi_scale = [np.mean(np.array(p)) for p in [ssim_list[i::4] for i in range(4)]]
    ssim_d0, ssim_d1, ssim_d2, ssim_d3 = ssim_multi_scale
    average = np.mean(np.array(ssim_multi_scale))
    print(ssim_d0, ssim_d1, ssim_d2, ssim_d3, average)


if __name__ == '__main__':
    main()
