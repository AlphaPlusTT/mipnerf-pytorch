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
from utils.loss import calc_psnr
from visdom import Visdom
from utils.vis import visualize_nerf_outputs, save_image_tensor
import pickle
import pdb
# import warnings
# warnings.simplefilter('error')

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

    # if 'batch_type' is 'single_image', make sure the 'batch_size' is 1
    train_dateset = get_dataset(cfg.data.name, cfg.data.path, 'train', cfg.train.white_bkgd, cfg.train.batch_type)
    val_dataset = get_dataset(cfg.data.name, cfg.data.path, 'val', cfg.val.white_bkgd, cfg.val.batch_type)
    train_dataloader = DataLoader(train_dateset,
                                  batch_size=cfg.train.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.train.num_work)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.val.batch_size,
                                shuffle=True,
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
    model.to(device)

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr_init)  # TODO: With out weight decay?
    lr_scheduler = MipLRDecay(cfg.optimizer.lr_init, cfg.optimizer.lr_final, cfg.optimizer.max_steps,
                              cfg.optimizer.lr_delay_steps, cfg.optimizer.lr_delay_mult)

    # Set the model to the training mode.
    model.train()

    # Init the visualization visdom env.
    if cfg.visualization.visdom:
        viz = Visdom(
            server=cfg.visualization.visdom_server,
            port=cfg.visualization.visdom_port,
            use_incoming_socket=False,
        )
    else:
        viz = None

    # Run the main training loop.
    total_step = 0
    epoch = -1
    # stats = Stats(
    #     ["loss", "mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine", "sec/it"],
    # )
    stats = Stats(['loss', 'mse_coarse', 'mse_fine', 'psnr_coarse', 'psnr_fine'])
    checkpoint_folder = os.path.join(os.getcwd(), cfg.checkpoint.path)
    val_image_folder = os.path.join(os.getcwd(), cfg.visualization.val_image_path)
    for p in [checkpoint_folder, val_image_folder]:
        os.makedirs(p)
    # for epoch in range(cfg.optimizer.max_epochs):
    while True:  # keep running
        stats.new_epoch()
        epoch += 1
        if total_step == cfg.optimizer.max_steps:
            break
        for iteration, batch in enumerate(train_dataloader):
            batch_rays, batch_pixels = batch
            # pdb.set_trace()
            # batch_rays = batch_rays.to(device)
            # [getattr(batch_rays, name).to(device) for name in Rays_keys]
            batch_rays = Rays(*[getattr(batch_rays, name).to(device) for name in Rays_keys])
            batch_pixels = batch_pixels.to(device)

            optimizer.zero_grad()

            ret = model(batch_rays, cfg.train.randomized, cfg.train.white_bkgd)

            mask = batch_rays.lossmult
            if cfg.loss.disable_multiscale_loss:
                mask = torch.ones_like(mask)
            losses = []
            psnrs = []
            for (rgb, _, _) in ret:
                losses.append(
                    (mask * (rgb - batch_pixels[..., :3]) ** 2).sum() / mask.sum())
                psnrs.append(calc_psnr(rgb, batch_pixels[..., :3]))
            # The loss is a sum of coarse and fine MSEs
            mse_corse, mse_fine = losses
            psnr_corse, psnr_fine = psnrs
            loss = cfg.loss.coarse_loss_mult * mse_corse + mse_fine

            # Take the training step.
            loss.backward()
            optimizer.step()

            # Update stats with the current metrics.
            stats.update(
                {'loss': float(loss), 'mse_coarse': float(mse_corse), 'mse_fine': float(mse_fine),
                 'psnr_coarse': float(psnr_corse), 'psnr_fine': float(psnr_fine)},
                stat_set="train",
            )

            if iteration % cfg.train.stats_print_interval == 0:
                stats.print(stat_set="train")

            # Adjust the learning rate.
            lr_scheduler.step(optimizer, total_step)
            total_step += 1
            if total_step == cfg.optimizer.max_steps:
                break

        # Validation
        # if epoch % cfg.val.epoch_interval == 0 and epoch > 0:
        if epoch % cfg.val.epoch_interval == 0:
            chunk_size = cfg.val.chunk_size
            for image_id in range(cfg.val.sample_num):
                single_image_rays, single_image_pixels = next(iter(val_dataloader))
                _, height, width, _ = single_image_pixels.shape  # N H W C
                rgb_gt = single_image_pixels[..., :3]
                rgb_gt = rgb_gt.to(device)
                # change Rays to list: [origins, directions, viewdirs, radii, lossmult, near, far]
                single_image_rays = [getattr(single_image_rays, key) for key in Rays_keys]
                val_mask = single_image_rays[-3].to(device)
                # flatten each Rays attribute and put on device
                single_image_rays = [rays_attr.reshape(-1, rays_attr.shape[-1]).to(device) for rays_attr in single_image_rays]
                # get the amount of full rays of an image
                length = single_image_rays[0].shape[0]
                # divide each Rays attr into N groups according to chunk_size,
                # the length of the last group <= chunk_size
                single_image_rays = [[rays_attr[i:i+chunk_size] for i in range(0, length, chunk_size)] for rays_attr in single_image_rays]
                # get N, the N for each Rays attr is the same
                length = len(single_image_rays[0])
                # generate N Rays instances
                single_image_rays = [Rays(*[rays_attr[i] for rays_attr in single_image_rays]) for i in range(length)]

                # Activate eval mode of the model (lets us do a full rendering pass).
                model.eval()
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
                val_mse_corse = (val_mask * (corse_rgb - rgb_gt) ** 2).sum() / val_mask.sum()
                val_mse_fine = (val_mask * (fine_rgb - rgb_gt) ** 2).sum() / val_mask.sum()
                val_loss = cfg.loss.coarse_loss_mult * val_mse_corse + val_mse_fine
                val_psnr_corse = calc_psnr(corse_rgb, rgb_gt)
                val_psnr_fine = calc_psnr(fine_rgb, rgb_gt)
                stats.update(
                    {'loss': float(val_loss), 'mse_coarse': float(val_mse_corse), 'mse_fine': float(val_mse_fine),
                     'psnr_coarse': float(val_psnr_corse), 'psnr_fine': float(val_psnr_fine)},
                    stat_set='val',
                )
                stats.print(stat_set="val")

                if viz is not None:
                    # Plot that loss curves into visdom.
                    stats.plot_stats(
                        viz=viz,
                        visdom_env=cfg.visualization.visdom_env,
                        plot_file=None,
                    )
                    # Visualize the intermediate results.
                    val_nerf_out = {'rgb_coarse': corse_rgb, 'rgb_fine': fine_rgb, 'rgb_gt': rgb_gt}
                    visualize_nerf_outputs(
                        val_nerf_out, viz, cfg.visualization.visdom_env
                    )
                for rgb, name in zip([corse_rgb, fine_rgb, rgb_gt], ['coarse', 'fine', 'gt']):
                    save_path = os.path.join(val_image_folder, '{:d}_{:d}_{:s}.png'.format(epoch, image_id, name))
                    save_image_tensor(rgb, height, width, save_path, nhwc=True)

                # Set the model back to train mode.
                model.train()

        # Checkpoint.
        if (
                epoch % cfg.checkpoint.epoch_interval == 0
                and len(cfg.checkpoint.path) > 0
                and epoch > 0
        ):
            print(f"Storing checkpoint {cfg.checkpoint.name} to {checkpoint_folder}.")
            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stats": pickle.dumps(stats),
            }
            torch.save(data_to_store, os.path.join(checkpoint_folder, cfg.checkpoint.name))
    # Checkpoint.
    if (
            epoch % cfg.checkpoint.epoch_interval == 0
            and len(cfg.checkpoint.path) > 0
            and epoch > 0
    ):
        print('training phase over:')
        print(f"Storing final checkpoint {cfg.checkpoint.name} to {checkpoint_folder}.")
        data_to_store = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stats": pickle.dumps(stats),
        }
        torch.save(data_to_store, os.path.join(checkpoint_folder, cfg.checkpoint.name))


if __name__ == '__main__':
    main()
