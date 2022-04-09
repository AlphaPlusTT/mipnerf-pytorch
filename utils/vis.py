import torch
from typing import List
from visdom import Visdom


def visualize_nerf_outputs(
    nerf_out: dict, viz: Visdom, visdom_env: str
):
    """
    Visualizes the outputs of the `RadianceFieldRenderer`.

    Args:
        nerf_out: An output of the validation rendering pass.
        viz: A visdom connection object.
        visdom_env: The name of visdom environment for visualization.
    """

    # Show the coarse and fine renders together with the ground truth images.
    ims_full = torch.cat(
        [
            nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
            for imvar in ("rgb_coarse", "rgb_fine", "rgb_gt")
        ],
        dim=2,
    )
    viz.image(
        ims_full,
        env=visdom_env,
        win="images_full",
        opts={"title": "coarse | fine | target"},
    )








