import torch
from visdom import Visdom
import torchvision


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


def save_image_tensor(image: torch.tensor, height: int, width: int, save_path: str, nhwc: bool = True):
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.dim() == 3:
        image = image[None, ...]
        if nhwc:  # nhwc -> nchw
            image = image.permute(0, 3, 1, 2)
        torchvision.utils.save_image(image, save_path)
    elif image.dim() == 4:
        if nhwc:  # nhwc -> nchw
            image = image.permute(0, 3, 1, 2)
        torchvision.utils.save_image(image, save_path)
    elif image.dim() == 2:  # flatten
        assert image.shape[0] == height * width
        image = image.reshape(1, height, width, image.shape[-1])
        if nhwc:  # nhwc -> nchw
            image = image.permute(0, 3, 1, 2)
        torchvision.utils.save_image(image, save_path)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # import numpy as np
    # from PIL import Image
    # image_path = 'temp/000_d0.png'
    # with open(image_path, 'rb') as image_file:
    #     image = np.array(Image.open(image_file), dtype=np.float32) / 255.
    # print(image.shape)
    # image_tensor = torch.from_numpy(image)[None, ...]
    # print(image_tensor.shape)
    # image_tensor = image_tensor.permute(0, 3, 1, 2)
    # print(image_tensor.shape)
    # torchvision.utils.save_image(image_tensor, 'test.png')
    # image_o = image = np.array(Image.open('test.png'), dtype=np.float32) / 255.
    # print(np.sum((image_o-image)**2))
    pass






