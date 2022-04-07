from torch.utils.data import Dataset
import os
import json
import numpy as np
from PIL import Image
import collections

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


class Blender(Dataset):
    def __init__(self):
        super(Blender, self).__init__()
        raise NotImplementedError

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class LLFF(Dataset):
    def __init__(self):
        super(LLFF, self).__init__()
        raise NotImplementedError

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class MultiCamera(Dataset):
    def __init__(self, data_dir, split, white_bkgd=True, batch_type='all_images'):
        super(MultiCamera, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.white_bkgd = white_bkgd
        self.batch_type = batch_type
        if not self.check_cache():
            self._load_renderings()
            self._generate_rays()
            self.images = self._flatten(self.images)
            self.rays = namedtuple_map(self._flatten, self.rays)
            self.cache_data()

    def check_cache(self):
        if self.white_bkgd:
            bkgd = 'white'
        else:
            bkgd = 'black'
        image_cache_name = '_'.join(['images', self.split, bkgd, self.batch_type]) + '.npy'
        rays_cache_name = '_'.join(['rays', self.split, bkgd, self.batch_type]) + '.npy'
        image_cache_path = os.path.join(self.data_dir, image_cache_name)
        rays_cache_path = os.path.join(self.data_dir, rays_cache_name)
        if os.path.exists(image_cache_path) and os.path.exists(rays_cache_path):
            self.images = np.load(image_cache_path)
            self.rays = np.load(rays_cache_path)
            return True
        else:
            return False

    def cache_data(self):
        if self.white_bkgd:
            bkgd = 'white'
        else:
            bkgd = 'black'
        image_cache_name = '_'.join(['images', self.split, bkgd, self.batch_type])
        rays_cache_name = '_'.join(['rays', self.split, bkgd, self.batch_type])
        np.save(os.path.join(self.data_dir, image_cache_name), self.images)
        np.save(os.path.join(self.data_dir, rays_cache_name), self.rays)

    def _load_renderings(self):
        """Load images from disk."""
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as fp:
            self.meta = json.load(fp)[self.split]
        self.meta = {k: np.array(self.meta[k]) for k in self.meta}
        # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
        images = []
        for relative_path in self.meta['file_path']:
            image_path = os.path.join(self.data_dir, relative_path)
            with open(image_path, 'rb') as image_file:
                image = np.array(Image.open(image_file), dtype=np.float32) / 255.
            if self.white_bkgd:
                # image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
                # pixels with alpha between 0 and 1 has a weird color!
                mask = np.where(image[..., -1] > 1e-6, 1., 0.)[..., None]
                image = image[..., :3] * mask + (1. - mask)
            images.append(image[..., :3])
        self.images = images
        # self.n_examples = len(self.images)

    def _generate_rays(self):
        """Generating rays for all images."""
        pix2cam = self.meta['pix2cam']
        cam2world = self.meta['cam2world']
        width = self.meta['width']
        height = self.meta['height']

        def res2grid(w, h):
            return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
                np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
                np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
                indexing='xy')

        xy = [res2grid(w, h) for w, h in zip(width, height)]
        pixel_dirs = [np.stack([x, y, np.ones_like(x)], axis=-1) for x, y in xy]
        camera_dirs = [v @ p2c[:3, :3].T for v, p2c in zip(pixel_dirs, pix2cam)]
        directions = [v @ c2w[:3, :3].T for v, c2w in zip(camera_dirs, cam2world)]
        origins = [
            np.broadcast_to(c2w[:3, -1], v.shape)
            for v, c2w in zip(directions, cam2world)
        ]
        viewdirs = [
            v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
        ]

        def broadcast_scalar_attribute(x):
            return [
                np.broadcast_to(x[i], origins[i][..., :1].shape)
                for i in range(len(self.images))
            ]

        lossmult = broadcast_scalar_attribute(self.meta['lossmult'])
        near = broadcast_scalar_attribute(self.meta['near'])
        far = broadcast_scalar_attribute(self.meta['far'])

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = [
            np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
        ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=lossmult,
            near=near,
            far=far)

    def _flatten(self, x):
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        if self.batch_type == 'all_images':
            # If global batching, also concatenate all data into one list
            x = np.concatenate(x, axis=0)
        return x

    def __len__(self):
        if self.batch_type == 'all_images':
            return self.images.shape[0]
        elif self.batch_type == 'single_image':
            return len(self.images)
        else:
            raise NotImplementedError(f'{self.batch_type} batching strategy is not implemented.')

    def __getitem__(self, index):
        if self.batch_type == 'all_images':
            return Rays(*[self.rays[i][index] for i in range(len(self.rays))]), self.images[index]
        elif self.batch_type == 'single_image':
            raise NotImplementedError
        else:
            raise NotImplementedError(f'{self.batch_type} batching strategy is not implemented.')


def get_dataset(data_dir, split, cfg):
    return dataset_dict[cfg.data.name](data_dir, split, cfg.data.white_bkgd, cfg.data.batch_type)


dataset_dict = {
    'blender': Blender,
    'llff': LLFF,
    'multicam': MultiCamera,
}
