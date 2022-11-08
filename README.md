# mipnerf-pytorch

This repository contains the code release for
[Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://jonbarron.info/mipnerf/).
This implementation is written in [pytorch](https://pytorch.org/), and
is a fork of Google's [mipnerf implementation](https://github.com/google/mipnerf).

![rays](https://user-images.githubusercontent.com/3310961/118305131-6ce86700-b49c-11eb-99b8-adcf276e9fe9.jpg)

## Abstract

The rendering procedure used by neural radiance fields (NeRF) samples a scene
with a single ray per pixel and may therefore produce renderings that are
excessively blurred or aliased when training or testing images observe scene
content at different resolutions. The straightforward solution of supersampling
by rendering with multiple rays per pixel is impractical for NeRF, because
rendering each ray requires querying a multilayer perceptron hundreds of times.
Our solution, which we call "mip-NeRF" (à la "mipmap"), extends NeRF to
represent the scene at a continuously-valued scale. By efficiently rendering
anti-aliased conical frustums instead of rays, mip-NeRF reduces objectionable
aliasing artifacts and significantly improves NeRF's ability to represent
fine details, while also being 7% faster than NeRF and half the size. Compared
to NeRF, mip-NeRF reduces average error rates by 17% on the dataset presented
with NeRF and by 60% on a challenging multiscale variant of that dataset that
we present. mip-NeRF is also able to match the accuracy of a brute-force
supersampled NeRF on our multiscale dataset while being 22x faster.


## Installation
We recommend using [Anaconda](https://www.anaconda.com/products/individual) to set
up the environment. Run the following commands:

```
# Clone the repo
git clone https://github.com/AlphaPlusTT/mipnerf-pytorch.git; cd mipnerf-pytorch
# Create a conda environment, note you can use python 3.6-3.8 as one of the dependencies.
conda create --name mipnerf python=3.8.11; conda activate mipnerf
# Prepare pip
conda install pip; pip install --upgrade pip
# Install requirements
We used torch==1.9.1, torchvision==0.10.1, CUDA==10.2, hydra-core==1.1.1, visdom==0.1.8.9, matplotlib==3.5.1, einops==0.4.1 

```

## Data

Then, you'll need to download the datasets
from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please download and unzip `nerf_synthetic.zip` and `nerf_llff_data.zip`.

### Generate multiscale dataset
You can generate the multiscale dataset used in the paper by running the following command,
```
python scripts/convert_blender_data.py --blenderdir /nerf_synthetic --outdir /multiscale
```

## Running

Just modify the config file and
```
python trian.py
```

### OOM errors
You may need to reduce the batch size to avoid out of memory errors. For example the model can be run on a NVIDIA 3080 (10Gb) using the following yaml. 
```
train:
  batch_size: 1024
```

## Citation
Kudos to the authors for their amazing results:

```
@misc{barron2021mipnerf,
      title={Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields},
      author={Jonathan T. Barron and Ben Mildenhall and Matthew Tancik and Peter Hedman and Ricardo Martin-Brualla and Pratul P. Srinivasan},
      year={2021},
      eprint={2103.13415},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
Thanks to [Boyang Deng](https://boyangdeng.com/) for JaxNeRF.
Thanks to [jonbarron](https://github.com/jonbarron) for mipnerf.
Thanks to [facebookresearch](https://github.com/facebookresearch/pytorch3d) for nerf.
