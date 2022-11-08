# mipnerf-pytorch

This repository contains the code release for
[Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://jonbarron.info/mipnerf/).
This implementation is written in [pytorch](https://pytorch.org/), and
is an unofficial pytorch implement of of Google's [mipnerf implementation](https://github.com/google/mipnerf).

![rays](https://user-images.githubusercontent.com/3310961/118305131-6ce86700-b49c-11eb-99b8-adcf276e9fe9.jpg)

## Result

<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-baqh" colspan="12">Multi Scale Train And Multi Scale Test</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-c3ow" colspan="6"><span style="font-weight:400;font-style:normal">PNSR</span></td>
    <td class="tg-c3ow" colspan="6"><span style="font-weight:400;font-style:normal">SSIM</span></td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-c3ow">Full Res</td>
    <td class="tg-c3ow">1/2 Res</td>
    <td class="tg-c3ow">1/4 Res</td>
    <td class="tg-c3ow">1/8 Res</td>
    <td class="tg-c3ow">Aveage <br>(PyTorch)</td>
    <td class="tg-c3ow">Aveage <br>(Jax)</td>
    <td class="tg-0pky">Full Res</td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">1/2 Res</span></td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">1/4 Res</span></td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">1/8 Res</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">Average</span><br><span style="font-weight:400;font-style:normal">(PyTorch)</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">Average</span><br><span style="font-weight:400;font-style:normal">(Jax)</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">lego</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">34.74</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">35.93</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">36.29</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">35.62</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">35.65</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">35.74</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.9719</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.9841</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.9890</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">0.9894</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">0.9836</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">0.9843</span></td>
  </tr>
</tbody>
</table>

The above results are trained on the `lego` dataset with 750k `(Training not completed yet and 1000k steps in total)` steps for multi-scale datasets, and the pre-trained model can be found [here](https://pan.baidu.com/s/1btYFOfx-q9dj_rwvPgUw7Q) with password `psrd`.

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
