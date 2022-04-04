import os
import warnings
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from dataset.dataset import get_dataset
from torch.utils.data import DataLoader


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
