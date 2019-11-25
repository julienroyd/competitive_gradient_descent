# import pipeline
import sys
sys.path.append('..')
import pipeline
from pipeline.utils.directory_tree import DirectoryTree
from pipeline.utils.misc import create_logger
from pipeline.utils.config import config_to_str

# other imports
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch
from torchvision import datasets, transforms

from im_gen_experiments.GAN import GAN


def get_main_args(overwritten_cmd_line=None):
    """Defines all the command-line arguments of the main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", default="", type=str, help="Description of the experiment to be run")
    parser.add_argument("--alg_name", default="", type=str, help="Name of the algorithm")
    parser.add_argument("--task_name", default="", type=str, help="Name of the rl-environment or dataset")

    parser.add_argument("--device", default="cpu", type=str, choices=['cpu', 'cuda'])

    # Image Generation Experiments

    parser.add_argument("--z_size", default=100, type=int)
    parser.add_argument("--batch_size", default=128, type=float)
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--n_epochs", default=100, type=int)

    return parser.parse_args(overwritten_cmd_line)


def check_main_args(config):
    assert type(config) is argparse.ArgumentParser


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config, dir_manager=None, logger=None, pbar="default_pbar"):

    # test the config for some requirements

    check_main_args(config)

    # Creates a directory manager that encapsulates our directory-tree structure

    if dir_manager is None:
        dir_manager = DirectoryTree(agent_alg=config.agent_alg,
                                    task_name=config.task_name,
                                    desc=config.desc,
                                    seed=config.seed)
        dir_manager.create_directories()

    # Creates logger and prints config

    if logger is None:
        logger = create_logger('MASTER', config.log_level, dir_manager.seed_dir / 'logger.out')
    logger.debug(config_to_str(config))

    # Creates a progress-bar

    if type(pbar) is str:
        if pbar == "default_pbar":
            pbar = tqdm()

    if pbar is not None:
        pbar.n = 0
        pbar.desc += f'{dir_manager.storage_dir.name}/{dir_manager.experiment_dir.name}/{dir_manager.seed_dir.name}'
        pbar.total = config.n_episodes

    # instantiates the model and dataset

    if config.task_name == "imgenMNIST":

        # data loader

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform),
            batch_size=config.batch_size,
            shuffle=True)

        # algorithm

        alg = GAN(z_size=config.z_size,
                  im_size=train_loader.dataset.data.shape[1],
                  lr=config.lr)

    # TRAINING LOOP

    for epoch in range(config.n_epochs):



if __name__ == '__main__':
    config = get_main_args()
    main(config)