# import pipeline
import sys
sys.path.append('..')
import os
import logging
from pathlib import Path
from pipeline.utils.directory_tree import DirectoryTree
from pipeline.utils.misc import create_logger
from pipeline.utils.config import config_to_str, parse_log_level, load_config_from_json

DirectoryTree.git_repos_to_track['cgd'] = str(
    os.path.join(*([os.path.dirname(os.path.dirname(__file__))])))

# other imports
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms

from exp3_image_generation.src.GAN import GAN


def get_main_args(overwritten_cmd_line=None):
    """Defines all the command-line arguments of the main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", default="", type=str, help="Description of the experiment to be run")
    parser.add_argument("--alg_name", default="", type=str, help="Name of the algorithm", choices=["GDA", "CGD"])
    parser.add_argument("--task_name", default="", type=str, help="Name of the rl-environment or dataset", choices=["imgenMNIST"])

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, choices=['cpu', 'cuda'])
    parser.add_argument("--root", default="./storage", type=str)
    parser.add_argument("--log_level", default=logging.INFO, type=parse_log_level)

    # Algorithm hyperparameters

    parser.add_argument("--resume", default=None, type=str,
                        help="This argument is used to resume a training that had been interrupted"
                             "To use, enter the entire path to seed_dir to be resumed: --resume PATH_TO_SEED_DIR"
                             "Note: if defined, all other 'Algorithm hyperparameters' will not be used.")

    parser.add_argument("--z_size", default=100, type=int)
    parser.add_argument("--batch_size", default=128, type=float)
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--optimizer", default="sgd", type=str, choices=["sgd", "adam"])
    parser.add_argument("--n_epochs", default=100, type=int)

    return parser.parse_args(overwritten_cmd_line)


def check_main_args(config):
    assert type(config) is argparse.Namespace
    if "cuda" in config.device:
        assert torch.cuda.is_available(), f"config.device={config.device} but cuda is not available."


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config, dir_manager=None, logger=None, pbar="default_pbar"):
    # test the config for some requirements

    check_main_args(config)

    # Creates a directory manager that encapsulates our directory-tree structure

    DirectoryTree.root = Path(config.root)
    if dir_manager is None:
        dir_manager = DirectoryTree(alg_name=config.alg_name,
                                    task_name=config.task_name,
                                    desc=config.desc,
                                    seed=config.seed)
        dir_manager.create_directories()

    # Creates logger and prints config

    if logger is None:
        logger = create_logger('MASTER', config.log_level, dir_manager.seed_dir / 'logger.out')
    logger.debug(config_to_str(config))

    # Setting the random seed (for reproducibility)

    set_seeds(config.seed)

    # instantiates the model and dataset

    if config.task_name == "imgenMNIST":

        # data loader

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform),
            batch_size=config.batch_size,
            shuffle=True)

        # algorithm

        if config.resume is not None:  # initialise from checkpoint in provided dir_manager

            alg = GAN.init_from_checkpoint(train_loader=train_loader,
                                           logger=logger,
                                           dir_manager=dir_manager,
                                           device=config.device)

        else:  # initialise from provided config

            alg = GAN(z_size=config.z_size,
                      im_size=train_loader.dataset.data.shape[1],
                      lr=config.lr,
                      optimizer=config.optimizer,
                      n_epochs=config.n_epochs,
                      train_loader=train_loader,
                      logger=logger,
                      dir_manager=dir_manager,
                      device=config.device,
                      alg=config.alg_name)

    else:
        raise NotImplemented

    # Train the model

    alg.train_model()


if __name__ == '__main__':
    config = get_main_args()

    if config.resume is not None:
        seed_dir = Path(config.resume)
        config = load_config_from_json(filename=config.resume)
        dir_manager = DirectoryTree.init_from_seed_path(seed_dir)

    else:
        dir_manager = None

    main(config, dir_manager=dir_manager)
