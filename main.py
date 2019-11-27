# import pipeline
import os
import logging
from pathlib import Path
import sys
sys.path.append('..')
import pipeline
from pipeline.utils.directory_tree import DirectoryTree
from pipeline.utils.misc import create_logger
from pipeline.utils.config import config_to_str, parse_log_level
DirectoryTree.git_repos_to_track['cgd'] = str(os.path.join(os.path.dirname(__file__)))

# other imports
from tqdm import tqdm
import argparse
import pickle
import imageio
import numpy as np
import torch
from torchvision import datasets, transforms
from im_gen_experiments.utils import show_result, show_train_hist

from im_gen_experiments.GAN import GAN


def get_main_args(overwritten_cmd_line=None):
    """Defines all the command-line arguments of the main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", default="", type=str, help="Description of the experiment to be run")
    parser.add_argument("--alg_name", default="", type=str, help="Name of the algorithm")
    parser.add_argument("--task_name", default="", type=str, help="Name of the rl-environment or dataset")

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, choices=['cpu', 'cuda'])
    parser.add_argument("--root", default="./storage", type=str)
    parser.add_argument("--log_level", default=logging.INFO, type=parse_log_level)

    # Image Generation Experiments

    parser.add_argument("--z_size", default=100, type=int)
    parser.add_argument("--batch_size", default=128, type=float)
    parser.add_argument("--lr", default=0.0002, type=float)
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

    # # Creates a progress-bar
    #
    # if type(pbar) is str:
    #     if pbar == "default_pbar":
    #         pbar = tqdm()
    #
    # if pbar is not None:
    #     pbar.n = 0
    #     pbar.desc += f'{dir_manager.storage_dir.name}/{dir_manager.experiment_dir.name}/{dir_manager.seed_dir.name}'
    #     pbar.total = config.n_epochs
    #
    # # Setting the random seed (for reproducibility)
    #
    # set_seeds(config.seed)

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

        gan = GAN(z_size=config.z_size,
                  im_size=train_loader.dataset.data.shape[1],
                  lr=config.lr,
                  device=config.device)

    else:
        raise NotImplemented

    # TRAINING LOOP

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []

    for epoch in range(config.n_epochs):

        D_losses = []
        G_losses = []

        for x, _ in tqdm(train_loader):

            D_loss, G_loss = gan.update_step(x)

            D_losses.append(D_loss)
            G_losses.append(G_loss)

        print(f'[{epoch + 1}/{config.n_epochs}]: '
              f'loss_d: {torch.mean(torch.FloatTensor(D_losses)):.3f}, '
              f'loss_g: {torch.mean(torch.FloatTensor(G_losses)):.3f}')

        random_results_path = dir_manager.random_results_dir / f'randomRes_epoch{epoch}.png'
        fixed_results_path = dir_manager.fixed_results_dir / f'fixedRes_epoch{epoch}.png'

        show_result(gan.G, gan.fixed_z, epoch, save=True, path=random_results_path, isFix=False)
        show_result(gan.G, gan.fixed_z, epoch, save=True, path=fixed_results_path, isFix=True)

        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    print("Training finish!... save training results")
    torch.save(gan.G.state_dict(), dir_manager.seed_dir / "G_params.pt")
    torch.save(gan.D.state_dict(), dir_manager.seed_dir / "D_params.pt")

    with open(dir_manager.recorders_dir / 'train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path=dir_manager.seed_dir / 'train_hist.png')

    images = []
    for epoch in range(config.n_epochs):
        img_name = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)


if __name__ == '__main__':
    config = get_main_args()
    main(config)
