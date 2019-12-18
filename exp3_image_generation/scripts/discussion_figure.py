from pathlib import Path
import torch
import matplotlib.pyplot as plt
import itertools
from collections import OrderedDict

from pipeline.utils.directory_tree import DirectoryTree
from pipeline.utils.config import load_config_from_json

from im_gen_experiments.GAN import Generator

STORAGE_PATH = Path('../storage/Ju5_f5afc59_GDA_imgenMNIST_grid_AdamLearningRates')
N_SAMPLES_PER_MODEL = 5

if __name__ == '__main__':

    # get all seed directories

    all_seeds = []
    for experiment_path in sorted([path for path in STORAGE_PATH.iterdir() if path.is_dir() and path.name.startswith('experiment')]):
        tmp = DirectoryTree.get_all_seeds(experiment_path)
        all_seeds += DirectoryTree.get_all_seeds(experiment_path)

    # creates fixed noise vectors

    fixed_z = torch.randn((N_SAMPLES_PER_MODEL, 100), requires_grad=False)

    # produces new samples for each model

    all_samples = OrderedDict()

    for seed_dir in all_seeds:

        # loads config

        config = load_config_from_json(filename=seed_dir / 'config.json')

        # instantiates the model

        G = Generator(input_size=config.z_size, output_size=28 * 28)
        G.load_state_dict(torch.load(seed_dir / 'G_params.pt'))

        # generate image samples

        G.eval()
        with torch.no_grad():
            all_samples[rf"$\eta$={config.lr:.2E}"] = G(fixed_z).numpy()

    # create figure

    texts = []
    fig, ax = plt.subplots(len(all_seeds), N_SAMPLES_PER_MODEL, figsize=(N_SAMPLES_PER_MODEL, len(all_seeds)))
    for i, key, in enumerate(all_samples.keys()):
        for j in range(N_SAMPLES_PER_MODEL):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            ax[i, j].cla()
            ax[i, j].imshow(all_samples[key][j].reshape(28, 28), cmap='gray')

            if j == 0:
                texts.append(ax[i, j].text(s=key, rotation=35, x=-50., y=40., fontsize=14))

    # save figure

    plt.savefig('GAN_sensitive_to_lr.png', bbox_extra_artists=texts, bbox_inches='tight', dpi=400)
    plt.close(fig)
