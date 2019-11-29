import itertools
import matplotlib.pyplot as plt
import os


def save_generated_samples(G, z_noise, path='result.png'):
    # generate image samples

    G.eval()
    test_images = G(z_noise)
    G.train()

    # create figure

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5 * 5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, :].cpu().data.view(28, 28).numpy(), cmap='gray')

    fig.text(0.5, 0.04, path.name, ha='center')

    # save figure

    os.makedirs(str(path.parent), exist_ok=True)
    plt.savefig(path)
    plt.close(fig)
