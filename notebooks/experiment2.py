import matplotlib.pyplot as plt
import numpy as np
import itertools
from notebooks.experiments12_common import *


# Experiment 2

def f(x, y, alpha):
    return alpha * (x ** 2 - y ** 2)


def g(*f_args):
    return - f(*f_args)


x_init = 0.5
y_init = 0.5

alphas = [1., 3., 6.]
learning_rates = [0.2]
experiments = list(itertools.product(alphas, learning_rates))

n_updates = 100

alg_names = ['GDA', 'LCGD', 'CGD', 'OGDA', 'ConOPT']
alg_updates = [GDA_step, LCGD_step, CGD_step, OGDA_step, ConOPT_step]
colors = ['cyan', 'cyan', 'cyan', 'green', 'orange']  # paper's color-mapping
# colors = ['cyan', 'cyan', 'cyan', 'orange', 'green']  # more logical color-map for which SGA/ConOpt=Green, OGDA/LCGD=Orange

recorder = run_experiment(alg_names, alg_updates, experiments, x_init, y_init, f, g, n_updates)

# Plotting

fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
for i, (alpha, lr) in enumerate(experiments):

    for alg_name, color in zip(alg_names, colors):
        x_s, y_s = np.array(recorder[alg_name][f'alpha={alpha:.1f}, lr={lr:.2f}']).T
        axes[i].plot(np.log10(np.linalg.norm(np.array([x_s, y_s]), axis=0)), color=color, linewidth=1, label=alg_name)

    axes[i].set_title(rf'$\alpha$={alpha:.1f}', fontsize=18)
    axes[i].set_xlabel('x', fontsize=18)

    if i == 0:
        axes[i].set_ylabel('y', fontsize=18)

    axes[i].tick_params(axis='both', which='major', labelsize=12)
    axes[i].locator_params('x', nbins=3)
    axes[i].locator_params('y', nbins=5)
    axes[i].legend(loc='upper right')

plt.show()
fig.savefig(f'experiment2.png')
plt.close(fig)
