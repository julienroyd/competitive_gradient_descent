import matplotlib.pyplot as plt
import numpy as np
import itertools
from notebooks.experiments12_common import *


# Experiment 2

def f2(x, y, alpha):
    return alpha * (x ** 2 - y ** 2)


def g2(*f_args):
    return - f2(*f_args)


def f3(x, y, alpha):
    return alpha * (-x ** 2 + y ** 2)


def g3(*f_args):
    return - f3(*f_args)


x_init = 0.5
y_init = 0.5

alphas = [1., 3., 6.]
learning_rates = [0.2]
experiments = list(itertools.product(alphas, learning_rates))

n_updates = 100

alg_names = ['ConOPT', 'OGDA', 'GDA', 'LCGD', 'CGD']
alg_updates = [ConOPT_step, OGDA_step, GDA_step, LCGD_step, CGD_step]
# colors = ['cyan', 'cyan', 'cyan', 'green', 'orange']  # paper's color-mapping
colors = ['green', 'magenta', 'brown', 'brown', 'brown']

recorder2 = run_experiment(alg_names, alg_updates, experiments, x_init, y_init, f2, g2, n_updates)
recorder3 = run_experiment(alg_names, alg_updates, experiments, x_init, y_init, f3, g3, n_updates)

# Plotting

fig, axes = plt.subplots(2, 3, figsize=(15, 5))
for i, recorder in enumerate([recorder2, recorder3]):
    for j, (alpha, lr) in enumerate(experiments):

        for alg_name, color in zip(alg_names, colors):
            x_s, y_s = np.array(recorder[alg_name][f'alpha={alpha:.1f}, lr={lr:.2f}']).T
            axes[i,j].plot(np.log10(np.linalg.norm(np.array([x_s, y_s]), axis=0)), color=color, linewidth=1, label=alg_name)

        if i == 0:
            axes[i,j].set_title(rf'$\alpha$={alpha:.1f}', fontsize=18)
        if i == 1:
            axes[i,j].set_xlabel('updates', fontsize=18)

        if j == 0:
            axes[i,j].set_ylabel(r'$log_{10}(||x_k, y_k||)$', fontsize=16, rotation=90)

        axes[i,j].tick_params(axis='both', which='major', labelsize=12)
        axes[i,j].locator_params('x', nbins=3)
        axes[i,j].locator_params('y', nbins=5)
        if i == 1 and j == 2:
            legend = axes[i, j].legend(loc='upper center', bbox_to_anchor=(-0.8, -0.4), fancybox=True, ncol=5, prop={'size': 16})

plt.show()
fig.savefig(f'experiment2.png', bbox_extra_artists=(legend,), bbox_inches='tight')
plt.close(fig)
