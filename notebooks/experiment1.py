import matplotlib.pyplot as plt
import numpy as np
import itertools
from notebooks.experiments12_common import *


# Experiment 1

def f1(x, y, alpha):
    return alpha * x * y


def g1(*f_args):
    return - f1(*f_args)

x_init = 0.5
y_init = 0.5

alphas = [1., 3., 6.]
learning_rates = [0.04, 0.2, 1.]
experiments = list(itertools.product(alphas, learning_rates))

n_updates = 100

alg_names = ['ConOPT', 'OGDA', 'GDA', 'LCGD', 'CGD']
alg_updates = [ConOPT_step, OGDA_step, GDA_step, LCGD_step, CGD_step]
colors = ['green', 'magenta', 'cyan', 'orange', 'brown']
markers = ['s', 'x', '^', '+', 'o']

recorder = run_experiment(alg_names, alg_updates, experiments, x_init, y_init, f1, g1, n_updates)

# Plotting

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for (j,i), (alpha, lr) in zip(itertools.product(range(3), range(3)), experiments):

    for alg_name, color, marker in zip(alg_names, colors, markers):
        x_s, y_s = np.array(recorder[alg_name][f'alpha={alpha:.1f}, lr={lr:.2f}']).T
        axes[i,j].plot(x_s, y_s, marker=marker, color=color, fillstyle='none', linewidth=1, label=alg_name)

    if i == 0:
        axes[i, j].set_title(rf'$\alpha$={alpha:.1f}', fontsize=18)
    if j == 2:
        axes[i, j].text(s=rf'$\eta$={lr:.2f}', rotation=-35, x=2.2, y=-0.5, fontsize=18)
    if i == 2:
        axes[i, j].set_xlabel('x', fontsize=18)
    if j == 0:
        axes[i, j].set_ylabel('y', fontsize=18)
    axes[i, j].set_xlim(-2, 2)
    axes[i, j].set_ylim(-2, 2)
    axes[i, j].tick_params(axis='both', which='major', labelsize=12)
    axes[i, j].locator_params('x', nbins=3)
    axes[i, j].locator_params('y', nbins=3)
    if i == 2 and j == 2:
        axes[i, j].legend(loc='upper center', bbox_to_anchor=(-0.75, -0.25), fancybox=True, ncol=5, prop={'size': 16})

plt.show()
fig.savefig(f'experiment1.png')
plt.close(fig)
