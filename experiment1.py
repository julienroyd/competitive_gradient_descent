import torch
import matplotlib.pyplot as plt
import numpy as np


# The problem

def f(x, y, alpha):
    return alpha * x * y


def g(x, y, alpha):
    return - alpha * x * y


x = torch.tensor([0.5], requires_grad=True)
y = torch.tensor([0.5], requires_grad=True)
alphas = [1., 3., 6.]
n_updates = 50
lr = 0.2


# GDA

def GDA_step(x, y, out_f, out_g):
    df_dx = torch.autograd.grad(outputs=out_f, inputs=x, create_graph=False)[0]
    dg_dy = torch.autograd.grad(outputs=out_g, inputs=y, create_graph=False)[0]
    return df_dx, dg_dy


# CGD

def CGD_step(x, y, out_f, out_g):
    df_dx, df_dy = torch.autograd.grad(outputs=out_f, inputs=[x, y], create_graph=True)
    dg_dx, dg_dy = torch.autograd.grad(outputs=out_g, inputs=[x, y], create_graph=True)
    return


# Execution loops

alg_names = ['GDA']
alg_updates = [GDA_step]

recorder = {alg_name: {f'alpha={alpha}': [[x, y]] for alpha in alphas} for alg_name in alg_names}
for alg_name, alg_update in zip(alg_names, alg_updates):
    for alpha in alphas:
        for i in range(n_updates):
            step_x, step_y = alg_update(x=x, y=y, out_f=f(x, y, alpha), out_g=g(x, y, alpha))
            x = x - lr * step_x
            y = y - lr * step_y

            recorder[alg_name][f'alpha={alpha}'].append([x, y])

# Plotting

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, alpha in zip(axes, alphas):

    for alg_name in alg_names:
        x_s, y_s = np.array(recorder[alg_name][f'alpha={alpha}']).T
        ax.plot(x_s, y_s, marker='o', color='blue', fillstyle='none', linewidth=1, label=alg_name)

    ax.set_title(rf'$\alpha$={alpha}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.locator_params('x', nbins=5)
    ax.locator_params('y', nbins=5)
    ax.legend(loc='upper right')
plt.show()
