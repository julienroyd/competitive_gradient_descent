import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from copy import deepcopy
import time

import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_batch(batchlen, plot=False):
    cov = torch.tensor(np.identity(2) * 0.01, dtype=torch.float64)
    mu1 = torch.tensor([2 ** (-1 / 2), 2 ** (-1 / 2)], dtype=torch.float64)
    mu2 = torch.tensor([0, 1], dtype=torch.float64)

    gaussian1 = MultivariateNormal(loc=mu1, covariance_matrix=cov)
    gaussian2 = MultivariateNormal(loc=mu2, covariance_matrix=cov)

    d1 = gaussian1.rsample((int(batchlen / 2),))
    d2 = gaussian2.rsample((int(batchlen / 2),))

    data = np.concatenate((d1, d2), axis=0)
    np.random.shuffle(data)

    if plot:
        plt.scatter(data[:, 0], data[:, 1], s=2.0)
    return torch.Tensor(data).to(device)


def homemade_BCE(outputs, labels):
    return -torch.mean(labels * torch.log(outputs) + (1. - labels) * torch.log(1. - outputs))

# Define the generator
class Generator(nn.Module):
    def __init__(self, hidden_size=0, noise_size=1, noise_std=1.):
        super().__init__()
        self.noise_size = noise_size
        self.noise_std = noise_std

        self.fc1 = nn.Linear(noise_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)

        self.fc6 = nn.Linear(hidden_size, 2)

    def __call__(self, z):
        h = F.relu(self.fc1(z))

        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))

        return self.fc6(h)

    def generate(self, batchlen):
        z = torch.normal(torch.zeros(batchlen, self.noise_size), self.noise_std)
        return self.__call__(z)


# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self, hidden_size=0):
        super().__init__()

        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, 1)

    def __call__(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return torch.sigmoid(self.fc6(x))


def GAN(TRAIN_RATIO=1, N_ITER=5000, BATCHLEN=128, hidden_size_G=0, hidden_size_D=0, noise_size=1, noise_std=1., frame=1000, verbose=False, algorithm='GDA', eta=0.05):
    """
    TRAIN_RATIO : int, number of times to train the discriminator between two generator steps
    N_ITER : int, total number of training iterations for the generator
    BATCHLEN : int, Batch size to use
    hidden_size_G : int, width of the generator (number of neurons in hidden layers)
    hidden_size_D : int, width of the discriminator (number of neurons in hidden layers)
    noise_size : int, dimension of input noise
    noise_std : float, standard deviation of p(z)
    frame : int, display data each 'frame' iteration
    """
    if algorithm == 'GDA':
        compute_update = compute_gda_update
    elif algorithm == 'CGD':
        compute_update = compute_cgd_update
    else:
        raise NotImplemented

    G = Generator(hidden_size=hidden_size_G, noise_size=noise_size, noise_std=noise_std)
    D = Discriminator(hidden_size=hidden_size_D)

    for i in tqdm(range(N_ITER)):

        # train the discriminator
        real_batch = generate_batch(BATCHLEN)
        fake_batch = G.generate(BATCHLEN)

        # Compute here the total loss
        h_real = D(real_batch)
        h_fake = D(fake_batch)

        loss_real = homemade_BCE(h_real, torch.ones((BATCHLEN, 1)))
        loss_fake = homemade_BCE(h_fake, torch.zeros((BATCHLEN, 1)))

        total_loss = loss_real + loss_fake

        # Compute the update for both agents
        D_update, G_update = compute_update(f=total_loss,
                                            g=-total_loss,
                                            x=list(D.parameters()),
                                            y=list(G.parameters()),
                                            eta=eta)  # real learning rate

        with torch.no_grad():

            # Apply the update
            for p, update in zip(D.parameters(), D_update):
                p.copy_(p - eta * update)

            for p, update in zip(G.parameters(), G_update):
                p.copy_(p - eta * update)

        # visualization
        if i % frame == 0:
            if verbose:
                print('step {}: total loss: {:.3e}'.format(i, float(total_loss)))
                print("loss_real", loss_real)
                print("loss_fake", loss_fake)
            real_batch = generate_batch(1024)
            fake_batch = G.generate(1024).detach()
            plt.scatter(real_batch[:, 0], real_batch[:, 1], s=2.0, label='real data')
            plt.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2.0, label='fake data')
            plt.xlim(-1.5, 1.1)
            plt.ylim(-1., 2.5)
            plt.show()


def compute_gda_update(f, x, g, y, eta=None):
    """
    Computes the gradient step for both players
    f: loss function to minimise for player X
    x: current action (parameters) of player X
    g: loss function to minimise for player Y
    y: current action (parameters) of player y
    """
    x_update = list(grad(outputs=f, inputs=x, retain_graph=True))
    y_update = list(grad(outputs=g, inputs=y))

    return x_update, y_update

def compute_cgd_update(f, x, g, y, eta, max_it=100):
    """
    Iteratively estimate the solution for the local Nash equilibrium using the conjugate gradient method
    f: loss function to minimise for player X
    x: current action (parameters) of player X
    g: loss function to minimise for player Y
    y: current action (parameters) of player y
    """
    start = time.time()

    # Computing the gradients

    df_dx = grad(outputs=f, inputs=x, create_graph=True, retain_graph=True)
    dg_dy = grad(outputs=g, inputs=y, create_graph=True, retain_graph=True)

    with torch.no_grad():

        # Creating the appropriate structure for the parameter updates and initialising to 0

        x_update, y_update = [], []
        for x_grad_group, y_grad_group in zip(df_dx, dg_dy):
            x_update.append(torch.zeros_like(x_grad_group))
            y_update.append(torch.zeros_like(y_grad_group))

        # Creating the appropriate structure for the residuals and basis vectors and initialise them

        r_xk, r_yk, p_xk, p_yk = [], [], [], []
        for x_param_update, y_param_update in zip(df_dx, dg_dy):
            r_xk.append(torch.clone(x_param_update))
            p_xk.append(torch.clone(x_param_update))
            r_yk.append(torch.clone(y_param_update))
            p_yk.append(torch.clone(y_param_update))

    # Iteratively solve for the local Nash Equilibrium

    for k in range(max_it):

        # Computes the Hessian-vector product Ap

        hvp_y = grad(outputs=df_dx, inputs=y, grad_outputs=p_xk, retain_graph=True)
        hvp_x = grad(outputs=dg_dy, inputs=x, grad_outputs=p_yk, retain_graph=True)

        with torch.no_grad():

            # Computes step size alpha_k

            num, denom = 0., 0.
            for i in range(len(r_xk)):

                num += torch.sum(r_xk[i] ** 2.) + torch.sum(r_yk[i] ** 2.)

                tmp_x = p_xk[i] + eta * hvp_x[i]  # TODO: make sure we use the right vector here (p_yk_i or p_xk_i)
                tmp_y = p_yk[i] + eta * hvp_y[i]  # TODO: make sure we use the right vector here (p_yk_i or p_xk_i)

                denom += torch.sum(tmp_x ** 2.) + torch.sum(tmp_y ** 2.)


            alpha_k = num / denom

            # Computes new updates

            for i in range(len(x_update)):
                x_update[i] += alpha_k * hvp_x[i]
                y_update[i] += alpha_k * hvp_y[i]

            # Computes new residuals

            r_xkplus1, r_ykplus1 = [], []
            for i in range(len(r_xk)):
                r_xkplus1.append(r_xk[i] - alpha_k * hvp_x[i])
                r_ykplus1.append(r_yk[i] - alpha_k * hvp_y[i])

            # Check convergence condition

            r_xkplus1_norm, r_ykplus1_norm = [], []
            for i in range(len(r_xkplus1)):
                r_xkplus1_norm.append(torch.norm(r_xkplus1[i]))
                r_ykplus1_norm.append(torch.norm(r_ykplus1[i]))

            distance = torch.mean(torch.stack(r_xkplus1_norm + r_ykplus1_norm))
            print(distance)

            if distance <= 1e-6:
                break

            else:

                # Computes beta_k

                num, denom = 0., 0.
                for i in range(len(r_xk)):
                    num += torch.sum(r_xkplus1[i] ** 2.) + torch.sum(r_ykplus1[i] ** 2.)
                    denom += torch.sum(r_xk[i] ** 2.) + torch.sum(r_yk[i] ** 2.)

                beta_k = num / denom

                # Computes new basis vectors

                for i in range(len(p_xk)):
                    p_xk[i] = r_xkplus1[i] + beta_k * p_xk[i]
                    p_yk[i] = r_ykplus1[i] + beta_k * p_yk[i]

                r_xk = deepcopy(r_xkplus1)
                r_yk = deepcopy(r_ykplus1)


    if k+1 == max_it:
        print(f'WARNING: The conjugate gradient method required the maximum number of iterations '
              f'(max_it={max_it}) and had not even converged then. Is this normal?')
    else:
        print(f'Conjugate Gradient converged after {k} iterations')
    print(f'Compute time: {time.time() - start:.2f}')

    return x_update, y_update


if __name__ == '__main__':
    batch = generate_batch(256, plot=True)

    GAN(TRAIN_RATIO=2,
        N_ITER=5000,
        BATCHLEN=128,
        hidden_size_G=128,
        hidden_size_D=128,
        noise_size=512,
        noise_std=6,
        frame=100)
