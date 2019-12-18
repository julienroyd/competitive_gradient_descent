import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

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


def GAN(TRAIN_RATIO=1, N_ITER=5000, BATCHLEN=128, hidden_size_G=0, hidden_size_D=0, noise_size=1, noise_std=1., frame=1000, verbose=False):
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

    criterion = nn.BCELoss()

    G = Generator(hidden_size=hidden_size_G, noise_size=noise_size, noise_std=noise_std)
    optimizer_G = torch.optim.SGD(G.parameters(), lr=0.05)
    D = Discriminator(hidden_size=hidden_size_D)
    optimizer_D = torch.optim.SGD(D.parameters(), lr=0.05)

    for i in tqdm(range(N_ITER)):

        # train the discriminator
        D.zero_grad()
        real_batch = generate_batch(BATCHLEN)
        fake_batch = G.generate(BATCHLEN)

        # Compute here the discriminator loss
        h_real = D(real_batch)
        h_fake = D(fake_batch)

        loss_real = criterion(h_real, torch.ones((BATCHLEN, 1)))
        loss_fake = criterion(h_fake, torch.zeros((BATCHLEN, 1)))

        disc_loss = loss_real + loss_fake
        disc_loss.backward(retain_graph=True)
        optimizer_D.step()

        G.zero_grad()
        gen_loss = - loss_real - loss_fake
        gen_loss.backward()
        optimizer_G.step()

        # visualization
        if i % frame == 0:
            if verbose:
                print('step {}: discriminator: {:.3e}, generator: {:.3e}'.format(i, float(disc_loss), float(gen_loss)))
                print("loss_real", loss_real)
                print("loss_fake", loss_fake)
            real_batch = generate_batch(1024)
            fake_batch = G.generate(1024).detach()
            plt.scatter(real_batch[:, 0], real_batch[:, 1], s=2.0, label='real data')
            plt.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2.0, label='fake data')
            plt.xlim(-0.5, 1.)
            plt.ylim(0, 1.5)
            plt.show()


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