import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from exp3_image_generation.src.utils import save_generated_samples, save_gif
from exp2_gaussian_mixture.scripts.CGD_vs_GDA_GaussianMixture_GAN import compute_gda_update, compute_cgd_update
from pipeline.utils.plots import create_fig, plot_curves


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, output_size)

    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, output_size)

    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = self.fc4(x)
        return x


class GAN(nn.Module):
    def __init__(self, z_size, im_size, lr, optimizer, n_epochs, train_loader, logger, dir_manager, device, alg):
        super(GAN, self).__init__()

        self.z_size = z_size
        self.im_size = im_size
        self.lr = lr
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.logger = logger
        self.dir_manager = dir_manager
        self.device = device

        if alg == 'GDA':
            self.compute_update = compute_gda_update
        elif alg == 'CGD':
            self.compute_update = compute_cgd_update
        else:
            raise NotImplemented

        self.seed = int(self.dir_manager.seed_dir.name.strip('seed'))

        # models

        self.G = Generator(input_size=z_size, output_size=im_size ** 2)
        self.D = Discriminator(input_size=im_size ** 2, output_size=1)

        # fixed noise vectors for evaluation

        self.fixed_z = torch.randn((25, 100), requires_grad=False)

        # send models on specified device

        self.send_to(device)

        # optimizers

        if self.optimizer == "adam":
            self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr)
            self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr)

        elif self.optimizer == "sgd":
            self.G_optimizer = optim.SGD(self.G.parameters(), lr=self.lr)
            self.D_optimizer = optim.SGD(self.D.parameters(), lr=self.lr)
        
        # info on training

        self.updates_completed = 0
        self.epochs_completed = 0

        self.D_loss_recorder = []  # list of lists of shape (epochs, updates)
        self.G_loss_recorder = []  # list of lists of shape (epochs, updates)

    def save_learning_curves(self):

        # Adds the loss on the very first minibatch as our point for epoch=0

        d_curve = [np.array(self.D_loss_recorder)[0][0]] + list(np.array(self.D_loss_recorder).mean(axis=1))
        g_curve = [np.array(self.G_loss_recorder)[0][0]] + list(np.array(self.G_loss_recorder).mean(axis=1))

        # Creates and saves the plot

        fig, ax = create_fig(axes_shape=(1, 1), figsize=(8, 5))
        ax = plot_curves(ax=ax,
                         ys=[d_curve, g_curve],
                         labels=['D', 'G'],
                         xlabel="Epochs",
                         ylabel="Loss")
        fig.savefig(self.dir_manager.seed_dir / "learning_curves.png")
        plt.close(fig)

    def save_checkpoint(self):
        """ Saves all the information necessary to resume training as well as training history """

        self.send_to('cpu')

        # dictionary for initialisation info

        init_dict = {'z_size': self.z_size,
                     'im_size': self.im_size,
                     'lr': self.lr,
                     'optimizer': self.optimizer,
                     'n_epochs': self.n_epochs
                     }

        # dictionary for training history

        train_dict = {'D_loss_recorder': self.D_loss_recorder,
                      'G_loss_recorder': self.G_loss_recorder,
                      'updates_completed': self.updates_completed,
                      'epochs_completed': self.epochs_completed
                      }

        # dictionary for models' parameters

        params_dict = {'D_params': self.D.state_dict(),
                       'G_params': self.G.state_dict()}

        # dictionary for the optimizers' state

        optimizers_dict = {'G_optimizer_state': self.G_optimizer.state_dict(),
                           'D_optimizer_state': self.D_optimizer.state_dict()}

        save_dict = {'init_dict': init_dict,
                     'train_dict': train_dict,
                     'params_dict': params_dict,
                     'optimizers_dict': optimizers_dict
                     }

        torch.save(save_dict, self.dir_manager.seed_dir / 'checkpoint.pt')

        self.send_to(self.device)

    @classmethod
    def init_from_checkpoint(cls, train_loader, logger, dir_manager, device):

        save_dict = torch.load(dir_manager.seed_dir / 'checkpoint.pt')

        init_dict = save_dict['init_dict']
        train_dict = save_dict['train_dict']
        params_dict = save_dict['params_dict']
        optimizers_dict = save_dict['optimizers_dict']

        # Instantiate the algorithm

        alg = cls(**init_dict,
                  train_loader=train_loader,
                  logger=logger,
                  dir_manager=dir_manager,
                  device=device)

        alg.D_loss_recorder = train_dict['D_loss_recorder']
        alg.G_loss_recorder = train_dict['G_loss_recorder']
        alg.updates_completed = train_dict['updates_completed']
        alg.epochs_completed = train_dict['epochs_completed']

        # Loads the models' parameters and optimizers' states

        alg.G.load_state_dict(params_dict['G_params'])
        alg.D.load_state_dict(params_dict['D_params'])

        alg.G_optimizer.load_state_dict(optimizers_dict['G_optimizer_state'])
        alg.D_optimizer.load_state_dict(optimizers_dict['D_optimizer_state'])

        alg.send_to(alg.device)

        return alg

    def send_to(self, device):
        self.to(device)
        self.fixed_z = self.fixed_z.to(device)

    def update_step(self, x_mb_real):

        # binary cross-entropy loss

        BCElogits_loss = nn.BCEWithLogitsLoss()

        # construct loss for D

        x_mb_real = x_mb_real.view(-1, self.im_size ** 2)

        mb_size = x_mb_real.size()[0]

        y_real1 = torch.ones((mb_size, 1))
        y_fake1 = torch.zeros((mb_size, 1))

        x_mb_real1, y_real1, y_fake1 = x_mb_real.to(self.device), y_real1.to(self.device), y_fake1.to(self.device)
        D_preds = self.D(x_mb_real1)
        D_real_loss = BCElogits_loss(D_preds, y_real1)

        z1 = torch.randn((mb_size, self.z_size)).to(self.device)
        x_mb_fake1 = self.G(z1)

        D_preds = self.D(x_mb_fake1)
        D_fake_loss = BCElogits_loss(D_preds, y_fake1)

        D_train_loss = D_real_loss + D_fake_loss

        # construct loss for G

        z2 = torch.randn((mb_size, self.z_size))
        y_real2 = torch.ones((mb_size, 1))
        z2, y_real2 = z2.to(self.device), y_real2.to(self.device)
        x_mb_fake2 = self.G(z2)
        D_preds2 = self.D(x_mb_fake2)
        G_train_loss = BCElogits_loss(D_preds2, y_real2)

        # Compute the update for both agents
        D_update, G_update = self.compute_update(f=D_train_loss,
                                                 g=G_train_loss,
                                                 x=list(self.D.parameters()),
                                                 y=list(self.G.parameters()),
                                                 eta=self.lr)  # real learning rate

        with torch.no_grad():

            # Apply the update
            for p, update in zip(self.D.parameters(), D_update):
                p.grad = update

            for p, update in zip(self.G.parameters(), G_update):
                p.grad = update

        self.D_optimizer.step()
        self.G_optimizer.step()

        # Book-keeping

        self.updates_completed += 1

        self.D_loss_recorder[self.epochs_completed].append(float(D_train_loss.data.cpu()))
        self.G_loss_recorder[self.epochs_completed].append(float(G_train_loss.data.cpu()))

    def train_model(self):

        self.logger.info('training begins')

        # training loop

        for epoch in range(self.epochs_completed, self.n_epochs):

            self.D_loss_recorder.append([])
            self.G_loss_recorder.append([])

            start_time = time.time()
            
            # mini-batch loop

            for x, _ in self.train_loader:
                self.update_step(x)

            # Some monitoring

            self.logger.info(f'[{epoch + 1}/{self.n_epochs}]: '
                             f'loss_D: {np.mean(self.D_loss_recorder[epoch]):.3f}, '
                             f'loss_G: {np.mean(self.G_loss_recorder[epoch]):.3f}, '
                             f'time: {time.time() - start_time:.2f}s for one epoch')

            self.epochs_completed += 1

            # Save generated images samples

            sampled_noise = torch.randn((5 * 5, 100), requires_grad=False, device=self.device)

            save_generated_samples(self.G, z_noise=sampled_noise,
                                   path=self.dir_manager.random_results_dir / f'randomRes_epoch{epoch}.png')

            save_generated_samples(self.G, z_noise=self.fixed_z,
                                   path=self.dir_manager.fixed_results_dir / f'fixedRes_epoch{epoch}.png')

            # Save models and learning curves

            self.save_learning_curves()
            self.save_checkpoint()

        self.logger.info('training completed')

        # Saving the models one more time and removing (now useless) checkpoint

        os.remove(self.dir_manager.seed_dir / 'checkpoint.pt')

        self.send_to(device='cpu')
        torch.save(self.G.state_dict(), self.dir_manager.seed_dir / "G_params.pt")
        torch.save(self.D.state_dict(), self.dir_manager.seed_dir / "D_params.pt")

        # Saving an animation on fixed generated samples (throughout training) to visualise progress

        save_gif(path_to_images=self.dir_manager.fixed_results_dir)
