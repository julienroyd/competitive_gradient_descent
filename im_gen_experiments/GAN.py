import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from im_gen_experiments.utils import save_generated_samples

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
        x = torch.sigmoid(self.fc4(x))
        return x

class GAN(nn.Module):
    def __init__(self, z_size, im_size, lr, n_epochs, train_loader, logger, dir_manager, device):
        super(GAN, self).__init__()

        self.z_size = z_size
        self.im_size = im_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.logger = logger
        self.dir_manager = dir_manager
        self.device = device

        # models

        self.G = Generator(input_size=z_size, output_size=im_size**2)
        self.D = Discriminator(input_size=im_size**2, output_size=1)

        # optimizers

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lr)

        # fixed noise vectors for evaluation

        self.fixed_z = torch.randn((25, 100), requires_grad=False)

        # send models on specified device

        self.send_to(device)

        # info on training

        self.updates_completed = 0
        self.epochs_completed = 0

        self.D_loss_recorder = {}
        self.G_loss_recorder = {}

    def save_graphs(self):
        # TODO
        pass

    def save_checkpoint(self):
        # TODO: save model, optimiser state
        pass

    def init_from_checkpoint(self):
        # TODO
        pass

    def send_to(self, device):
        self.to(device)
        self.fixed_z = self.fixed_z.to(device)
        self.device = device

    def update_step(self, x_mb_real):

        # binary cross-entropy loss

        BCE_loss = nn.BCELoss()

        # train discriminator D

        self.D.zero_grad()

        x_mb_real = x_mb_real.view(-1, self.im_size**2)

        mb_size = x_mb_real.size()[0]

        y_real = torch.ones((mb_size, 1))
        y_fake = torch.zeros((mb_size, 1))

        x_mb_real, y_real, y_fake = x_mb_real.to(self.device), y_real.to(self.device), y_fake.to(self.device)
        D_preds = self.D(x_mb_real)
        D_real_loss = BCE_loss(D_preds, y_real)

        z = torch.randn((mb_size, self.z_size)).to(self.device)
        x_mb_fake = self.G(z)

        D_preds = self.D(x_mb_fake)
        D_fake_loss = BCE_loss(D_preds, y_fake)

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        self.D_optimizer.step()

        # train generator G

        self.G.zero_grad()

        z = torch.randn((mb_size, self.z_size))
        y_real = torch.ones((mb_size, 1))

        z, y_real = z.to(self.device), y_real.to(self.device)
        x_mb_fake = self.G(z)
        D_preds = self.D(x_mb_fake)
        G_train_loss = BCE_loss(D_preds, y_real)
        G_train_loss.backward()
        self.G_optimizer.step()

        # Book-keeping

        self.updates_completed += 1

        self.D_loss_recorder[f'epoch{self.epochs_completed + 1}'].append(float(D_train_loss.data.cpu()))
        self.G_loss_recorder[f'epoch{self.epochs_completed + 1}'].append(float(G_train_loss.data.cpu()))

    def train(self):

        # training loop

        for epoch in range(self.n_epochs):

            self.D_loss_recorder[f'epoch{self.epochs_completed + 1}'] = []
            self.G_loss_recorder[f'epoch{self.epochs_completed + 1}'] = []

            # mini-batch loop

            for x, _ in self.train_loader:

                self.update_step(x)

            # Some monitoring

            self.logger.info(f'[{epoch + 1}/{self.n_epochs}]: '
                        f'loss_D: {np.mean(self.D_loss_recorder[f"epoch{self.epochs_completed + 1}"]):.3f}, '
                        f'loss_G: {np.mean(self.G_loss_recorder[f"epoch{self.epochs_completed + 1}"]):.3f}')

            self.epochs_completed += 1

            # Save generated images samples

            sampled_noise = torch.randn((5 * 5, 100), requires_grad=False, device=self.device)

            save_generated_samples(self.G, z_noise=sampled_noise,
                                   path=self.dir_manager.random_results_dir / f'randomRes_epoch{epoch}.png')

            save_generated_samples(self.G, z_noise=self.fixed_z,
                                   path=self.dir_manager.fixed_results_dir / f'fixedRes_epoch{epoch}.png')

        # TODO: save model in GAN class
        # torch.save(gan.G.state_dict(), dir_manager.seed_dir / "G_params.pt")
        # torch.save(gan.D.state_dict(), dir_manager.seed_dir / "D_params.pt")

        # TODO: save model data
        # with open(dir_manager.recorders_dir / 'train_hist.pkl', 'wb') as f:
        #     pickle.dump(None, f)

        # TODO: plot the losses
        # show_train_hist(None, save=True, path=dir_manager.seed_dir / 'train_hist.png')

        # TODO: create animation
        # images = []
        # for epoch in range(config.n_epochs):
        #     img_name = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
        #     images.append(imageio.imread(img_name))
        # imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)
