# from ast import Tuple
import sys
import os
# import time
# import datetime
import wandb

import torch
# from torch import nn
# import torch.optim as optim
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

sys.path.append('.')
sys.path.append('..')
from scripts.VAE import VAE, VAELoss
from scripts.image_dataset import ImageDataset
from scripts.trainer import Tranier
from scripts.plot_result import formatImages, plot_reconstructed_image
from scripts.plot_result import plot_latent_space, plot_2D_Manifold
from scripts.plot_result import plot_latent_traversal


class VAETrainer(Tranier):
    def __init__(
        self,
        data_path: str,
        out_dir: str,
        batch_size: int,
        image_size: int,
        learning_rate: float,
        wandb_flag: bool,
        gpu: list = [0],
        conditional: bool = False,
    ):
        self.out_dir = out_dir
        self.loss_fn = VAELoss()
        self.conditional = conditional
        self.device = torch.device(f'cuda:{gpu[0]}'
                                   if torch.cuda.is_available() else 'cpu')

        # figure
        self.fig_reconstructed_image = plt.figure(figsize=(20, 10))
        self.fig_latent_space = plt.figure(figsize=(10, 10))
        self.fig_2D_Manifold = plt.figure(figsize=(10, 10))
        self.fig_latent_traversal = plt.figure(figsize=(10, 10))
        self.fig_loss = plt.figure(figsize=(10, 10))
        self.valid_ans = []
        self.valid_hat = []
        self.valid_mean = []
        self.valid_label = []
        self.train_loss_mse = 0.0
        self.train_loss_ssim = 0.0
        self.train_loss_kl = 0.0
        self.valid_loss_mse = 0.0
        self.valid_loss_ssim = 0.0
        self.valid_loss_kl = 0.0

        train_dataset, valid_dataset, label_dim = torchvision_dataset(
            data_path, image_size=image_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            # pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            # pin_memory=True,
            # drop_last=True,
        )

        print('train data num:', len(train_dataset))
        print('valid data num:', len(valid_dataset))

        image, label = train_dataset[0]
        n_channel = image.shape[-3]
        # def label_transform(x): return x
        if args.conditional:
            self.label_transform = self.to_one_hot(label_dim)
        else:
            label_dim = 0
            self.label_transform = lambda x: None
        model = VAE(
            z_dim=10,
            image_size=args.image_size,
            n_channel=n_channel,
            label_dim=label_dim
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        super().__init__(
            train_loader=train_loader,
            valid_loader=valid_loader,
            model=model,
            calc_loss=self.calc_loss,
            optimizer=optimizer,
            out_dir=out_dir,
            wandb_flag=wandb_flag,
            gpu=gpu,
        )

        if wandb_flag:
            wandb.init(project='VAE')
            wandb.watch(model)

            config = wandb.config

            config.data_path = args.data
            config.epoch = args.epoch
            config.batch_size = args.batch_size
            config.learning_rate = args.learning_rate
            config.image_size = args.image_size
            config.gpu_num = args.gpu
            config.conditional = args.conditional

            config.train_data_num = len(train_dataset)
            config.valid_data_num = len(valid_dataset)

    def to_one_hot(self, n_class=10):
        return lambda labels: torch.eye(n_class)[labels]

    def calc_loss(
        self,
        batch,
        valid: bool = False,
    ) -> torch.Tensor:
        image, label = batch
        image = image.to(self.device)

        with torch.cuda.amp.autocast():
            if self.conditional:
                label_one_hot = self.label_transform(label)
                label_one_hot = label_one_hot.to(self.device)
                y, mean, std = self.model(image, label_one_hot, affine=True)
            else:
                y, mean, std = self.model(image, affine=True)
            loss_mse, loss_ssim, loss_kl = self.loss_fn(image, y, mean, std)

        if valid and len(self.valid_mean) * mean.shape[0] < 1000:
            self.valid_mean.append(mean)
            self.valid_label.append(label)

        if valid and len(self.valid_ans) * y.shape[0] < 16:
            self.valid_ans.append(image)
            self.valid_hat.append(y)

        if not valid:
            self.train_loss_mse += loss_mse.item()
            self.train_loss_ssim += loss_ssim.item()
            self.train_loss_kl += loss_kl.item()
        else:
            self.valid_loss_mse += loss_mse.item()
            self.valid_loss_ssim += loss_ssim.item()
            self.valid_loss_kl += loss_kl.item()

        loss = loss_mse + loss_ssim + loss_kl
        return loss

    def plot_result(self, epoch: int):
        # wandb
        train_loss_mse = self.train_loss_mse / len(self.train_loader)
        train_loss_ssim = self.train_loss_ssim / len(self.train_loader)
        train_loss_kl = self.train_loss_kl / len(self.train_loader)
        valid_loss_mse = self.valid_loss_mse / len(self.valid_loader)
        valid_loss_ssim = self.valid_loss_ssim / len(self.valid_loader)
        valid_loss_kl = self.valid_loss_kl / len(self.valid_loader)
        self.train_loss_mse = 0.0
        self.train_loss_ssim = 0.0
        self.train_loss_kl = 0.0
        self.valid_loss_mse = 0.0
        self.valid_loss_ssim = 0.0
        self.valid_loss_kl = 0.0
        if self.wandb_flag:
            wandb.log({
                'epoch': epoch,
                'iteration': len(self.train_loader) * epoch,
                'train_loss_mse': train_loss_mse,
                'train_loss_ssim': train_loss_ssim,
                'train_loss_kl': train_loss_kl,
                'valid_loss_mse': valid_loss_mse,
                'valid_loss_ssim': valid_loss_ssim,
                'valid_loss_kl': valid_loss_kl,
            })

        if epoch % 100 == 0 or (epoch % 10 == 0 and epoch <= 100):
            valid_mean = torch.cat(self.valid_mean, dim=0)
            valid_label = torch.cat(self.valid_label, dim=0)
            self.valid_mean = []
            self.valid_label = []

            valid_ans = torch.cat(self.valid_ans, dim=0)
            valid_hat = torch.cat(self.valid_hat, dim=0)
            image_ans = formatImages(valid_ans)
            image_hat = formatImages(valid_hat)
            self.valid_ans = []
            self.valid_hat = []
            self.fig_reconstructed_image.clf()
            plot_reconstructed_image(
                self.fig_reconstructed_image,
                image_ans,
                image_hat,
                col=4,
                epoch=epoch
            )
            self.fig_reconstructed_image.savefig(
                os.path.join(self.out_dir, 'reconstructed_image.png'))

            self.fig_latent_space.clf()
            plot_latent_space(
                self.fig_latent_space,
                valid_mean,
                valid_label,
                epoch=epoch,
            )
            self.fig_latent_space.savefig(
                os.path.join(self.out_dir, 'latent_space.png'))

            # self.fig_2D_Manifold.clf()
            # if self.conditional:
            #     label = self.label[0]
            # else:
            #     label = None
            # plot_2D_Manifold(
            #     self.fig_2D_Manifold,
            #     self.model,
            #     self.device,
            #     z_sumple=valid_mean,
            #     col=10,
            #     epoch=epoch,
            #     label=label,
            #     label_transform=self.label_transform,
            # )
            # self.fig_2D_Manifold.savefig(
            #     os.path.join(self.out_dir, '2D_Manifold.png'))

            # self.fig_latent_traversal.clf()
            # plot_latent_traversal(
            #     self.fig_latent_traversal,
            #     self.model,
            #     self.device,
            #     row=valid_mean.shape[1],
            #     col=10,
            #     epoch=epoch,
            #     label=label,
            #     label_transform=self.label_transform,
            # )
            # self.fig_latent_traversal.savefig(
            #     os.path.join(self.out_dir, 'latent_traversal.png'))

            if self.wandb_flag:
                wandb.log({
                    'epoch': epoch,
                    'reconstructed_image': wandb.Image(self.fig_reconstructed_image),
                    'latent_space': wandb.Image(self.fig_latent_space),
                    # '2D_Manifold': wandb.Image(self.fig_2D_Manifold),
                    # 'latent_traversal': wandb.Image(self.fig_latent_traversal),
                })

            plt.close()

    def train(self, n_epochs: int):
        return super().train(n_epochs, callback=self.plot_result)


def torchvision_dataset(dataset, image_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
    ])

    label_dim = 10
    if dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root='../datasets/mnist', train=True,
            download=True, transform=transform)
        valid_dataset = torchvision.datasets.MNIST(
            root='../datasets/mnist', train=False,
            download=True, transform=transform)
    elif dataset == 'emnist':
        train_dataset = torchvision.datasets.EMNIST(
            root='../datasets/emnist', split='balanced', train=True,
            download=True, transform=transform)
        valid_dataset = torchvision.datasets.EMNIST(
            root='../datasets/emnist', split='balanced', train=False,
            download=True, transform=transform)
    elif dataset == 'fashion-mnist':
        train_dataset = torchvision.datasets.FashionMNIST(
            root='../datasets/fashion-mnist', train=True,
            download=True, transform=transform)
        valid_dataset = torchvision.datasets.FashionMNIST(
            root='../datasets/fashion-mnist', train=False,
            download=True, transform=transform)
    elif dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='../datasets/cifar10', train=True,
            download=True, transform=transform)
        valid_dataset = torchvision.datasets.CIFAR10(
            root='../datasets/cifar10', train=False,
            download=True, transform=transform)
    elif dataset == 'stl10':
        train_dataset = torchvision.datasets.STL10(
            root='../datasets/stl10', split='train',
            download=True, transform=transform)
        valid_dataset = torchvision.datasets.STL10(
            root='../datasets/stl10', split='test',
            download=True, transform=transform)
    elif dataset == 'celebA':
        train_dataset = torchvision.datasets.CelebA(
            root='../datasets/celebA', split='train', target_type='identity',
            download=True, transform=transform)
        valid_dataset = torchvision.datasets.CelebA(
            root='../datasets/celebA', split='valid', target_type='identity',
            download=True, transform=transform)
    else:
        train_dataset = ImageDataset(
            dataset, train=True, image_size=image_size)
        valid_dataset = ImageDataset(
            dataset, train=False, image_size=image_size)
        label_dim = train_dataset.label_dim

    return train_dataset, valid_dataset, label_dim


def main(args):
    VAETrainer(
        data_path=args.data,
        out_dir=args.output,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        wandb_flag=args.wandb,
        gpu=args.gpu,
        conditional=args.conditional,
    ).train(args.epoch)


def argparse():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--output', type=str, default='./results/test/')
    parser.add_argument('--epoch', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--wandb', action='store_true')

    def tp(x):
        return list(map(int, x.split(',')))
    parser.add_argument('--gpu', type=tp, default='0')
    parser.add_argument('--conditional', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparse()
    main(args)
