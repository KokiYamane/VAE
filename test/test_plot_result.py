import unittest

import os
import time

import torch
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.VAE import VAE
from scripts.plot_result import *
from scripts.image_dataset import ImageDataset
from dataset_path import datafolder


class TestPlotResult(unittest.TestCase):
    def test_plot_result(self):
        print('\n========== test plot result ==========')

        image_size = 32

        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize(image_size),
        # ])

        # dataset = torchvision.datasets.MNIST(
        #     root='../datasets/MNIST',
        #     train=False,
        #     transform=transform,
        #     download=True,
        # )
        dataset = ImageDataset(datafolder, data_num=50, image_size=image_size)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=100,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        folder_name = 'test_image'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        for image, label in dataloader:
            image_ans = formatImages(image)

            fig = plt.figure(figsize=(20, 10))
            plot_reconstructed_image(fig, image_ans, image_ans, col=10)
            plt.savefig(folder_name + '/test_reconstructed_image.png')
            break

        start = time.time()
        fig = plt.figure(figsize=(10, 10))
        point_num = 1000
        z_dim = 10
        zs = torch.randn(point_num, z_dim)
        labels = torch.randint(high=10, size=(point_num,))
        plot_latent_space(fig, zs, labels)
        plt.savefig(folder_name + '/test_latent_space.png')
        end = time.time()
        print('elasped time:', end - start)

        model = VAE(z_dim, image_size)
        model.eval()

        # device setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = 'cpu'
        print('device:', device)
        model.to(device)

        start = time.time()
        fig = plt.figure(figsize=(10, 10))
        plot_2D_Manifold(fig, model, z_sumple=zs, device=device, image_size=124)
        plt.savefig(folder_name + '/test_2D_Manifold.png')
        end = time.time()
        print('elasped time:', end - start)

        start = time.time()
        fig = plt.figure(figsize=(10, 10))
        plot_latent_traversal(fig, model, row=z_dim, device=device, image_size=124)
        plt.savefig(folder_name + '/test_latent_traversal.png')
        end = time.time()
        print('elasped time:', end - start)


if __name__ == "__main__":
    unittest.main()
