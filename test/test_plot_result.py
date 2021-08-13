import unittest

import os

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


class TestPlotResult(unittest.TestCase):
    def test_plot_result(self):
        print('\n========== test plot result ==========')

        image_size = 96

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
        ])

        dataset = torchvision.datasets.MNIST(
            root='../datasets/MNIST',
            train=False,
            transform=transform,
            download=True,
        )

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

        fig = plt.figure(figsize=(10, 10))
        point_num = 1000
        zs = torch.randn(point_num, 10)
        labels = torch.randint(high=10, size=(point_num,))
        plot_latent_space(fig, zs, labels)
        plt.savefig(folder_name + '/test_latent_space.png')

        model = VAE(image_size=image_size)
        model.eval()

        fig = plt.figure(figsize=(10, 10))
        plot_generated_image(fig, model, device='cpu')
        plt.savefig(folder_name + '/test_generated_image.png')


if __name__ == "__main__":
    unittest.main()
