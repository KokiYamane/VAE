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
            plot_generated_image(fig, image_ans, image_ans)
            plt.savefig(folder_name + '/test_generated_image.png')
            break

        fig = plt.figure(figsize=(10, 10))
        zs = torch.randn(100, 2)
        plot_latent_space(fig, zs, label)
        plt.savefig(folder_name + '/test_latent_space.png')


if __name__ == "__main__":
    unittest.main()
