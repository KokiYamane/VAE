import unittest

import os
import time
import torch
import cv2
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.VAE import VAE
from scripts.image_dataset import ImageDataset
from dataset_path import datafolder
from scripts.plot_result import formatImages, plot_reconstructed_image
from scripts.plot_result import plot_latent_space, plot_2D_Manifold
from scripts.plot_result import plot_latent_traversal


class TestPlotResult(unittest.TestCase):
    def test_plot_result(self):
        print('\n========== test plot result ==========')

        image_size = 256

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
            # print(image.shape)
            image_ans = formatImages(image)

            # preprocess
            # image_preprocessed = []
            # for image in image_ans:
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)[:, :, 2]
            #     image = (255 * image).astype(np.uint8)
            #     ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
            #     # threshold = 200
            #     # ret, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            #     image_preprocessed.append(image)
            # image_preprocessed = np.array(image_preprocessed)
            image_preprocessed = image_ans

            fig = plt.figure(figsize=(20, 10))
            plot_reconstructed_image(fig, image_ans, image_preprocessed, col=4)
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
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        print('device:', device)
        model.to(device)

        start = time.time()
        fig = plt.figure(figsize=(10, 10))
        plot_2D_Manifold(
            fig,
            model,
            z_sumple=zs,
            device=device,
            image_size=124,
        )
        plt.savefig(folder_name + '/test_2D_Manifold.png')
        end = time.time()
        print('elasped time:', end - start)

        start = time.time()
        fig = plt.figure(figsize=(10, 10))
        plot_latent_traversal(
            fig,
            model,
            row=z_dim,
            device=device,
            image_size=124,
        )
        plt.savefig(folder_name + '/test_latent_traversal.png')
        end = time.time()
        print('elasped time:', end - start)


if __name__ == "__main__":
    unittest.main()
