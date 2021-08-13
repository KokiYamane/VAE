import unittest

import torch
import torchvision
from torchvision import transforms

from tqdm import tqdm

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.VAE import VAE, VAELoss
from scripts.plot_result import *


class TestVAE(unittest.TestCase):
    def test_VAE(self):
        print('\n========== test VAE model ==========')

        image_size = 96

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
        ])

        # dataset = torchvision.datasets.STL10(
        #     root='../datasets/stl10/',
        #     split='train',
        #     transform=transform,
        #     download=True,
        # )
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

        model = VAE(image_size=image_size)
        model.eval()
        print(model)

        # device setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device:', device)
        model.to(device)

        loss_fn = VAELoss()

        for image, label in tqdm(dataloader):
            image = image.to(device)
            y, mean, std = model(image)
            loss_KL, loss_reconstruction = loss_fn(image, y, mean, std)
            loss = loss_KL + loss_reconstruction
            loss.backward()
        print('loss_KL:', loss_KL.item())
        print('loss_reconstruction:', loss_reconstruction.item())


if __name__ == "__main__":
    unittest.main()
