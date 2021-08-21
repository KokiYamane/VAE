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

        image_size = 32

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

        image, label = dataset[0]
        image_channel = image.shape[-3]
        model = VAE(image_size=image_size, image_channel=image_channel, label_dim=1)
        model.eval()
        print(model)

        # device setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device:', device)
        model.to(device)

        loss_fn = VAELoss()

        for image, label in tqdm(dataloader):
            image = image.to(device)
            label = label.to(device)
            y, mean, std = model(image, label)
            loss_KL, loss_reconstruction = loss_fn(image, y, mean, std)
            loss = loss_KL + loss_reconstruction
            loss.backward()
        print('loss_KL:', loss_KL.item())
        print('loss_reconstruction:', loss_reconstruction.item())

        z = torch.randn(size=(10, 2)).to(device)
        label = torch.zeros(size=(10,)).to(device)
        images = model.decode(z, label)


if __name__ == "__main__":
    unittest.main()
