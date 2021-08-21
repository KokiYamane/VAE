import torch
from torch import nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, expand_ratio=6):
        super().__init__()

        hidden_channels = round(in_channels * expand_ratio)

        self.invertedResidual = nn.Sequential(
            # point wise
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(),

            # depth wise
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(),

            # point wise
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

    def forward(self, x):
        return self.invertedResidual(x)


class Encoder(nn.Module):
    def __init__(self, z_dim, image_size, channels, label_dim=0):
        super().__init__()

        conv_list = []
        conv_list.append(nn.Dropout(0.01))
        for i in range(len(channels)-1):
            conv_list.append(InvertedResidual(channels[i], channels[i+1], kernel_size=5, stride=2, padding=2, expand_ratio=6))
        conv_list.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_list)

        feature_size = image_size // 2**(len(channels)-1)
        feature_dim = channels[-1] * feature_size ** 2
        feature_dim += label_dim
        self.dense_encmean = nn.Linear(feature_dim, z_dim)
        self.dense_encvar = nn.Linear(feature_dim, z_dim)

    def forward(self, x, label=None):
        x = self.conv(x)
        if not label == None:
            label = label.unsqueeze(1)
            x = torch.cat([x, label], dim=1)
        mean = self.dense_encmean(x)
        std = F.relu(self.dense_encvar(x))
        return mean, std


class Decoder(nn.Module):
    def __init__(self, z_dim, image_size, channels, label_dim=0):
        super().__init__()

        self.channels = channels

        self.feature_size = image_size // 2**(len(channels)-1)

        self.dense = nn.Linear(z_dim + label_dim, channels[-1] * self.feature_size ** 2)

        deconv_list = []
        for i in reversed(range(1, len(channels))):
            deconv_list.append(nn.ReLU())
            deconv_list.append(nn.ConvTranspose2d(channels[i], channels[i-1], kernel_size=2, stride=2))
        deconv_list.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, z, label=None):
        if not label == None:
            label = label.unsqueeze(1)
            x = torch.cat([z, label], dim=1)
        x = self.dense(x)
        x = x.reshape(x.shape[0], self.channels[-1], self.feature_size, self.feature_size)
        return self.deconv(x)


class VAE(nn.Module):
    def __init__(self, z_dim=2, image_size=64, image_channel=3, label_dim=0):
        super().__init__()

        channels = [image_channel, 8, 16, 32, 64, 128]
        self.encoder = Encoder(z_dim, image_size, channels, label_dim)
        self.decoder = Decoder(z_dim, image_size, channels, label_dim)

    def _sample_z(self, mean, std):
        epsilon = torch.randn(mean.shape).to(mean.device)
        return mean + std * epsilon

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, label=None):
        mean, std = self.encoder(x, label)
        z = self._sample_z(mean, std)
        y = self.decoder(z, label)
        return y, mean, std


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _torch_log(self, x, eps=1e-10):
        return torch.log(torch.clamp(x, min=eps))

    def forward(self, x, y, mean, std):
        # Mean Squared Error
        reconstruction = ((x - y)**2).reshape(x.shape[0], -1).sum(axis=1).mean()

        # Kullback–Leibler divergence
        KL = -0.5 * (1 + self._torch_log(std**2) - mean**2 - std**2).sum(axis=1).mean()

        return reconstruction, KL
