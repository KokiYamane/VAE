import torch
from torch import nn
import torch.nn.functional as F
from scripts.SSIM import SSIMLoss
import copy


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        expand_ratio: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_channels = round(in_channels * expand_ratio)

        self.invertedResidual = nn.Sequential(
            # point wise
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),

            # depth wise
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                stride,
                padding,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),

            # point wise
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.invertedResidual(x)


class Encoder(nn.Module):
    def __init__(self, z_dim, channels, label_dim=0):
        super().__init__()

        self.channels = copy.deepcopy(channels)

        # add label dim
        if label_dim != 0:
            label_vec_dim = 3
            self.dense_label = nn.Linear(label_dim, label_vec_dim)
            self.channels[0] += label_vec_dim

        conv_list = []
        conv_list.append(nn.Dropout(0.01))
        for i in range(len(self.channels) - 1):
            conv_list.append(
                InvertedResidual(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i + 1],
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    expand_ratio=6,
                    dropout=0.1,
                )
            )
        self.conv = nn.Sequential(*conv_list)

        feature_dim = self.channels[-1]

        self.dense_mean = nn.Linear(feature_dim, z_dim)
        self.dense_var = nn.Linear(feature_dim, z_dim)
        self.dense_affine = nn.Linear(feature_dim, 4)

    def forward(self, x, label=None, affine=False):
        if label is not None:
            if label.dim() == 1:
                label = label.unsqueeze(1)
            v = self.dense_label(label.float())
            image_size = x.shape[-1]
            v = v.repeat(image_size, image_size, 1, 1)
            v = v.reshape(x.shape[0], -1, image_size, image_size)
            x = torch.cat([x, v], dim=1)

        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze(-1).squeeze(-1)

        mean = self.dense_mean(x)
        std = F.relu(self.dense_var(x))

        if affine:
            affine = self.dense_affine(x)
            return mean, std, affine
        else:
            return mean, std


class Decoder(nn.Module):
    def __init__(self, z_dim, channels, label_dim=0):
        super().__init__()

        self.channels = copy.deepcopy(channels)
        input_dim = 2 + z_dim

        # add label dim
        if label_dim != 0:
            label_vec_dim = 3
            self.dense_label = nn.Linear(label_dim, label_vec_dim)
            input_dim += label_vec_dim

        self.channels.append(input_dim)
        self.channels.reverse()

        layer_list = []
        for i in range(0, len(self.channels) - 1):
            layer_list.append(
                InvertedResidual(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i + 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    expand_ratio=6,
                )
            )

            if i != len(channels) - 2:
                layer_list.append(nn.ReLU())
        layer_list.append(nn.Sigmoid())
        self.conv = nn.Sequential(*layer_list)

    def forward(self, z, image_size, label=None, affine=None):
        if type(image_size) == int:
            height = width = image_size
        else:
            height, width = image_size

        x = torch.linspace(-1, 1, width).repeat(height, 1)
        y = torch.linspace(-1, 1, height).tile(width, 1).permute(1, 0)
        xy = torch.stack([x, y]).to(z.device)
        xy = xy.repeat(z.shape[0], 1, 1, 1)

        if affine is not None:
            scale, theta, tx, ty = affine.split(1, dim=1)

            # translate
            t = torch.cat([tx, ty], dim=1)
            t = t.repeat(xy.shape[2], xy.shape[3], 1, 1)
            t = t.permute(2, 3, 0, 1)
            xy += t

            # scale
            # scale = scale.repeat(xy.shape[2], xy.shape[3], 1, 2)
            # scale = scale.permute(2, 3, 0, 1)
            # xy *= torch.exp(scale)

            # rotation and scale
            # cos = scale * torch.cos(theta)
            # sin = scale * torch.sin(theta)
            # rot = torch.stack([
            #     cos, sin,
            #     -sin, cos
            # ]).reshape(cos.shape[0], 2, 2)
            # print(rot.shape)
            # rot = rot.unsqueeze(1).repeat(1, xy.shape[1], 1, 1)
            # print(rot.shape)
            # print(xy.shape)
            # batch_size, pixel_num, _ = xy.shape
            # rot = rot.reshape(batch_size * pixel_num, 2, 2)
            # xy = xy.reshape(batch_size * pixel_num, 2)
            # xy = torch.matmul(xy, rot)
            # xy = xy.reshape(batch_size, pixel_num, 2)
            # print(xy.shape)

        if label is not None:
            v = self.dense_label(label.float()).to(z.device)
            z = torch.cat([z, v], dim=1)

        z = z.repeat(height, width, 1, 1)
        z = z.permute(2, 3, 0, 1)
        z = torch.cat([xy, z], dim=1)

        image = self.conv(z)

        return image


class VAE(nn.Module):
    def __init__(self, z_dim=2, image_size=64, n_channel=3, label_dim=0):
        super().__init__()

        self.image_size = image_size

        channels = [n_channel, 8, 16, 32, 64, 128]
        self.encoder = Encoder(z_dim, channels, label_dim)
        self.decoder = Decoder(z_dim, channels, label_dim)

    def _sample_z(self, mean, std):
        epsilon = torch.randn(mean.shape).to(mean.device)
        return mean + std * epsilon

    def forward(self, x, label=None, affine=True):
        if affine:
            mean, std, affine = self.encoder(x, label, affine)
        else:
            mean, std = self.encoder(x, label)
            affine = None
        z = self._sample_z(mean, std)
        y = self.decoder(z, self.image_size, label, affine)
        return y, mean, std


class VAELoss(nn.Module):
    def __init__(self, weight_mse=1000.0, weight_ssim=0.0):
        super().__init__()

        self.weight_mse = weight_mse
        self.weight_ssim = weight_ssim
        self.ssim_loss = SSIMLoss()

    def _torch_log(self, x, eps=1e-10):
        return torch.log(torch.clamp(x, min=eps))

    def forward(self, x, y, mean, std):
        # Mean Squared Error
        mse = self.weight_mse * F.mse_loss(x, y)

        # Structural Similarity
        ssim = self.weight_ssim * self.ssim_loss(x, y)

        # Kullback–Leibler divergence
        kld = -0.5 * (1 + self._torch_log(std**2) - mean**2 - std**2).mean()

        return mse, ssim, kld
