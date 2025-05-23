import os
import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.VAE import VAE
from scripts.image_dataset import ImageDataset


def load_model_param(filepath):
    state_dict = torch.load(filepath)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict


def plot_anomaly_detection(
    images_ans: np.array,
    images_hat: np.array,
    error: np.array,
    col=4,
):
    fig = plt.figure(figsize=(20, 10))
    image_num = int(col**2)
    images_ans = images_ans[:image_num]
    images_hat = images_hat[:image_num]

    cmap = None
    channel = images_ans.shape[3]
    if channel == 1:
        cmap = 'gray'
        # cmap = 'binary'
        images_ans = np.squeeze(images_ans)
        images_hat = np.squeeze(images_hat)

    row = -(-len(images_ans) // col)
    for i, (image_ans, image_hat, error) in enumerate(zip(images_ans, images_hat, error)):
        ax = fig.add_subplot(row, 3 * col, 3 * i + 1)
        ax.imshow(image_ans, cmap=cmap)
        ax.axis('off')
        # ax.set_title('$i_{' + str(i) + '}$')

        ax = fig.add_subplot(row, 3 * col, 3 * i + 2)
        ax.imshow(image_hat, cmap=cmap)
        ax.axis('off')
        # ax.set_title(r'$\hat{i}_{' + str(i) + '}$')

        ax = fig.add_subplot(row, 3 * col, 3 * i + 3)
        ax.imshow(error, cmap='jet')
        ax.axis('off')

    # fig.suptitle('{} epoch'.format(epoch))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    return fig


def main(args):
    valid_dataset = ImageDataset(
        args.data,
        train=False,
        image_size=args.image_size,
        split_ratio=0.5,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
    )

    image, label = valid_dataset[0]
    model = VAE(
        z_dim=10,
        image_size=args.image_size,
        n_channel=image.shape[-3]
    )
    state_dict = load_model_param(args.model)
    model.load_state_dict(state_dict)
    model.eval()

    # device setting
    cuda_flag = torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu[0]}' if cuda_flag else 'cpu')
    model = model.to(device)

    pred_list = []
    for image, label in tqdm(valid_loader):
        # print(image.shape)
        image = image.to(device)
        pred, mean, std = model(image, affine=True)
        # print(pred.shape)
        pred_list.append(pred)
    pred = torch.cat(pred_list, dim=0)
    pred = pred.cpu().detach().numpy().copy()
    pred = pred.transpose(0, 2, 3, 1)

    image = valid_dataset.image
    image = image.cpu().detach().numpy().copy()
    image = image.transpose(0, 2, 3, 1)

    diff = image - pred

    fig = plot_anomaly_detection(image, pred, diff, col=4)
    fig.savefig('results/anomaly_detection.png')

    out_dir = './results/anomaly_detection'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, (image_ans, image_hat, error) in enumerate(zip(image, pred, diff)):
        fig = plt.figure(figsize=(20, 10))

        cmap = 'gray'
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(image_ans, cmap=cmap)
        ax.axis('off')
        # ax.set_title('$i_{' + str(i) + '}$')

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(image_hat, cmap=cmap)
        ax.axis('off')
        # ax.set_title(r'$\hat{i}_{' + str(i) + '}$')

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(error, cmap='jet')
        ax.axis('off')
        fig.savefig(f'./results/anomaly_detection/{i}.png')



def argparse():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str)
    parser.add_argument('--image_size', type=int, default=256)

    def tp(x):
        return list(map(int, x.split(',')))
    parser.add_argument('--gpu', type=tp, default='0')
    return parser.parse_args()


if __name__ == '__main__':
    args = argparse()
    main(args)
