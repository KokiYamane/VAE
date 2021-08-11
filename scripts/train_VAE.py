import sys
import os
import time
import datetime
import wandb

import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

sys.path.append('.')
sys.path.append('..')
from scripts.VAE import VAE, VAELoss
from scripts.plot_result import *


def train_VAE(n_epochs, train_loader, valid_loader, model, loss_fn,
              out_dir='', lr=0.001, optimizer_cls=optim.Adam,
              wandb_flag=False, gpu_num=0):
    train_losses, valid_losses = [], []
    total_elapsed_time = 0
    best_test = 1e10
    optimizer = optimizer_cls(model.parameters(), lr=lr)

    # device setting
    device = torch.device('cuda:{}'.format(gpu_num[0]) if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=gpu_num)
    model.to(device)

    # acceleration
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True

    # figure
    fig_generated_image = plt.figure(figsize=(20, 10))

    for epoch in range(n_epochs + 1):
        start = time.time()

        running_loss = 0.0
        running_loss_KL = 0.0
        running_loss_reconstruction = 0.0
        model.train()
        for image, label in tqdm(train_loader):
            image = image.to(device)

            with torch.cuda.amp.autocast():
                y, mean, std = model(image)
                loss_KL, loss_reconstruction = loss_fn(image, y, mean, std)

            loss = loss_KL + loss_reconstruction

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_loss_KL += loss_KL.item()
            running_loss_reconstruction += loss_reconstruction.item()
        train_loss = running_loss / len(train_loader)
        train_loss_KL = running_loss_KL / len(train_loader)
        train_loss_reconstruction = running_loss_reconstruction / len(train_loader)
        train_losses.append(train_loss)

        running_loss = 0.0
        running_loss_KL = 0.0
        running_loss_reconstruction = 0.0
        model.eval()
        for image, label in valid_loader:
            image = image.to(device)

            with torch.cuda.amp.autocast():
                y, mean, std = model(image)
                loss_KL, loss_reconstruction = loss_fn(image, y, mean, std)

            loss = loss_KL + loss_reconstruction

            running_loss += loss.item()
            running_loss_KL += loss_KL.item()
            running_loss_reconstruction += loss_reconstruction.item()
        valid_loss = running_loss / len(valid_loader)
        valid_loss_KL = running_loss_KL / len(valid_loader)
        valid_loss_reconstruction = running_loss_reconstruction / len(valid_loader)
        valid_losses.append(valid_loss)

        end = time.time()
        elapsed_time = end - start
        total_elapsed_time += elapsed_time
        print('epoch: {} train loss: {:}, vaild loss: {:}, elapsed time: {:.3f}'.format(
            epoch, train_loss, valid_loss, elapsed_time))

        # save model
        if valid_loss < best_test:
            best_test = valid_loss
            path_model_param_best = os.path.join(out_dir, 'model_param_best.pt')
            torch.save(model.state_dict(), path_model_param_best)
            if wandb_flag:
                wandb.save(path_model_param_best)

        if epoch % 100 == 0:
            model_param_dir = os.path.join(out_dir, 'model_param')
            if not os.path.exists(model_param_dir):
                os.mkdir(model_param_dir)
            path_model_param = os.path.join(
                model_param_dir,
                'model_param_{:06d}.pt'.format(epoch))
            torch.save(model.state_dict(), path_model_param)

            if wandb_flag:
                wandb.save(path_model_param)

        # save checkpoint
        path_checkpoint = os.path.join(out_dir, 'checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path_checkpoint)

        # plot loss
        plt.clf()
        plt.plot(train_losses, label='train')
        plt.plot(valid_losses, label='valid')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.title('loss')
        plt.savefig(os.path.join(out_dir, 'loss.png'))

        # show output
        if epoch % 10 == 0:
            image_ans = formatImages(image)
            image_hat = formatImages(y)
            fig_generated_image.clf()
            plot_generated_image(fig_generated_image, image_ans, image_hat)
            fig_generated_image.suptitle('{} epoch'.format(epoch))
            path_generated_image_png = os.path.join(out_dir, 'generated_image.png')
            fig_generated_image.savefig(path_generated_image_png)

            if wandb_flag:
                wandb.log({
                    'epoch': epoch,
                    'generated_image': wandb.Image(fig_generated_image),
                })
                wandb.save(path_generated_image_png)

        # wandb
        if wandb_flag:
            wandb.log({
                'epoch': epoch,
                'iteration': len(train_loader) * epoch,
                'train_loss': train_loss,
                'train_loss_KL': train_loss_KL,
                'train_loss_reconstruction': train_loss_reconstruction,
                'valid_loss': valid_loss,
                'valid_loss_KL': valid_loss_KL,
                'valid_loss_reconstruction': valid_loss_reconstruction,
            })
            wandb.save(path_checkpoint)

    print('total elapsed time: {} [s]'.format(total_elapsed_time))


def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.image_size),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(20)
    ])

    # train_dataset = torchvision.datasets.STL10(
    #     root='../datasets/stl10/',
    #     split='train',
    #     # split='unlabeled',
    #     transform=transform,
    #     download=True,
    # )
    # valid_dataset = torchvision.datasets.STL10(
    #     root='../datasets/stl10/',
    #     split='test',
    #     # split='train',
    #     transform=transform,
    #     download=True,
    # )
    train_dataset = torchvision.datasets.MNIST(
        root='../datasets/MNIST',
        train=True,
        transform=transform,
        download=True,
    )
    valid_dataset = torchvision.datasets.MNIST(
        root='../datasets/MNIST',
        train=False,
        transform=transform,
        download=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        # pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        # pin_memory=True,
    )

    model = VAE(image_size=args.image_size)

    if not os.path.exists('results'):
        os.mkdir('results')
    out_dir = 'results/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.mkdir(out_dir)

    if args.wandb:
        wandb.init(project='CVAE')
        config = wandb.config
        config.data_path = args.data_path
        config.epoch = args.epoch
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.gpu_num = args.gpu_num
        config.train_data_num = len(train_dataset)
        config.valid_data_num = len(valid_dataset)
        wandb.watch(model)
        wandb.save(os.path.join(out_dir, 'norm_mean.csv'))
        wandb.save(os.path.join(out_dir, 'norm_std.csv'))

    train_VAE(
        n_epochs=args.epoch,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        loss_fn=VAELoss(),
        out_dir=out_dir,
        lr=args.learning_rate,
        wandb_flag=args.wandb,
        gpu_num=args.gpu_num,
    )


def argparse():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=96)
    parser.add_argument('--wandb', action='store_true')
    tp = lambda x:list(map(int, x.split(',')))
    parser.add_argument('--gpu_num', type=tp, default='0')
    return parser.parse_args()


if __name__ == '__main__':
    args = argparse()
    main(args)
