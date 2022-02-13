import numpy as np
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
import torch


def torch2numpy(tensor):
    return tensor.cpu().detach().numpy().copy()


def formatImages(image):
    image = torch2numpy(image).astype(np.float32)
    image = image.transpose(0, 2, 3, 1)
    return image


def plot_reconstructed_image(fig, images_ans, images_hat, col=4, epoch=0):
    image_num = int(col**2)
    images_ans = images_ans[:image_num]
    images_hat = images_hat[:image_num]

    cmap = None
    channel = images_ans.shape[3]
    if channel == 1:
        # cmap = 'gray'
        cmap = 'binary'
        images_ans = np.squeeze(images_ans)
        images_hat = np.squeeze(images_hat)

    row = -(-len(images_ans) // col)
    for i, (image_ans, image_hat) in enumerate(zip(images_ans, images_hat)):
        ax = fig.add_subplot(row, 2 * col, 2 * i + 1)
        ax.imshow(image_ans, cmap=cmap)
        ax.axis('off')
        # ax.set_title('$i_{' + str(i) + '}$')

        ax = fig.add_subplot(row, 2 * col, 2 * i + 2)
        ax.imshow(image_hat, cmap=cmap)
        ax.axis('off')
        # ax.set_title(r'$\hat{i}_{' + str(i) + '}$')

    fig.suptitle('{} epoch'.format(epoch))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)


def plot_latent_space(fig, zs, labels, epoch=0):
    zs = torch2numpy(zs)
    labels = torch2numpy(labels)

    if zs.shape[1] > 2:
        pca = PCA()
        pca.fit(zs)
        zs = pca.transform(zs)
    # zs = TSNE(n_components=2, random_state=0).fit_transform(zs)

    ax = fig.add_subplot(111)
    im = ax.scatter(zs[:, 0], zs[:, 1], c=labels,
                    cmap='nipy_spectral', marker='.')
    lim = np.max(np.abs(zs)) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, ax=ax, orientation='vertical', cax=cax)
    ax.set_title('{} epoch'.format(epoch))


def plot_2D_Manifold(fig, model, device, z_sumple, col=10, epoch=0,
                     label=None, label_transform=lambda x: x, image_size=64):
    row = col

    x = np.tile(np.linspace(-2, 2, col), row)
    y = np.repeat(np.linspace(2, -2, row), col)
    z = np.stack([x, y]).transpose()
    zeros = np.zeros(shape=(z.shape[0], z_sumple.shape[1] - z.shape[1]))
    z = np.concatenate([z, zeros], axis=1)

    if z_sumple.shape[1] > 2:
        z_sumple = torch2numpy(z_sumple)
        pca = PCA()
        pca.fit(z_sumple)
        z = pca.inverse_transform(z)
    z = torch.from_numpy(z.astype(np.float32)).to(device)

    if label is not None:
        label_transformed = label_transform(label)
        label_transformed = label_transformed.repeat(z.shape[0], 1).to(device)
    else:
        label_transformed = None

    images = model.decoder(z, image_size=image_size, label=label_transformed)
    images = formatImages(images)

    cmap = None
    channel = images.shape[3]
    if channel == 1:
        # cmap = 'gray'
        cmap = 'binary'
        images = np.squeeze(images)

    for i, image in enumerate(images):
        ax = fig.add_subplot(row, col, i + 1)
        ax.imshow(image, cmap=cmap)
        ax.axis('off')

    suptitle = '{} epoch'.format(epoch)
    if label is not None:
        suptitle += '  label: {}'.format(label)
    fig.suptitle(suptitle)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)


def plot_loss(ax, train_loss, valid_loss):
    ax.plot(train_loss, label='train', alpha=0.8)
    ax.plot(valid_loss, label='valid', alpha=0.8)
    train_max = np.mean(train_loss) + 2 * np.std(train_loss)
    valid_max = np.mean(valid_loss) + 2 * np.std(valid_loss)
    y_max = max(train_max, valid_max)
    y_min = min(min(train_loss), min(valid_loss))
    ax.set_ylim(0.9 * y_min, 1.1 * y_max)
    ax.set_yscale('log')


def plot_losses(fig, train_loss, valid_loss,
                train_loss_mse, valid_loss_mse,
                train_loss_ssim, valid_loss_ssim,
                train_loss_kl, valid_loss_kl):
    ax = fig.add_subplot(411)
    plot_loss(ax, train_loss, valid_loss)
    ax.set_ylabel('Total Loss')
    ax.tick_params(bottom=False, labelbottom=False)
    ax.legend()

    ax = fig.add_subplot(412)
    plot_loss(ax, train_loss_mse, valid_loss_mse)
    ax.set_ylabel('Mean Squared Error')
    ax.tick_params(bottom=False, labelbottom=False)

    ax = fig.add_subplot(413)
    plot_loss(ax, train_loss_ssim, valid_loss_ssim)
    ax.set_ylabel('Structural Similarity')
    ax.tick_params(bottom=False, labelbottom=False)

    ax = fig.add_subplot(414)
    plot_loss(ax, train_loss_kl, valid_loss_kl)
    ax.set_ylabel('KL Divergence')
    ax.set_xlabel('epoch')

    fig.align_labels()


def plot_latent_traversal(fig, model, device, row, col=10, epoch=0, label=None,
                          label_transform=lambda x: x, image_size=64):
    gradation = np.linspace(-2, 2, col)
    z = np.zeros(shape=(row, col, row))
    for i in range(row):
        z[i, :, i] = gradation
    z = z.reshape(-1, row)
    z = torch.from_numpy(z.astype(np.float32)).to(device)

    if label is not None:
        label_transformed = label_transform(label)
        label_transformed = label_transformed.repeat(z.shape[0], 1).to(device)
    else:
        label_transformed = None

    images = model.decoder(z, image_size=image_size, label=label_transformed)
    images = formatImages(images)

    cmap = None
    channel = images.shape[3]
    if channel == 1:
        # cmap = 'gray'
        cmap = 'binary'
        images = np.squeeze(images)

    for i, image in enumerate(images):
        ax = fig.add_subplot(row, col, i + 1)
        ax.imshow(image, cmap=cmap)
        ax.axis('off')

    suptitle = '{} epoch'.format(epoch)
    if label is not None:
        suptitle += '  label: {}'.format(label)
    fig.suptitle(suptitle)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
