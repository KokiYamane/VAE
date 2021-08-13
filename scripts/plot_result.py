import numpy as np
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
        cmap = 'gray'

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
    ax = fig.add_subplot(111)
    # pca = PCA()
    # pca.fit(zs)
    # points = pca.transform(zs)
    # points = TSNE(n_components=2, random_state=0).fit_transform(zs)
    points = zs
    im = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='jet', alpha=0.6)
    lim = np.max(np.abs(points)) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, ax=ax, orientation='vertical', cax=cax)
    ax.set_title('{} epoch'.format(epoch))


def plot_generated_image(fig, model, device, col=10, epoch=0):
    row = col

    x = np.tile(np.linspace(-2, 2, col), row)
    y = np.repeat(np.linspace(2, -2, row), col)
    z = np.stack([x, y]).transpose()
    z = torch.from_numpy(z.astype(np.float32)).to(device)

    images = model.decode(z)
    images = formatImages(images)

    cmap = None
    channel = images.shape[3]
    if channel == 1:
        cmap = 'gray'

    for i, image in enumerate(images):
        ax = fig.add_subplot(row, col, i + 1)
        ax.imshow(image, cmap=cmap)
        ax.axis('off')

    fig.suptitle('{} epoch'.format(epoch))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
