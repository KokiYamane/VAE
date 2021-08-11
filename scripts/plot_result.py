import numpy as np
import seaborn as sns
sns.set()


def torch2numpy(tensor):
    return tensor.cpu().detach().numpy().copy()


def formatImages(image):
    image = torch2numpy(image).astype(np.float32)
    image = image.transpose(0, 2, 3, 1)
    return image


def plot_generated_image(fig, images_ans, images_hat):
    col = 4
    images_ans = images_ans[:16]
    images_hat = images_hat[:16]

    row = -(-len(images_ans) // col)
    for i, (image_ans, image_hat) in enumerate(zip(images_ans, images_hat)):
        ax = fig.add_subplot(row, 2 * col, 2 * i + 1)
        ax.imshow(image_ans)
        ax.axis('off')
        ax.set_title('$i_{' + str(i) + '}$')

        ax = fig.add_subplot(row, 2 * col, 2 * i + 2)
        ax.imshow(image_hat)
        ax.axis('off')
        ax.set_title(r'$\hat{i}_{' + str(i) + '}$')
