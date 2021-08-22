import torch
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
import numpy as np
import os
from concurrent import futures
from PIL import Image
import re


class ImageDataset(Dataset):
    def __init__(self, datafolder, data_num=None, train=True, split_ratio=0.8, image_size=64):
        self.image_size = image_size
        image_list = []
        label_list = []

        folders = glob.glob('{}/*'.format(datafolder))
        for i, folder in enumerate(folders):
            paths = glob.glob('{}/motion/*.csv'.format(folder))
            filenames = [os.path.splitext(os.path.basename(path))[0] for path in paths]
            filenums = [int(re.sub(r"\D", "", filename)) for filename in filenames]

            train_data_num = int(split_ratio * len(filenums))
            if train:
                filenums = filenums[:train_data_num]
            else:
                filenums = filenums[train_data_num:]

            if data_num != None and data_num < len(filenums):
                filenums = filenums[:data_num]

            print('loading {} data'.format(len(filenums)))
            for filenum in tqdm(filenums):
                image_folder_path = '{}/color/video_rgb{}'.format(folder, filenum)
                image_paths = glob.glob(os.path.join(image_folder_path, '*.jpg'))
                image = self._load_images(image_paths)
                image_list.extend(image)
                label_list.extend([i] * len(image))

        self.image = np.array(image_list).astype(np.float32)

        print('image data size: {} [MiB]'.format(self.image.__sizeof__()/1.049e+6))

        self.image = self.image.transpose(0, 3, 1, 2)
        self.image = self.image / 256

        self.image = torch.from_numpy(self.image)
        self.label = torch.tensor(label_list)

        print('image shape:', self.image.shape)
        print('label shape:', self.label.shape)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]

        # brightness data augmentation
        bias = 0.2 * torch.randn(1)
        mean = image.mean()
        bias = bias.clip(0.2 - mean, 0.8 - mean)
        image = image + bias
        image = image.clip(0, 1)

        return image, self.label[idx]

    def _crop_center(self, pil_img, crop_width, crop_height):
        img_width, img_height = pil_img.size
        return pil_img.crop(((img_width - crop_width) // 2,
                            (img_height - crop_height) // 2,
                            (img_width + crop_width) // 2,
                            (img_height + crop_height) // 2))

    def _load_images(self, image_paths):
        def load_one_frame(idx):
            if not os.path.exists(image_paths[idx]):
                return
            image = Image.open(image_paths[idx])
            image = self._crop_center(image, min(image.size), min(image.size))
            image = image.resize((self.image_size, self.image_size))
            return idx, np.array(image)

        image_list = []
        length = len(image_paths)
        image_list = [0] * length
        with futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_images = [
                executor.submit(
                    load_one_frame,
                    idx) for idx in range(length)]
            for future in futures.as_completed(future_images):
                idx, image = future.result()
                image_list[idx] = image
        return image_list
