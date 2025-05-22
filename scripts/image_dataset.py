import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
# from tqdm import tqdm
from concurrent import futures
from PIL import Image
# import cv2
import numpy as np


class ImageDataset(Dataset):
    def __init__(
        self,
        datafolder: str,
        data_num: int = None,
        train: bool = True,
        split_ratio: float = 0.8,
        image_size: int = 64
    ):
        self.train = train
        self.image_size = image_size
        self.transform = transforms.Compose([
            # transforms.RandomRotation(180, fill=(1,)),
            transforms.RandomAffine(
                degrees=180, translate=(0.1, 0.1), fill=(1,)),
            # transforms.ColorJitter(
            #     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ])

        image_list = []
        label_list = []

        # folders = glob.glob(f'{datafolder}/*')
        labels = [
            'blue',
            # 'brown',
            # 'green',
            # 'orange',
            'pink',
            # 'red',
            # 'yellow',
        ]

        # for i, folder in enumerate(folders):
        for i, label in enumerate(labels):
            folder = os.path.join(datafolder, label)

            # paths = glob.glob('{}/color/*'.format(folder))
            paths = glob.glob(os.path.join(folder, '*.png'))
            # print(os.path.join(datafolder, '*.png'))
            # print(paths)

            train_data_num = int(split_ratio * len(paths))
            if train:
                paths = paths[:train_data_num]
            else:
                paths = paths[train_data_num:]

            if data_num is not None and data_num < len(paths):
                paths = paths[:data_num]

            print(f'loading {len(paths)} data from {folder}')
            # for image_folder_path in tqdm(paths):
            #     image_paths = glob.glob(
            #         os.path.join(image_folder_path, '*.png'))
            image = self._load_images(paths)
            image_list.extend(image)
            label_list.extend([i] * len(image))
            # label_list.extend([0] * len(image))

        self.label_dim = i + 1
        # self.label_dim = 1

        self.image = torch.stack(image_list)
        self.label = torch.tensor(label_list)

        print('image shape:', self.image.shape)
        print('label shape:', self.label.shape)

        image_data_size = self.image.detach().numpy().copy().__sizeof__()
        label_data_size = self.label.detach().numpy().copy().__sizeof__()
        print('image data size: {} [MiB]'.format(image_data_size / 1.049e+6))
        print('label data size: {} [MiB]'.format(label_data_size / 1.049e+6))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        # print(image.shape)
        if self.train:
            image = self.transform(image)
        return image, self.label[idx]

    def _load_images(self, image_paths):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
        ])

        def load_one_frame(idx):
            if not os.path.exists(image_paths[idx]):
                return
            # image = Image.open(image_paths[idx])

            # edge
            # image = cv2.imread(image_paths[idx], 0)
            # image = cv2.Canny(image, 35, 50)
            # kernel = np.ones((5, 5), np.uint8)
            # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

            # hsv
            image = cv2.imread(image_paths[idx], cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)[:, :, 2]
            ret2, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
            # kernel = np.ones((1, 1), np.uint8)
            # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

            image = transform(image)
            return idx, image

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
        return torch.stack(image_list)
