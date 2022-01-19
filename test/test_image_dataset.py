import unittest
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.image_dataset import ImageDataset
from scripts.fastdataloader import FastDataLoader
from dataset_path import datafolder


class TestImageDataset(unittest.TestCase):
    def test_dataset(self):
        print('\n========== test fast dataset and fast data loader ==========')
        dataset = ImageDataset(datafolder, data_num=50, image_size=128)
        print('data length:', len(dataset))
        dataloader = FastDataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
        )
        print('\n---------- my data loader test ----------')
        for e in range(3):
            start = time.time()
            for i, (image, label) in enumerate(tqdm(dataloader)):
                pass
                # print('#', i)
                # print('shape:', image.shape)
            end = time.time()
            print('elapsed time:', end - start)

        torchdataloader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            num_workers=8,
            # pin_memory=True,
        )
        print('\n---------- pytorch data loader test ----------')
        for e in range(3):
            start = time.time()
            for i, (image, label) in enumerate(tqdm(torchdataloader)):
                pass
                # print('#', i)
                # print('shape:', image.shape)
            end = time.time()
            print('elapsed time:', end - start)


if __name__ == "__main__":
    unittest.main()
