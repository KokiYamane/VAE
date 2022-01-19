import unittest
import torch

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.SSIM import SSIMLoss


class TestSSIM(unittest.TestCase):
    def test_ssim(self):
        print('\n---------- test SSIM Loss ----------')
        loss_fn = SSIMLoss()
        # image1 = torch.rand(1, 3, 128, 128)
        # image2 = torch.rand(1, 3, 128, 128)
        image1 = torch.rand(1, 1, 128, 128)
        image2 = torch.rand(1, 1, 128, 128)
        image1_noise = image1 + 0.1 * torch.randn_like(image1)
        print('Some image:', loss_fn(image1, image1))
        print('Another image:', loss_fn(image1, image2))
        print('Noise:', loss_fn(image1, image1_noise))


if __name__ == "__main__":
    unittest.main()
