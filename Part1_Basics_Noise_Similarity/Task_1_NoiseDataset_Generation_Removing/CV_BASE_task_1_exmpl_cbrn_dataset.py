from __future__ import annotations

import numpy as np
import cv2
import copy
from typing import Tuple
import random
import torch
from torch.utils.data import Dataset


class ChessBoardRandNumsDataset(Dataset):
    """Chess board dataset with MNIST squares as defects"""

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        dataset_size: int,
        noise_dataset: Dataset,  # "mnist"
    ):
        self.image_size = img_size
        self.ps = patch_size
        self.img_base = np.zeros((self.ps[0] * 8, self.ps[1] * 8, 3), dtype=np.uint8)
        self.dataset_size = dataset_size
        self.noise_images = noise_dataset

        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    self.img_base[i * self.ps[0] : (i + 1) * self.ps[0], j * self.ps[1] : (j + 1) * self.ps[1], :] = (
                        np.ones((self.ps[0], self.ps[1], 3), dtype=np.uint8) * 255
                    )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:  # image, 17-class target 1:
        img = copy.deepcopy(self.img_base)

        count = random.randint(0, 11)  # count of defects

        # label list
        # [0] - 0/1 - not rejected/rejected image
        # [1:17] - 0/1 - note defected/defected image patch
        labels = [0 for n in range(17)]
        plus_i = 1

        # if defect count > 5, the image is rejected
        if count > 5:
            labels[0] = 1
        for i in range(count):
            # randomly choose defect image from noise dataset
            img_num, _ = self.noise_images[random.randint(0, len(self.noise_images) - 1)]
            img_num = cv2.resize(img_num, dsize=(self.ps[0], self.ps[1]))

            # randomly choose defect location
            i = random.randint(0, 7)
            j = random.randint(0, 7)

            idx = i // 2 * 4 + j // 2
            labels[idx + plus_i] = 1

            img[i * self.ps[0] : (i + 1) * self.ps[0], j * self.ps[1] : (j + 1) * self.ps[1], :] = img_num

        img = cv2.resize(img, dsize=(self.image_size[0], self.image_size[1]))
        img = img.astype(float)
        img /= 255  # normalize data
        return (
            torch.FloatTensor(img),
            torch.FloatTensor(np.array(labels)),
        )
