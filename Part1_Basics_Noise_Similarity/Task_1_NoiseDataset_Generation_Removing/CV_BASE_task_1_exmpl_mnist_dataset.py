import numpy as np
import struct
from array import array
from torch.utils.data import Dataset
from typing import Tuple
import torch


class MnistDataset(Dataset):
    """MNIST dataset"""

    def __init__(self, images_filepath: str, labels_filepath: str):
        self.image_paths = images_filepath
        self.label_paths = labels_filepath

        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError("Magic number mismatch, expected 2049, got {}".format(magic))
            labels = array("B", file.read())
        self.labels = labels
        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError("Magic number mismatch, expected 2051, got {}".format(magic))
            self.image_data = array("B", file.read())
        self.rows = rows
        self.cols = cols
        self.size = int(size // (rows * cols))

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # image, target label
        raw_data = np.array(self.image_data[idx * self.rows * self.cols : (idx + 1) * self.rows * self.cols])
        img = raw_data.reshape(28, 28)
        img = np.array([img, img, img]).transpose(1, 2, 0)

        return (
            img,
            self.labels[idx],
        )
