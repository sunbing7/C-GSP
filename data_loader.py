from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
from tqdm import tqdm
import random

import h5py
import pickle
import copy
import ast

import os
from PIL import Image


class CustomDataSet(Dataset):

    def __init__(self, data_file, target=150, transform=False):
        self.data_file = data_file
        self.transform = transform
        self.y = []

        dataset = np.load(data_file).transpose()
        dataset = np.transpose(dataset, (3, 1, 0, 2))

        self.x = dataset

        for i in range(0, len(self.x)):
            self.y.append(target)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
