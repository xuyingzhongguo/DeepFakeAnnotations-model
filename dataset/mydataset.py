"""
Author: Honggu Liu

"""

from PIL import Image
from torch.utils.data import Dataset
import os
import random
import cv2
import torchvision.transforms as transforms
import numpy as np


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # RGB
        img = Image.open(fn).convert('RGB')
        # grey
        # img = Image.open(fn).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


class MyDataset_test(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split(' ')
            if len(words) == 3:
                words[0] = words[0] + ' ' + words[1]
                words[1] = words[2]
                words.pop(2)

            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # RGB
        img = Image.open(fn).convert('RGB')
        # grey
        # img = Image.open(fn).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        label = int(label)

        return [img, label, fn]

    def __len__(self):
        return len(self.imgs)


class MyDataset_lap(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # RGB
        img = cv2.imread(fn)
        img = cv2.Laplacian(img, cv2.CV_64F)
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        # grey
        # img = Image.open(fn).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


# import pywt
# class MyDataset_wavelet(Dataset):
#     def __init__(self, txt_path, transform=None, target_transform=None):
#         fh = open(txt_path, 'r')
#         imgs = []
#         for line in fh:
#             line = line.rstrip()
#             words = line.split()
#             imgs.append((words[0], int(words[1])))
#
#         self.imgs = imgs
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, index):
#         fn, label = self.imgs[index]
#         # RGB
#         img = cv2.imread(fn)
#         img = cv2.resize(img, (1280, 1280))
#         wave = pywt.dwt2(img, 'haar', mode='periodization')
#
#         cA, (cH, cV, cD) = wave
#         waveplus = np.empty([cA.shape[0], cA.shape[1], 3])
#         waveplus[:, :, 0] = cH[:, :, 0]
#         waveplus[:, :, 1] = cV[:, :, 0]
#         waveplus[:, :, 2] = cD[:, :, 0]
#         waveplus = Image.fromarray(np.uint8(waveplus)).convert('RGB')
#
#         if self.transform is not None:
#             img = self.transform(waveplus)
#
#         return img, label
#
#     def __len__(self):
#         return len(self.imgs)

