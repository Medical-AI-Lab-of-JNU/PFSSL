import sys
import os
import random
import torch
import json
from torch.utils.data.dataset import Dataset, ConcatDataset, Subset, random_split
from copy import deepcopy
from data import transforms as T
from torchvision.transforms import *
import numpy as np
from PIL import Image
from skimage import io

# 联邦半监督方式中读取json数据


class LiverDataset(Dataset):
    def __init__(self, image_path=None, stage='train', image_size=256, is_augmentation=False, type='labeled', client_id=0, label_ratio=0.3):
        super(LiverDataset, self).__init__()
        self.image_size = image_size
        self.sep = '\\' if sys.platform[:3] == 'win' else '/'
        self.stage = stage
        self.is_augmentation = is_augmentation
        if self.stage == 'train':
            if label_ratio == 0.3:
                train_json_file = os.path.join(image_path, f'train.json')
            else:
                train_json_file = os.path.join(image_path, f'train-{label_ratio}.json')
            print(train_json_file)
            train_list = json.load(open(train_json_file, 'r'))
            data_list = train_list[f'site{client_id}'][type]
            image_file_list, mask_file_list = [], []
            for data in data_list:
                image_file = os.path.join(image_path, "image", data)
                mask_file = os.path.join(image_path, "mask", data.replace('.png', '_mask.png'))
                image_file_list.append(image_file)
                mask_file_list.append(mask_file)

            self.image_files = image_file_list
            self.label_files = mask_file_list
        else:
            test_list = json.load(open(os.path.join(image_path, 'test.json'), 'r'))
            data_list = test_list[f'site{client_id}'][type]
            image_file_list, mask_file_list = [], []
            for data in data_list:
                image_file = os.path.join(image_path, "image", data)
                mask_file = os.path.join(image_path, "mask", data.replace('.png', '_mask.png'))
                image_file_list.append(image_file)
                mask_file_list.append(mask_file)

            self.image_files = image_file_list
            self.label_files = mask_file_list
        # assert len(self.image_files) == len(self.label_files)
        if self.is_augmentation:
            self.augmentation = self.augmentation_transform()
        self.post_transform = self.post_transform()
        self.label_transform = self.label_transform()
        self.pre_transform = self.pre_transform()

    def __getitem__(self, item):
        # assert self.image_files[item].split(self.sep)[-1] == self.label_files[item].split(self.sep)[-1], f"{self.image_files[item]} != {self.label_files[item]}, {item}"

        image_file = self.image_files[item]
        label_file = image_file.replace('.png', '_mask.png').replace('image', 'mask')

        image = io.imread(image_file).astype('uint8')
        label = io.imread(label_file).astype('bool').astype('uint8')  # 255 --> 1
        img_name = os.path.basename(self.image_files[item]).split(".")[0]
        image = Image.fromarray(image).convert('RGB').resize((self.image_size, self.image_size))
        label = Image.fromarray(label).convert('L').resize((self.image_size, self.image_size))

        if self.stage == 'train':
            image, label = self.pre_transform(image, label)
            imageA1, imageA2 = deepcopy(image), deepcopy(image)
            imageA1, _ = self.augmentation(imageA1, label)
            imageA2, _ = self.augmentation(imageA1, label)
            image, label = self.post_transform(image), self.label_transform(label)
            imageA1 = self.post_transform(imageA1)
            imageA2 = self.post_transform(imageA2)
            label = torch.from_numpy(np.array(label)).unsqueeze(0)
            return image, label, imageA1, imageA2, img_name
        elif self.stage == 'test':
            image = self.post_transform(image)
            label = torch.from_numpy(np.array(label)).unsqueeze(0)
        else:
            image, label = self.post_transform(image), self.label_transform(label)
            label = torch.from_numpy(np.array(label)).unsqueeze(0)
        return image, label, img_name

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def augmentation_transform():
        return T.Compose([
            T.ColorJitter(0.5, 0.5, 0.5, 0.25),
            T.RandomPosterize(bits=5, p=0.2),
            T.RandomAutocontrast(p=0.2),
            T.RandomEqualize(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    def pre_transform(self):
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=180),
        ])

    def post_transform(self):
        return Compose([
            Resize([self.image_size, self.image_size], InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
        ])

    def label_transform(self):
        return Compose([
            Resize([self.image_size, self.image_size], InterpolationMode.NEAREST)
        ])