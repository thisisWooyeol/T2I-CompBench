# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import os

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform

        data_imgs = os.listdir(data_path)
        # filename pattern: f"{idx}_{itr}_{level}_" + prompt.replace(" ", "_").jpg
        data_imgs.sort(key=lambda x: int(x.split("_")[0]))  # sort by idx (first part)
        self.data_list = [os.path.join(data_path, data) for data in data_imgs]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path = self.data_list[index]
        image = Image.open(image_path).convert("RGB")
        true_img_size = image.size

        image = self.transform(image)
        image = np.array(image)[:, :, ::-1]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        img_size = image.shape
        return {
            "image": image,
            "height": img_size[1],
            "width": img_size[2],
            "true_height": true_img_size[0],
            "true_width": true_img_size[1],
            "image_path": image_path,
        }


def collate_fn(batch):
    image_list = []
    for image in batch:
        image_list.append(image)
    return image_list
