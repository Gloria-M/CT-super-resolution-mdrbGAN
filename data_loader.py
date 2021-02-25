import os
import numpy as np
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def read_trunc_slice(slice_name, in_dir):
    min_val = -1024
    max_val = 3027

    slice_path = os.path.join(in_dir, '{:s}.npy'.format(slice_name))
    slice_data = np.load(slice_path)

    slice_data[np.where(slice_data < min_val)] = min_val
    slice_data[np.where(slice_data > max_val)] = max_val

    return slice_data


def crop_train(lr_slice, hr_slice):

    crop_size = 96
    num_total_crops = 16
    full_h, full_w = lr_slice.shape
    crop_h, crop_w = (full_h - crop_size), (full_w - crop_size)

    h_idxs = random.sample(list(range(crop_h)), num_total_crops)
    w_idxs = random.sample(list(range(crop_w)), num_total_crops)

    lr_crops, hr_crops = [], []
    for crop_idx in range(num_total_crops):
        start_h, end_h = h_idxs[crop_idx], h_idxs[crop_idx] + crop_size
        start_w, end_w = w_idxs[crop_idx], w_idxs[crop_idx] + crop_size
        lr_crop = lr_slice[start_h:end_h, start_w:end_w]
        hr_crop = hr_slice[start_h:end_h, start_w:end_w]

        lr_crop, hr_crop = torch.tensor(lr_crop), torch.tensor(hr_crop)
        lr_crop, hr_crop = lr_crop.unsqueeze(0), hr_crop.unsqueeze(0)
        lr_crops.append(lr_crop)
        hr_crops.append(hr_crop)

    lr_crops, hr_crops = torch.stack(lr_crops), torch.stack(hr_crops)

    return lr_crops, hr_crops


def crop_test(lr_slice, hr_slice):
    crop_size = 128
    num_crops_h, num_crops_w = 4, 4

    lr_crops, hr_crops = [], []
    for h_idx in range(num_crops_h):
        for w_idx in range(num_crops_w):

            start_h, end_h = h_idx * crop_size, (h_idx + 1) * crop_size
            start_w, end_w = w_idx * crop_size, (w_idx + 1) * crop_size

            lr_crop = lr_slice[start_h:end_h, start_w:end_w]
            hr_crop = hr_slice[start_h:end_h, start_w:end_w]

            lr_crop, hr_crop = torch.tensor(lr_crop), torch.tensor(hr_crop)
            lr_crop, hr_crop = lr_crop.unsqueeze(0), hr_crop.unsqueeze(0)
            lr_crops.append(lr_crop)
            hr_crops.append(hr_crop)

    lr_crops, hr_crops = torch.stack(lr_crops), torch.stack(hr_crops)

    return lr_crops, hr_crops


class CTscansDataset(Dataset):

    def __init__(self, in_dir, ct_names, lr_transform, hr_transform, mode):

        self.ct_scans = ct_names
        self.in_dir = in_dir
        self.mode = mode

        self.ct_min = -1024
        self.ct_max = 3072

        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

    def __len__(self):

        return len(self.ct_scans)

    def __getitem__(self, idx):

        ct_name = self.ct_scans[idx]

        slice_data = read_trunc_slice(ct_name, self.in_dir)

        slice_data = torch.tensor(slice_data).float()
        slice_data = slice_data.unsqueeze(0)
        slice_data = (slice_data - self.ct_min) / (self.ct_max - self.ct_min)

        lr_ct = self.lr_transform(slice_data)
        lr_ct = lr_ct.squeeze()
        lr_ct = lr_ct.numpy()

        if self.hr_transform is not None:
            hr_ct = self.hr_transform(slice_data)
        else:
            hr_ct = slice_data
        hr_ct = hr_ct.squeeze()
        hr_ct = hr_ct.numpy()

        if self.mode == 'train':
            lr_crops, hr_crops = crop_train(lr_ct, hr_ct)
        else:
            lr_crops, hr_crops = crop_train(lr_ct, hr_ct)

        return lr_crops, hr_crops


def make_loader(in_dir, ct_names, mode='train'):
    ct_size = 512
    scale_factor = 4
    shuffle_batch = True if mode == 'train' else False

    lr_transform = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize(ct_size // scale_factor),
                                       transforms.Resize(ct_size, interpolation=Image.BICUBIC),
                                       transforms.ToTensor()])
    hr_transform = None

    dataset_ = CTscansDataset(in_dir, ct_names, lr_transform, hr_transform, mode)
    dataloader_ = DataLoader(dataset_, batch_size=1, drop_last=True, shuffle=shuffle_batch)

    return dataloader_
