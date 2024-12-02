
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import functional as TF
import numpy as np


class SegmentationTransform:
    def __init__(self, resize=None, crop_size=None, normalize=None, h_flip=False):
        self.resize = resize
        self.crop_size = crop_size
        self.normalize = normalize
        self.h_flip = h_flip

    def __call__(self, img, target):
        if self.resize:
            img = TF.resize(img, self.resize)
            target = TF.resize(target, self.resize, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        if self.h_flip and torch.rand(1) < 0.5:
            img = TF.hflip(img)
            target = TF.hflip(target)

        if self.crop_size:
            crop_params = torchvision.transforms.RandomCrop.get_params(img, output_size=self.crop_size)
            img = TF.crop(img, *crop_params)
            target = TF.crop(target, *crop_params)

        target = torch.as_tensor(np.array(target), dtype=torch.long)
        img = TF.to_tensor(img)
        if self.normalize:
            img = TF.normalize(img, mean=self.normalize['mean'], std=self.normalize['std'])

        return img, target


def get_dataloader(data_path, split, batch_size, shuffle):
    transform = SegmentationTransform(
        resize=(256, 256),
        crop_size=(224, 224),
        normalize={'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
        h_flip=(split == 'train')
    )

    dataset = VOCSegmentation(
        root=data_path,
        year='2012',
        image_set=split,
        download=False,
        transforms=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    return dataloader
