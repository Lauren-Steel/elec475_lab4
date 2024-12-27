import os
import torch
import numpy as np
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloader(root, image_set, batch_size=16, shuffle=True, drop_last=True):
    def transform_image_and_target(image, target):
        # Image transformation and target
        image_transform = transforms.Compose([
            transforms.Resize(256),  # Resize images to a fixed size
            transforms.CenterCrop(256),  # Crop images to a fixed size
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        target_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),  # NEAREST is used to avoid introducing new classes
            transforms.CenterCrop(256)
        ])

        image = image_transform(image)

        # Convert target to tensor
        target = target_transform(target)
        target = torch.tensor(np.array(target), dtype=torch.long)
        return image, target

    # Load dataset
    dataset = VOCSegmentation(
        root=root,
        year='2012',
        image_set=image_set,
        transforms=lambda img, tgt: transform_image_and_target(img, tgt),  # Ensure both img and tgt are passed
        download=True
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader
