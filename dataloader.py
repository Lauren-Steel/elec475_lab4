import os
import torch
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloader(root, image_set, batch_size=8, shuffle=True):
    def transform_image_and_target(image, target):
        # Image transformation
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        image = image_transform(image)

        # Convert target to tensor
        target = torch.tensor(np.array(target), dtype=torch.long)
        return image, target

    # Load dataset
    dataset = VOCSegmentation(
        root=root,
        year='2012',
        image_set=image_set,
        transforms=transform_image_and_target
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
