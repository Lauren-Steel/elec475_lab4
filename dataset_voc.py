# dataset download

import os
import torchvision.datasets as datasets

def download_voc(root_dir='data/VOC', year='2012'):
    """Download the PASCAL VOC dataset to a local directory."""
    os.makedirs(root_dir, exist_ok=True)

    # Download train split
    print("Downloading train split...")
    datasets.VOCSegmentation(root=root_dir, year=year, image_set='train', download=True)

    # Download val split
    print("Downloading val split...")
    datasets.VOCSegmentation(root=root_dir, year=year, image_set='val', download=True)

    print("Dataset successfully prepared.")


if __name__ == "__main__":
    # Default settings for downloading the dataset
    download_voc()