import os
import torchvision.datasets as datasets
from torchvision import transforms

def download_voc(root_dir='/content/drive/MyDrive/ColabNotebooks/lab4/data', year='2012'):
    from torchvision.datasets import VOCSegmentation

    # Ensure the root directory exists
    os.makedirs(root_dir, exist_ok=True)

    # Download train split
    print("Downloading train split...")
    VOCSegmentation(root=root_dir, year=year, image_set='train', download=True)

    # Download val split
    print("Downloading val split...")
    VOCSegmentation(root=root_dir, year=year, image_set='val', download=True)

    print("Dataset successfully prepared.")


if __name__ == "__main__":
    # Default settings for downloading the dataset
    download_voc()
