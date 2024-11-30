import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import matplotlib.pyplot as plt
import numpy as np

from pretrained_fcn_resnet import load_pretrained_fcn_resnet  # Import the model-loading function
from metrics import compute_miou  # Custom function for mIoU
from visualizations import visualize_sample  # Visualization utility

# Load PASCAL VOC 2012 Dataset
def load_dataset():
    def transform_image_and_target(image, target):
        # Transform the image
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        image = image_transform(image)

        # Transform the target (segmentation mask)
        target = torch.tensor(np.array(target), dtype=torch.long)
        return image, target

    # Use the custom transform for VOCSegmentation
    dataset = VOCSegmentation(
        root='/content/drive/MyDrive/ColabNotebooks/lab4/data',
        year='2012',
        image_set='val',
        download=False,
        transforms=transform_image_and_target  # Pass the custom transform
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


# Evaluate the model
def evaluate_model(model, dataloader, device):
    total_miou = 0
    num_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            # Calculate mIoU
            miou = compute_miou(preds, targets)
            total_miou += miou
            num_samples += 1

            # Save visualizations for the first 5 samples
            if num_samples <= 5:
                visualize_sample(images[0], preds[0], targets[0], sample_id=num_samples)

    avg_miou = total_miou / num_samples
    print(f"Average mIoU: {avg_miou:.4f}")



if __name__ == '__main__':
    # Load pre-trained model
    model = load_pretrained_fcn_resnet()  # Modularized model loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataloader = load_dataset()

    # Evaluate model
    evaluate_model(model, dataloader, device)

