import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pretrained_fcn_resnet import load_pretrained_fcn_resnet  # Import the model-loading function
from metrics import compute_miou  # Custom function for mIoU
from visualizations import visualize_sample  # Visualization utility
from dataloader import get_dataloader


# Evaluate the model
def evaluate_model(model, dataloader, device):
    total_miou = 0
    num_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating model"):
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
                visualize_sample(images[0].cpu(), preds[0].cpu(), targets[0].cpu(), sample_id=num_samples)
                # visualize_sample(images[0], preds[0], targets[0], sample_id=num_samples)

    avg_miou = total_miou / num_samples
    print(f"Average mIoU: {avg_miou:.4f}")



if __name__ == '__main__':
    # Load pre-trained model
    model = load_pretrained_fcn_resnet()  # Modularized model loading
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load dataset
    dataloader = get_dataloader('data/VOC/VOCdevkit', 'val', batch_size=1, shuffle=False)

    # Evaluate model
    evaluate_model(model, dataloader, device)