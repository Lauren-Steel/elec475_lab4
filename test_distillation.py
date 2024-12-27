import torch
import numpy as np
import matplotlib.pyplot as plt

from dataloader import get_dataloader
from seg_model import CompactUNet  # Ensure your student model is imported
from torchvision.transforms.functional import to_pil_image

def load_model(model_path, device):
    model = CompactUNet(num_classes=21).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def test_model(model, test_loader, device):
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for images, true_masks in test_loader:
            images = images.to(device)
            true_masks = true_masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # Convert probabilities to class labels
            
            all_preds.append(preds.cpu())
            all_gts.append(true_masks.cpu())
    
    return all_preds, all_gts

def calculate_iou(preds, gts, num_classes):
    ious = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (gts == cls)).sum()
        union = ((preds == cls) | (gts == cls)).sum()
        if union == 0:
            ious.append(float('nan'))  # avoid division by zero
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)  # Mean IoU ignoring NaNs

def visualize_segmentation(image, pred_mask, gt_mask):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(to_pil_image(image))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask, cmap='viridis')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    axes[2].imshow(gt_mask, cmap='viridis')
    axes[2].set_title('Ground Truth Mask')
    axes[2].axis('off')
    
    plt.show()

# Usage example
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = load_model('results/model/student_model.pth', device)
# test_loader = get_dataloader('data/VOC/VOCdevkit', 'test', batch_size=1, shuffle=False)

# preds, gts = test_model(model, test_loader, device)
# mIoU = calculate_iou(torch.cat(preds), torch.cat(gts), num_classes=21)
# print(f"Mean IoU: {mIoU}")

# # Visualize the first test image results
# visualize_segmentation(test_loader.dataset[0][0], preds[0][0], gts[0][0])

test_loader = get_dataloader('data/VOC/VOCdevkit', 'val', batch_size=1, shuffle=False)

# Assuming the rest of the code remains the same
preds, gts = test_model(model, test_loader, device)
mIoU = calculate_iou(torch.cat(preds), torch.cat(gts), num_classes=21)
print(f"Mean IoU: {mIoU}")

# For visualization, ensure that you have data available
if len(test_loader) > 0:
    first_image, first_target = next(iter(test_loader))
    visualize_segmentation(first_image[0], preds[0][0], gts[0][0])