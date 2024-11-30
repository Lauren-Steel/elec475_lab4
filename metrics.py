import numpy as np
import torch

def compute_miou(pred, target, num_classes=21):
    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = torch.sum(pred_mask & target_mask).item()
        union = torch.sum(pred_mask | target_mask).item()
        if union == 0:  # Avoid division by zero
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    return np.mean(ious)
