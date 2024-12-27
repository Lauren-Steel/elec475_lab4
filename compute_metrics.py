import time
import torch
import numpy as np
import math


# Function to calculate mIoU
def compute_miou(preds, targets, num_classes=21):
    # Convert probabilities to predicted class
    pred = torch.argmax(preds, dim=1)
    iou_list = []
    for i in range(num_classes):
        pred_inds = (pred == i)
        target_inds = (targets == i)
        intersection = (pred_inds & target_inds).sum().float().item()
        union = (pred_inds | target_inds).sum().float().item()
        if union == 0:
            iou_list.append(float('nan'))  # If there is no ground truth, do not include this class in average
        else:
            iou_list.append(intersection / union)
    valid_iou = [v for v in iou_list if not math.isnan(v)]
    mean_iou = sum(valid_iou) / len(valid_iou) if len(valid_iou) > 0 else 0
    return mean_iou



# Function to measure inference speed
def measure_inference_speed(model, dataloader, device):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)['out']
    total_time = time.time() - start_time
    return (total_time / len(dataloader.dataset)) * 1000  # ms per image


# Function to evaluate mIoU
def evaluate_miou(model, dataloader, device):
    model.eval()
    total_miou = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)['out']
            preds = outputs.argmax(dim=1)
            for i in range(len(images)):
                total_miou += compute_miou(preds[i].cpu(), targets[i].cpu())
    return total_miou / len(dataloader)


# Function to compute model parameters
def compute_model_params(model):
    return sum(p.numel() for p in model.parameters())


# Main to compute metrics
def compute_metrics(model, dataloader, device):
    miou = evaluate_miou(model, dataloader, device)
    speed = measure_inference_speed(model, dataloader, device)
    params = compute_model_params(model)
    return miou, speed, params