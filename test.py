import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from css_model import CompactSemanticSegmentationModel
from compute_metrics import compute_metrics, compute_miou


# Function to save sample images
def save_samples(images, predictions, targets, save_dir, prefix, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(min(len(images), num_samples)):
        plt.figure(figsize=(12, 4))

        # Original image
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(images[idx].permute(1, 2, 0).cpu().numpy())
        plt.axis("off")

        # Predicted mask
        plt.subplot(1, 3, 2)
        plt.title("Predicted Mask")
        plt.imshow(predictions[idx].cpu().numpy(), cmap="gray")
        plt.axis("off")

        # Ground truth mask
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth")
        plt.imshow(targets[idx].cpu().numpy(), cmap="gray")
        plt.axis("off")

        # Save the figure
        save_path = os.path.join(save_dir, f"{prefix}_sample_{idx + 1}.png")
        plt.savefig(save_path)
        plt.close()


# Evaluate model and save visualizations
def evaluate_and_visualize(model, dataloader, device, save_dir, num_samples=5):
    model.eval()
    successful_samples = []
    unsuccessful_samples = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)['out']
            preds = outputs.argmax(dim=1)

            # Save successful and unsuccessful samples based on mIoU
            for i in range(len(images)):
                miou = compute_miou(preds[i].cpu(), targets[i].cpu())
                if miou > 0.5:  # Threshold for success
                    successful_samples.append((images[i], preds[i], targets[i]))
                else:
                    unsuccessful_samples.append((images[i], preds[i], targets[i]))

    # Save visualizations
    save_samples(
        [img for img, _, _ in successful_samples],
        [pred for _, pred, _ in successful_samples],
        [target for _, _, target in successful_samples],
        save_dir,
        prefix="success",
        num_samples=num_samples,
    )
    save_samples(
        [img for img, _, _ in unsuccessful_samples],
        [pred for _, pred, _ in unsuccessful_samples],
        [target for _, _, target in unsuccessful_samples],
        save_dir,
        prefix="failure",
        num_samples=num_samples,
    )


# Main script
if __name__ == "__main__":
    # Load dataset
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])

    target_transform = Compose([
        Resize((224, 224)),
        Lambda(lambda x: torch.as_tensor(np.array(x), dtype=torch.int64)),
    ])

    val_dataset = VOCSegmentation(
        root='./data',
        year='2012',
        image_set='val',
        download=False,
        transform=transform,
        target_transform=target_transform,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Load models
    css_model_without_distillation = CompactSemanticSegmentationModel(num_classes=21)
    css_model_with_distillation = CompactSemanticSegmentationModel(num_classes=21)

    # Load pre-trained weights
    css_model_without_distillation.load_state_dict(torch.load('results/models/custom_model.pth'))
    css_model_with_distillation.load_state_dict(torch.load('results/models/final_distilled_model_best.pth'))

    # Device configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    css_model_without_distillation = css_model_without_distillation.to(device)
    css_model_with_distillation = css_model_with_distillation.to(device)

    # Compute metrics for models
    print("Evaluating CSS Model Without Distillation...")
    css_no_dist_miou, css_no_dist_speed, css_no_dist_params = compute_metrics(css_model_without_distillation, val_loader, device)
    print(f"Without Distillation - mIoU: {css_no_dist_miou:.4f}, Params: {css_no_dist_params}, Speed: {css_no_dist_speed:.2f} ms/image")

    print("Evaluating CSS Model With Distillation...")
    css_with_dist_miou, css_with_dist_speed, css_with_dist_params = compute_metrics(css_model_with_distillation, val_loader, device)
    print(f"With Distillation - mIoU: {css_with_dist_miou:.4f}, Params: {css_with_dist_params}, Speed: {css_with_dist_speed:.2f} ms/image")

    # Save visualizations
    evaluate_and_visualize(css_model_without_distillation, val_loader, device, "results/samples/css_without_distillation", num_samples=5)
    evaluate_and_visualize(css_model_with_distillation, val_loader, device, "results/samples/css_with_distillation", num_samples=5)
