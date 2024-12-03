import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from seg_model import CompactSegmentationModel  # Import model
from dataloader import get_dataloader  # Import dataloader
import matplotlib.pyplot as plt
from pretrained_teacher import load_pretrained_teacher


def distillation_loss(student_logits, teacher_logits, targets, alpha, beta, tau):
    """
    Computes the combined loss for knowledge distillation.
    """
    # Cross-Entropy Loss with ground truth
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)(student_logits, targets)

    # Knowledge Distillation Loss (KL Divergence)
    student_soft = F.log_softmax(student_logits / tau, dim=1)
    teacher_soft = F.softmax(teacher_logits / tau, dim=1)
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (tau ** 2)

    # Combine losses
    total_loss = alpha * ce_loss + beta * kd_loss
    return total_loss, ce_loss.item(), kd_loss.item()


def plot_loss(training_losses, validation_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    print(f"Loss plot saved to: {save_path}")


def train_model(data_path, batch_size, num_epochs, learning_rate, alpha, beta, tau, num_classes=21, device='cpu'):
    # Initialize models
    student_model = CompactSegmentationModel(num_classes).to(device)
    teacher_model = load_pretrained_teacher(device)  # Load teacher model

    optimizer = Adam(student_model.parameters(), lr=learning_rate)

    # Load data
    train_loader = get_dataloader(data_path, 'train', batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(data_path, 'val', batch_size=batch_size, shuffle=False)

    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        student_model.train()
        epoch_train_loss = 0
        epoch_ce_loss = 0
        epoch_kd_loss = 0

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for images, targets in tqdm(train_loader, desc="Training", unit="batch"):
            images, targets = images.to(device), targets.to(device)

            # Forward pass through student model
            student_logits = student_model(images)

            # Resize logits if dimensions mismatch
            if student_logits.shape[-2:] != targets.shape[-2:]:
                student_logits = F.interpolate(student_logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

            # Forward pass through teacher model (no grad)
            with torch.no_grad():
                teacher_logits = teacher_model(images)['out']

            # Compute distillation loss
            loss, ce_loss, kd_loss = distillation_loss(student_logits, teacher_logits, targets, alpha, beta, tau)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_ce_loss += ce_loss
            epoch_kd_loss += kd_loss

        avg_training_loss = epoch_train_loss / len(train_loader)
        avg_ce_loss = epoch_ce_loss / len(train_loader)
        avg_kd_loss = epoch_kd_loss / len(train_loader)
        training_losses.append(avg_training_loss)

        print(f"Training Loss: {avg_training_loss:.4f}, CE Loss: {avg_ce_loss:.4f}, KD Loss: {avg_kd_loss:.4f}")

        # Validation loop
        student_model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating", unit="batch"):
                images, targets = images.to(device), targets.to(device)
                outputs = student_model(images)

                if outputs.shape[-2:] != targets.shape[-2:]:
                    outputs = F.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)

                loss = nn.CrossEntropyLoss(ignore_index=255)(outputs, targets)
                epoch_val_loss += loss.item()

        avg_validation_loss = epoch_val_loss / len(val_loader)
        validation_losses.append(avg_validation_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_validation_loss:.4f}")

    # Save model and plot losses
    save_path = '/content/drive/MyDrive/ColabNotebooks/lab4/results/model/distillation_model.pth'
    torch.save(student_model.state_dict(), save_path)
    print(f"Student model saved to: {save_path}")

    plot_path = '/content/drive/MyDrive/ColabNotebooks/lab4/results/plots/distillation_loss_plot.png'
    plot_loss(training_losses, validation_losses, plot_path)


def main():
    parser = argparse.ArgumentParser(description="Train a compact segmentation model with knowledge distillation.")
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ColabNotebooks/lab4/data/VOCdevkit', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for optimizer')
    parser.add_argument('--alpha', type=float, default=0.9, help='Weight for cross-entropy loss')
    parser.add_argument('--beta', type=float, default=0.1, help='Weight for distillation loss')
    parser.add_argument('--tau', type=float, default=2.0, help='Temperature for distillation')
    parser.add_argument('--num_classes', type=int, default=21, help='Number of classes in dataset')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (e.g., "cpu", "cuda", "mps")')

    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        beta=args.beta,
        tau=args.tau,
        num_classes=args.num_classes,
        device=torch.device(args.device)
    )


if __name__ == '__main__':
    main()
