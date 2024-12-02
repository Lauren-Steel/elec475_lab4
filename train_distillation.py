import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models.segmentation import fcn_resnet50
from seg_model import CompactSegmentationModel  # Your student model
from dataloader import get_dataloader  # DataLoader function
import matplotlib.pyplot as plt


def distillation_loss(student_logits, teacher_logits, targets, alpha, beta, tau):
    """
    Computes the combined distillation loss.
    """
    ce_loss = F.cross_entropy(student_logits, targets, ignore_index=255)

    student_soft = F.log_softmax(student_logits / tau, dim=1)
    teacher_soft = F.softmax(teacher_logits / tau, dim=1)
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (tau * tau)

    total_loss = alpha * ce_loss + beta * kd_loss
    return total_loss


def plot_loss(train_losses, val_losses, save_path):
    """
    Plots training and validation losses.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Loss plot saved to: {save_path}")


def evaluate(student_model, val_loader, device):
    """
    Evaluates the model on the validation set and computes validation loss.
    """
    student_model.eval()
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating", unit="batch"):
            images, targets = images.to(device), targets.to(device)
            student_logits = student_model(images)

            if student_logits.shape[-2:] != targets.shape[-2:]:
                student_logits = F.interpolate(student_logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

            val_loss += criterion(student_logits, targets).item()

    return val_loss / len(val_loader)


def train_with_distillation(student_model, teacher_model, train_loader, val_loader, device, num_epochs, alpha, beta, tau):
    """
    Trains the student model using distillation.
    """
    student_model.to(device)
    teacher_model.to(device)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4, weight_decay=1e-5)
    alpha = 0.6
    beta = 0.4
    tau = 3.0
    train_losses = []
    val_losses = []

    teacher_model.eval()

    for epoch in range(num_epochs):
        student_model.train()
        epoch_train_loss = 0.0

        print(f"Epoch {epoch + 1}/{num_epochs}")
        for images, targets in tqdm(train_loader, desc="Training", unit="batch"):
            images, targets = images.to(device), targets.to(device)

            student_logits = student_model(images)
            with torch.no_grad():
                teacher_logits = teacher_model(images)["out"]

            if student_logits.shape[-2:] != targets.shape[-2:]:
                student_logits = F.interpolate(student_logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

            loss = distillation_loss(student_logits, teacher_logits, targets, alpha, beta, tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))
        val_loss = evaluate(student_model, val_loader, device)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

    save_path = "/content/drive/MyDrive/ColabNotebooks/lab4/results/model/compact_seg_model.pth"
    torch.save(student_model.state_dict(), save_path)
    print(f"Student model saved to: {save_path}")

    plot_path = "/content/drive/MyDrive/ColabNotebooks/lab4/results/plots/training_validation_loss.png"
    plot_loss(train_losses, val_losses, plot_path)


if __name__ == "__main__":
    alpha = 0.7
    beta = 0.3
    tau = 2.0
    num_epochs = 20
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_model = CompactSegmentationModel(num_classes=21)
    teacher_model = fcn_resnet50(weights="DEFAULT")

    data_path = '/content/drive/MyDrive/ColabNotebooks/lab4/data/VOCdevkit'
    train_loader = get_dataloader(data_path, 'train', batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(data_path, 'val', batch_size=batch_size, shuffle=False)

    train_with_distillation(student_model, teacher_model, train_loader, val_loader, device, num_epochs, alpha, beta, tau)