import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from pretrained_fcn_resnet import pretrained_fcn_resnet  # Import the model-loading function
from seg_model import CompactUNet  # Ensure your student model is imported
from dataloader import get_dataloader
from visualizations import plot_training_validation_loss

# Constants
alpha = 0.5
beta = 0.5
temperature = 3.0

def train_distillation(data_path, batch_size, num_epochs, learning_rate, num_classes=21, device="cpu"):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Models
    student_model = CompactUNet(num_classes).to(device)
    teacher_model = pretrained_fcn_resnet().to(device)
    teacher_model.eval()  # Set teacher model to evaluation mode

    # Optimizer and loss functions
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    criterion_ce = nn.CrossEntropyLoss()  # Define cross-entropy loss here

    def distillation_loss(output, target, teacher_output, T=temperature, alpha=alpha, beta=beta):
        """Calculate the distillation loss."""
        soft_targets = nn.functional.softmax(teacher_output / T, dim=1)
        soft_output = nn.functional.log_softmax(output / T, dim=1)
        distillation = nn.functional.kl_div(soft_output, soft_targets, reduction='batchmean') * (T * T)
        return alpha * distillation + beta * criterion_ce(output, target)

    # Load data
    train_loader = get_dataloader(data_path, 'train', batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(data_path, 'val', batch_size=batch_size, shuffle=False)

    training_losses = []
    validation_losses = []

    # Training loop
    for epoch in range(num_epochs):
        student_model.train()
        total_epoch_loss = 0
        for images, targets in tqdm(train_loader, desc="Training", unit="batch"):
            images, targets = images.to(device), targets.to(device)

            # Student forward pass
            student_output = student_model(images)

            # Teacher forward pass
            with torch.no_grad():
                teacher_output = teacher_model(images)['out']

            # Loss computation
            loss = distillation_loss(student_output, targets, teacher_output)
            total_epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log training progress
        average_epoch_loss = total_epoch_loss / len(train_loader)
        training_losses.append(average_epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_epoch_loss:.4f}')

        # Validation loop
        student_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating", unit="batch"):
                images, targets = images.to(device), targets.to(device)
                outputs = student_model(images)
                loss = criterion_ce(outputs, targets)
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(average_val_loss)
        print(f'Validation Loss: {average_val_loss:.4f}')

    # Plot training and validation losses
    plot_path = 'results/plots/distillation_loss.html'
    plot_training_validation_loss(training_losses, validation_losses, save_path=plot_path)

    # Save the trained student model
    torch.save(student_model.state_dict(), 'results/model/student_model.pth')
    print("Training completed and model saved.")



if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train_distillation(
        data_path='data/VOC/VOCdevkit',
        batch_size=32,
        num_epochs=20,
        learning_rate=0.001,
        num_classes=21,
        device=device
    )
