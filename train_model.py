import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from seg_model import CompactSegmentationModel  # Import your model

from dataloader import get_dataloader  # Import dataloader
from visualizations import plot_training_validation_loss




def train_model(data_path, batch_size, num_epochs, learning_rate, num_classes=21, device='cpu'):
    # Initialize model, loss, and optimizer
    model = CompactSegmentationModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Load data
    train_loader = get_dataloader(data_path, 'train', batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(data_path, 'val', batch_size=batch_size, shuffle=False)

    training_losses = []
    validation_losses = []

    total_start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Track epoch start time
        model.train()
        epoch_loss = 0

        # Training with tqdm
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for images, targets in tqdm(train_loader, desc="Training", unit="batch"):
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_training_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_training_loss)
        print(f"Training Loss: {avg_training_loss:.4f}")

        # Validation loop
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for images, targets in tqdm(val_loader, desc="Validating", unit="batch"):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

            avg_validation_loss = total_val_loss / len(val_loader)
            validation_losses.append(avg_validation_loss)
            print(f"Validation Loss: {avg_validation_loss:.4f}")

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch Time: {epoch_duration:.2f} seconds")
    
    total_training_time = time.time() - total_start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

    # Save model
    save_path = 'results/model/seg_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # plot training and val loss
    plot_path = 'results/plots/training_validation_loss.html'
    plot_training_validation_loss(training_losses, validation_losses, save_path=plot_path)


def main():
    # Command-line arguments for training parameters
    parser = argparse.ArgumentParser(description="Train a compact segmentation model.")
    parser.add_argument('--data_path', type=str, default='data/VOC/VOCdevkit', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_classes', type=int, default=21, help='Number of classes in dataset')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (e.g., "cpu", "cuda", "mps")')

    args = parser.parse_args()

    # Train the model
    train_model(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_classes=args.num_classes,
        device=torch.device(args.device)
    )


if __name__ == '__main__':
    main()