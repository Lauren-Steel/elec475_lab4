import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

# Personal Code
from css_model import CompactSemanticSegmentationModel
from visualizations import plot_training_validation_loss

def convert_to_tensor(x):
    return torch.as_tensor(np.array(x), dtype=torch.int64)

def train_compact(num_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, save_file=None, plot_file=None):

    training_losses = []
    total_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        print('Training ...')
        model.train()
        print('Epoch', epoch)
        epoch_loss = 0.0

        for images, targets in tqdm(train_loader):
            images, targets = images.to(device), targets.to(device)
                        
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs['out'], targets)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step(epoch_loss)

        avg_training_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_training_loss)
        print(f"Training Loss: {avg_training_loss:.4f}")

        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch Time: {epoch_duration:.2f} seconds")

    total_training_time = time.time() - total_start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

    print(f"Saving model weights to {save_file}")
    torch.save(model.state_dict(), save_file)

    plot_training_validation_loss(training_losses, save_path=plot_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model on PASCAL VOC 2012")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()

    # Device configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using {device}")

    # Transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])

    # Transformation for the segmentation targets (no normalization, only resize)
    target_transform = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(convert_to_tensor)
    ])


    # Load the PASCAL VOC 2012 dataset (training set)
    train_dataset = datasets.VOCSegmentation(
        root="data", 
        year="2012", 
        image_set="train", 
        download=False, 
        transform=transform, 
        target_transform=target_transform
    )


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize the model
    model = CompactSemanticSegmentationModel(num_classes=21).to(device)

    # Optimizer, scheduler, and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    # Define paths for saving plots and models
    save_file = os.path.join("results/models/custom_model.pth")
    plot_file = os.path.join("results/plots/custom_semantic_segmentation_loss.html")

    # epochs = 10
    # Train the model
    train_compact(
        num_epochs=args.epochs,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        scheduler=scheduler,
        device=device,
        save_file=save_file,
        plot_file=plot_file
    )