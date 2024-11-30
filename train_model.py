import torch
import torch.nn as nn
from torch.optim import Adam
from seg_model import CompactSegmentationModel  # Import your model
from dataloader import get_dataloader  # Import dataloader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
data_path = '/content/drive/MyDrive/ColabNotebooks/lab4/data/VOCdevkit'

# Training parameters
batch_size = 8
num_epochs = 20
learning_rate = 0.001
num_classes = 21  # VOC2012 has 21 classes

# Initialize model, loss, and optimizer
model = CompactSegmentationModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Load data
train_loader = get_dataloader(data_path, 'train', batch_size=batch_size, shuffle=True)
val_loader = get_dataloader(data_path, 'val', batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    # Validate after each epoch (optional)
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()
        print(f"Validation Loss: {total_val_loss / len(val_loader):.4f}")

# Save model
torch.save(model.state_dict(), '/content/drive/MyDrive/ColabNotebooks/lab4/seg_model.pth')
print("Model saved to /content/drive/MyDrive/ColabNotebooks/lab4/seg_model.pth")
