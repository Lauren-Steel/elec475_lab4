import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(CompactSegmentationModel, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.upconv3 = nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.pool1(x1)

        x3 = F.relu(self.bn2(self.conv2(x2)))
        x4 = self.pool2(x3)

        x5 = F.relu(self.bn3(self.conv3(x4)))

        # Decoder with skip connections
        x6 = F.relu(self.bn4(self.upconv1(x5)) + x3)  # Skip connection
        x7 = F.relu(self.bn5(self.upconv2(x6)) + x1)  # Skip connection
        x8 = self.upconv3(x7)

        return x8