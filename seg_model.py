import torch
import torch.nn as nn
import torch.nn.functional as F



class CompactSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(CompactSegmentationModel, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output 128x128 for 256x256 input

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output 64x64 for 128x128 input

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # Upsamples to 128x128
        self.bn3 = nn.BatchNorm2d(16)
        self.upconv2 = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2)  # Upsamples to 256x256


    def forward(self, x):
        # Encoder
        e1 = F.relu(self.bn1(self.conv1(x)))  # 128x128
        x = self.pool1(e1)
        x = F.relu(self.bn2(self.conv2(x)))  # 64x64
        x = self.pool2(x)

        # Decoder
        x = F.relu(self.bn3(self.upconv1(x)))  # 128x128
        e1_resized = F.interpolate(e1, size=x.size()[2:], mode='bilinear', align_corners=False)  # Resize e1 to 128x128
        x = e1_resized + x  # Skip connection
        x = self.upconv2(x)  # 256x256

        return x


class CompactUNet(nn.Module):
    def __init__(self, num_classes):
        super(CompactUNet, self).__init__()
        # Encoder (Downsampling Path)
        self.enc_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Decoder (Upsampling Path)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        
        # Final Convolution
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = F.relu(self.enc_conv1(x))
        e1_pool = self.pool1(e1)
        e2 = F.relu(self.enc_conv2(e1_pool))
        e2_pool = self.pool2(e2)

        # Bottleneck
        b = F.relu(self.bottleneck_conv(e2_pool))

        # Decoder
        d1 = self.upconv1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = F.relu(self.dec_conv1(d1))
        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = F.relu(self.dec_conv2(d2))

        # Output
        out = self.final_conv(d2)
        return out
