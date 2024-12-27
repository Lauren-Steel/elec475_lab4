import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactSemanticSegmentationModel(nn.Module):
    def __init__(self, num_classes=21):
        super(CompactSemanticSegmentationModel, self).__init__()

        # Initial convolution block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Encoder blocks with increasing receptive fields
        self.encoder1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Context aggregation layer
        self.context_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder blocks with skip connections
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final classification layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        # Initial convolution
        initial = self.initial_conv(x)

        # Encoder path
        enc1 = self.encoder1(initial)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        # Context aggregation
        context_feat = self.context_conv(enc3)

        # Decoder path with upsampling
        dec1 = F.interpolate(context_feat, scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = self.decoder1(dec1)

        dec2 = F.interpolate(dec1, scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = self.decoder2(dec2)

        dec3 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = self.decoder3(dec3)

        # Final upsampling and classification
        logits = F.interpolate(dec3, size=x.size()[2:], mode='bilinear', align_corners=True)
        logits = self.final_conv(self.dropout(logits))

        return {'out': logits}