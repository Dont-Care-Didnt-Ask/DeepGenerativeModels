import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, bias = False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride = stride,
            padding = (kernel_size - 1) // 2,
            bias=bias
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.norm(self.conv(x + torch.randn_like(x) * 0.05)))

class Classifier(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(1, 16, 3), # 64x64x1 -> 64x64x16
            ConvBlock(16, 16, 3),
            ConvBlock(16, 32, 3, stride = 2), # 64x64 -> 32x32x32
            ConvBlock(32, 32, 3),
            ConvBlock(32, 64, 3, stride = 2), # 32x32x32 -> 16x16x64
            ConvBlock(64, 64, 3),
            ConvBlock(64, 128, 3, stride = 2), # 16x16x64 -> 8x8x128
            ConvBlock(128, 128, 3),
            ConvBlock(128, 128, 3, stride = 2), # 8x8x128 -> 4x4x128
        )
        self.fc1 = nn.Linear(4 * 4 * 128, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 4 * 4 * 128)
        x = self.fc1(x)
        x = nn.Dropout(0.2)(x)
        x = self.fc2(x)
        return x
    
    def get_activations(self, x):
        x = self.feature_extractor(x)
        return self.fc1(x), self.fc2(x)