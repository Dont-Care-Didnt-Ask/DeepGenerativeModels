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
            ConvBlock(128, 128, 3)
        )
        self.fc1 = nn.Linear(8 * 8 * 128, 2048)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2048, n_classes)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 8 * 8 * 128)
        x = self.act(self.fc1(x))
        x = self.fc2(self.drop(x))
        return x
    
    def get_activations(self, x):
        x = self.feature_extractor(x).view(-1, 8 * 8 * 128)
        inner_activations = self.act(self.fc1(x))
        return inner_activations