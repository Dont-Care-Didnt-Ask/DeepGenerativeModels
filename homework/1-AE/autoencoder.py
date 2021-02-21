import torch
import torch.nn as nn
import torch.nn.functional as F
        
class DenoisingBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, bias=False, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_features, out_features, kernel, stride=stride, padding=(kernel-1)//2, bias=bias)
        self.norm = nn.BatchNorm2d(out_features)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        x = x + torch.randn_like(x) * 0.05
        x = self.dropout(x)
        return self.act(self.norm(self.conv(x)))

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            DenoisingBlock(1, 16, 3, stride=2),  # 64 -> 32
            DenoisingBlock(16, 32, 3, stride=2), # 32 -> 16
            DenoisingBlock(32, 32, 3, stride=2), # 16 -> 8
            DenoisingBlock(32, 32, 3, stride=2), # 8 -> 4
            DenoisingBlock(32, 32, 3, stride=1).conv,
            DenoisingBlock(32, 32, 3, stride=1).norm
        )
        
        self.decoder = nn.Sequential(
            DenoisingBlock(32, 32, 3, upsample=True), # 4 -> 8
            DenoisingBlock(32, 32, 3, upsample=True), # 8 -> 16
            DenoisingBlock(32, 32, 3, upsample=True), # 16 -> 32
            DenoisingBlock(32, 16, 3, upsample=True), # 32 -> 64
            DenoisingBlock(16, 1, 3).conv,
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x
    
    def get_latent_features(self, x):
        return self.encoder(x)
