import torch
import torch.nn as nn
import torch.nn.functional as F
        
class SparseBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, bias=False, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_features, out_features, kernel, stride=stride, padding=(kernel-1)//2, bias=bias)
        self.norm = nn.BatchNorm2d(out_features)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return self.act(self.norm(self.conv(x)))

class SparseAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            SparseBlock(1, 16, 3),            # 64x64x1 -> 64x64x16  
            SparseBlock(16, 16, 3, stride=2), # 64x64x16 -> 32x32x16
            SparseBlock(16, 16, 3),
            SparseBlock(16, 32, 3, stride=2), # 32x32x16 -> 16x16x32
            SparseBlock(32, 32, 3),
            SparseBlock(32, 32, 3, stride=2), # 16x16x32 -> 8x8x32
            SparseBlock(32, 32, 3),
            SparseBlock(32, 64, 3, stride=2), # 8x8x32 -> 4x4x64
            SparseBlock(64, 64, 3),
            SparseBlock(64, 64, 3).conv,
            SparseBlock(64, 64, 3).norm
        )
        
        self.decoder = nn.Sequential(
            SparseBlock(64, 64, 3),
            SparseBlock(64, 32, 3, upsample=True), # 4x4x64 -> 8x8x32
            SparseBlock(32, 32, 3),
            SparseBlock(32, 32, 3, upsample=True), # 8x8x32 -> 16x16x32
            SparseBlock(32, 32, 3),
            SparseBlock(32, 32, 3, upsample=True), # 16x16x32 -> 32x32x32
            SparseBlock(32, 32, 3),
            SparseBlock(32, 16, 3, upsample=True), # 32x32x32 -> 64x64x16
            SparseBlock(16, 1, 3).conv,            # 64x64x16 -> 64x64x1
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x
    
    def get_latent_features(self, x):
        return self.encoder(x)
