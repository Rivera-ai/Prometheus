import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x / torch.norm(x, dim=-1, keepdim=True)

class EncoderV1(nn.Module):
    def __init__(self, dim_latent, device='cuda'):
        super(EncoderV1, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),  # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),      # 64x64
            ResidualBlock(64, 128, stride=2),          # 32x32
            ResidualBlock(128, 256, stride=2),         # 16x16
            nn.Conv2d(256, dim_latent, 1),            # 16x16xdim_latent
        )
        
        self.rearrange = Rearrange('b d h w -> b h w d')
        self.normalize = Normalize()
        self.to(device)
        
    def forward(self, x):
        x = self.features(x)
        x = self.rearrange(x)
        x = self.normalize(x)
        return x

class DecoderV1(nn.Module):
    def __init__(self, dim_latent, device='cuda'):
        super(DecoderV1, self).__init__()
        
        self.rearrange = Rearrange('b h w d -> b d h w')
        
        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(dim_latent, 256, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),    # 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),     # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),       # 256x256
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        x = self.rearrange(x)
        x = self.decoder_net(x)
        return x