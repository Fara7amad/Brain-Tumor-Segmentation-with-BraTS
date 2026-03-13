"""
model.py — 3D U-Net for BraTS2020 Brain Tumor Segmentation
============================================================
Architecture: Encoder → Bottleneck → Decoder with skip connections.
Each level doubles/halves the feature maps and halves/doubles spatial dims.

Input:  (B, 4,  128, 128, 128)  — batch of 4-modality MRI volumes
Output: (B, 4,  128, 128, 128)  — per-voxel class logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Residual Block ───────────────────────────────────────────────────────────
# Two conv3d layers with a skip connection.
# InstanceNorm3d is used instead of BatchNorm because BraTS batch size is 1
# (one 128³ volume barely fits in VRAM), and BatchNorm is unstable at batch=1.
# LeakyReLU(0.01) avoids dead neurons better than standard ReLU.

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act   = nn.LeakyReLU(0.01, inplace=True)

        # 1×1×1 projection so skip connection can match channel count
        # If in_ch == out_ch this is just an identity (no parameters added)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False) \
                    if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)   # add skip then activate


# ─── Encoder Block ────────────────────────────────────────────────────────────
# Each encoder level:
#   1. ResidualBlock to extract features at current resolution
#   2. Strided Conv3d (stride=2) to halve spatial dimensions
# Returns both the downsampled output AND the pre-downsample features (skip).
# The skip connection is later concatenated in the corresponding decoder level.

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res  = ResidualBlock(in_ch, out_ch)
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=3,
                              stride=2, padding=1, bias=False)

    def forward(self, x):
        skip = self.res(x)      # full-resolution features → stored for skip
        out  = self.down(skip)  # halved spatial dims → passed to next level
        return out, skip


# ─── Decoder Block ────────────────────────────────────────────────────────────
# Each decoder level:
#   1. Trilinear upsample to double spatial dimensions
#   2. Concatenate with the skip connection from the matching encoder level
#   3. ResidualBlock to fuse upsampled + skip features
# The concat doubles the channel count, so ResidualBlock takes in_ch + skip_ch.

class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up  = nn.Upsample(scale_factor=2, mode="trilinear",
                               align_corners=True)
        self.res = ResidualBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Pad if spatial dims don't match exactly (can happen with odd input sizes)
        if x.shape != skip.shape:
            x = F.pad(x, _pad_to_match(x, skip))

        return self.res(torch.cat([x, skip], dim=1))


# ─── Full 3D U-Net ────────────────────────────────────────────────────────────
# depth=4 means 4 encoder levels, 1 bottleneck, 4 decoder levels.
# base_filters=32 means the first level has 32 feature maps.
# Each subsequent level doubles: [32, 64, 128, 256] with bottleneck at 512.
# Total parameters with default settings: ~19M

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4,
                 base_filters=32, depth=4):
        super().__init__()
        self.depth = depth

        # Build filter counts per level: [32, 64, 128, 256, 512]
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]

        # Encoder: depth DownBlocks
        self.encoders = nn.ModuleList()
        self.encoders.append(DownBlock(in_channels, filters[0]))
        for i in range(1, depth):
            self.encoders.append(DownBlock(filters[i - 1], filters[i]))

        # Bottleneck: single ResidualBlock at lowest resolution
        self.bottleneck = ResidualBlock(filters[depth - 1], filters[depth])

        # Decoder: depth UpBlocks (mirror of encoder)
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.decoders.append(UpBlock(filters[i + 1], filters[i], filters[i]))

        # Final 1×1×1 conv: map feature maps → class logits
        self.head = nn.Conv3d(filters[0], out_channels, kernel_size=1)

        self._init_weights()

    def forward(self, x):
        # Encode — collect skip connections
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode — consume skip connections in reverse order
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])

        return self.head(x)   # (B, 4, 128, 128, 128) logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self):
        # Kaiming init for conv layers — designed for LeakyReLU
        # Ones/zeros for InstanceNorm affine parameters (standard)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ─── Utility ──────────────────────────────────────────────────────────────────
# Computes the padding needed to make tensor x match the spatial dims of target.
# Only needed when input dimensions are odd, causing off-by-one after downsample.

def _pad_to_match(x: torch.Tensor, target: torch.Tensor):
    diffs = [t - s for s, t in zip(x.shape[2:], target.shape[2:])]
    pad = []
    for d in reversed(diffs):
        pad += [d // 2, d - d // 2]
    return pad