"""
Residual Attention U-Net with ASPP bottleneck.

Upgrades over the baseline U-Net:
  1. Residual conv blocks   -> deeper effective network, better gradient
                                flow, closer to what production backbones
                                (ResNet/EfficientNet via segmentation-models-
                                pytorch) actually use internally.
  2. Attention gates on skip connections (Oktay et al., 2018) -> lets the
                                decoder suppress irrelevant background
                                activations (roads, vegetation) and focus
                                on building-like regions. This directly
                                targets the false-positive-on-roads problem
                                seen in the baseline model.
  3. ASPP bottleneck (atrous spatial pyramid pooling, from DeepLabv3+)
                             -> captures multi-scale context so small
                                buildings and large industrial rooftops are
                                both segmented well from the same model.

Still trains from scratch on CPU (no external pretrained weights needed -
this sandbox has no route to download ImageNet checkpoints), but the
architecture itself is now aligned with what the tech stack names as the
production choice (DeepLabv3+ / attention-augmented encoder-decoder).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class AttentionGate(nn.Module):
    """Oktay et al. attention gate: uses the coarser decoder signal (gating)
    to reweight the finer encoder skip-connection features before
    concatenation, so irrelevant spatial regions get suppressed."""
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode='bilinear', align_corners=False)
        attn = self.psi(self.relu(g + s))
        return skip * attn


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling - parallel dilated convolutions at
    multiple rates to capture multi-scale context at the bottleneck,
    plus global average pooling branch for whole-tile context."""
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            for r in rates
        ])
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        g = self.global_branch(x)
        g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        feats.append(g)
        return self.project(torch.cat(feats, dim=1))


class AttentionResUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=16):
        super().__init__()
        self.enc1 = ResidualConvBlock(in_ch, base)
        self.enc2 = ResidualConvBlock(base, base * 2)
        self.enc3 = ResidualConvBlock(base * 2, base * 4)
        self.enc4 = ResidualConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.aspp = ASPP(base * 8, base * 16)

        self.att4 = AttentionGate(base * 16, base * 8, base * 8)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ResidualConvBlock(base * 16, base * 8)

        self.att3 = AttentionGate(base * 8, base * 4, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ResidualConvBlock(base * 8, base * 4)

        self.att2 = AttentionGate(base * 4, base * 2, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ResidualConvBlock(base * 4, base * 2)

        self.att1 = AttentionGate(base * 2, base, base)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ResidualConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.aspp(self.pool(e4))

        s4 = self.att4(b, e4)
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        s3 = self.att3(d4, e3)
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        s2 = self.att2(d3, e2)
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        s1 = self.att1(d2, e1)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.out(d1)


# Backward-compat alias so existing scripts that import UNet still work
# during the transition; new code should use AttentionResUNet directly.
UNet = AttentionResUNet


if __name__ == '__main__':
    m = AttentionResUNet(base=16)
    x = torch.randn(2, 3, 128, 128)
    y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f'Output shape: {y.shape}')
    print(f'Total parameters: {n_params:,}')
