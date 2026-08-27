"""
models/siamese_unet.py

Siamese U-Net for bi-temporal change detection.

- A single ResNet (weights shared, i.e. "Siamese") encodes both the "before"
  and "after" image at 4 spatial scales.
- At every scale we take the absolute difference of the two feature maps.
  This is the standard, well-validated "feature differencing" fusion strategy
  used across the change-detection literature (FC-Siam-diff, STANet, etc.)
- A U-Net-style decoder (transpose-conv upsample + concat with the
  corresponding-scale difference features) reconstructs a full-resolution
  1-channel change map.

Backbone options: resnet18 (fast, default) or resnet34 (slightly more
capacity, still trains quickly).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNetEncoder(nn.Module):
    """Wraps a torchvision ResNet and exposes intermediate feature maps."""

    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            net = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.channels = [64, 64, 128, 256, 512]  # stem, layer1..4
        elif backbone == "resnet34":
            net = torchvision.models.resnet34(
                weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)  # /2
        self.maxpool = net.maxpool                                # /4
        self.layer1 = net.layer1                                  # /4
        self.layer2 = net.layer2                                  # /8
        self.layer3 = net.layer3                                  # /16
        self.layer4 = net.layer4                                  # /32

    def forward(self, x):
        f0 = self.stem(x)          # /2   ch[0]
        p0 = self.maxpool(f0)
        f1 = self.layer1(p0)       # /4   ch[1]
        f2 = self.layer2(f1)       # /8   ch[2]
        f3 = self.layer3(f2)       # /16  ch[3]
        f4 = self.layer4(f3)       # /32  ch[4]
        return [f0, f1, f2, f3, f4]


class DecoderBlock(nn.Module):
    """Upsample, concat with skip (difference) features, conv-bn-relu x2."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, out_channels=1):
        super().__init__()
        self.encoder = ResNetEncoder(backbone, pretrained)
        c = self.encoder.channels  # [64, 64, 128, 256, 512] at strides [2,4,8,16,32]

        self.center = nn.Sequential(
            nn.Conv2d(c[4], c[4], 3, padding=1, bias=False),
            nn.BatchNorm2d(c[4]),
            nn.ReLU(inplace=True),
        )

        self.dec4 = DecoderBlock(c[4], c[3], 256)   # /32 -> /16
        self.dec3 = DecoderBlock(256, c[2], 128)    # /16 -> /8
        self.dec2 = DecoderBlock(128, c[1], 64)     # /8  -> /4
        self.dec1 = DecoderBlock(64, c[0], 32)      # /4  -> /2

        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  # /2 -> /1
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, img_a, img_b):
        feats_a = self.encoder(img_a)
        feats_b = self.encoder(img_b)  # shared weights -> same self.encoder call

        # multi-scale absolute feature difference
        diffs = [torch.abs(fa - fb) for fa, fb in zip(feats_a, feats_b)]
        d0, d1, d2, d3, d4 = diffs  # strides 2,4,8,16,32

        x = self.center(d4)
        x = self.dec4(x, d3)
        x = self.dec3(x, d2)
        x = self.dec2(x, d1)
        x = self.dec1(x, d0)
        x = self.final_up(x)
        logits = self.final_conv(x)

        if logits.shape[-2:] != img_a.shape[-2:]:
            logits = F.interpolate(logits, size=img_a.shape[-2:], mode="bilinear", align_corners=False)
        return logits  # raw logits, shape (B, 1, H, W) -> apply sigmoid outside


def build_model(backbone="resnet18", pretrained=True):
    return SiameseUNet(backbone=backbone, pretrained=pretrained)


if __name__ == "__main__":
    model = build_model()
    a = torch.randn(2, 3, 256, 256)
    b = torch.randn(2, 3, 256, 256)
    out = model(a, b)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("Output shape:", out.shape)
    print(f"Trainable params: {n_params:.2f}M")
