"""
model.py — U-Net ResNet-18, 5 classes ZeroWaste
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

import config


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBnRelu(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet18(nn.Module):
    """U-Net ResNet-18 pré-entraîné ImageNet, sortie NUM_CLASSES canaux."""

    def __init__(self, num_classes=config.NUM_CLASSES,
                 pretrained=True, freeze_encoder=False):
        super().__init__()
        bb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        self.enc1 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)  # /2   64ch
        self.pool = bb.maxpool
        self.enc2 = bb.layer1   # /4   64ch
        self.enc3 = bb.layer2   # /8  128ch
        self.enc4 = bb.layer3   # /16 256ch
        self.enc5 = bb.layer4   # /32 512ch

        if freeze_encoder:
            for m in [self.enc1,self.enc2,self.enc3,self.enc4,self.enc5]:
                for p in m.parameters():
                    p.requires_grad = False

        self.bot  = ConvBnRelu(512, 256)
        self.dec5 = DecoderBlock(256, 256, 128)
        self.dec4 = DecoderBlock(128, 128,  64)
        self.dec3 = DecoderBlock( 64,  64,  64)
        self.dec2 = DecoderBlock( 64,  64,  32)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        s5 = self.enc5(s4)
        x  = self.bot(s5)
        x  = self.dec5(x, s4)
        x  = self.dec4(x, s3)
        x  = self.dec3(x, s2)
        x  = self.dec2(x, s1)
        x  = self.dec1(x)
        return self.head(x)

    def get_param_groups(self, lr):
        """Differential LR : encodeur à lr/10, décodeur à lr plein."""
        enc = (list(self.enc1.parameters()) + list(self.enc2.parameters()) +
               list(self.enc3.parameters()) + list(self.enc4.parameters()) +
               list(self.enc5.parameters()))
        dec = (list(self.bot.parameters())  + list(self.dec5.parameters()) +
               list(self.dec4.parameters()) + list(self.dec3.parameters()) +
               list(self.dec2.parameters()) + list(self.dec1.parameters()) +
               list(self.head.parameters()))
        return [
            {"params": enc, "lr": lr / 10, "initial_lr": lr / 10},
            {"params": dec, "lr": lr,      "initial_lr": lr},
        ]


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """FL(p) = -α(1-p)^γ log(p)  — force le modèle sur les classes difficiles."""
    def __init__(self, gamma=config.FOCAL_GAMMA, alpha=None, ignore_index=255):
        super().__init__()
        self.gamma  = gamma
        self.alpha  = alpha      # tensor (C,) ou None
        self.ignore = ignore_index

    def forward(self, logits, targets):
        valid    = targets != self.ignore
        logits_v = logits.permute(0, 2, 3, 1)[valid]   # (N, C)
        targets_v = targets[valid]                       # (N,)

        log_p  = F.log_softmax(logits_v, dim=1)
        pt     = log_p.exp().gather(1, targets_v.unsqueeze(1)).squeeze(1)
        log_pt = log_p.gather(1, targets_v.unsqueeze(1)).squeeze(1)
        fw     = (1 - pt) ** self.gamma

        if self.alpha is not None:
            fw = fw * self.alpha.to(logits.device)[targets_v]

        return -(fw * log_pt).mean()


class CombinedLoss(nn.Module):
    """Focal Loss + Dice Loss (50/50)."""
    def __init__(self, num_classes=config.NUM_CLASSES, class_weights=None):
        super().__init__()
        self.num_cls = num_classes
        self.focal   = FocalLoss(gamma=config.FOCAL_GAMMA, alpha=class_weights)
        print(f"  Loss : Focal(γ={config.FOCAL_GAMMA}) + Dice")

    def forward(self, logits, targets):
        return 0.5 * self.focal(logits, targets) + 0.5 * self._dice(logits, targets)

    def _dice(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        oh    = F.one_hot(targets.clamp(0, self.num_cls - 1),
                           self.num_cls).permute(0, 3, 1, 2).float()
        inter = (probs * oh).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + oh.sum(dim=(0, 2, 3))
        return (1 - (2 * inter + 1e-6) / (union + 1e-6))[1:].mean()


# Alias rétrocompatibilité
UNetResNet50 = UNetResNet18
compute_class_weights = None   # remplacé par compute_class_weights_from_loader dans train.py


if __name__ == "__main__":
    m = UNetResNet18()
    x = torch.randn(2, 3, 256, 256)
    print(f"Output : {m(x).shape}")
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total : {total/1e6:.2f}M | Entraînables : {trainable/1e6:.2f}M")
