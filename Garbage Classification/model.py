"""
model.py — avec Focal Loss
==========================
Changements par rapport à la version précédente :
  • Focal Loss : pénalise les prédictions trop confiantes sur les classes faciles
    (ici cardboard qui domine). Formule : FL = -α(1-p)^γ * log(p)
  • Encodeur DÉGELÉ : nécessaire pour apprendre les features distinctives
    verre (transparent), métal (brillant), papier (blanc mat)
  • Differential LR : encodeur à LR/10, décodeur à LR plein
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

import config


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch // 2 + skip_ch, out_ch, dropout),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet18(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES,
                 pretrained=True, freeze_encoder=False):
        super().__init__()

        bb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.enc1  = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.pool  = bb.maxpool
        self.enc2  = bb.layer1
        self.enc3  = bb.layer2
        self.enc4  = bb.layer3
        self.enc5  = bb.layer4

        if freeze_encoder:
            for m in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
                for p in m.parameters():
                    p.requires_grad = False

        self.bottleneck = ConvBnRelu(512, 128)
        self.dec5 = DecoderBlock(128, 256, 96)
        self.dec4 = DecoderBlock(96,  128, 64)
        self.dec3 = DecoderBlock(64,   64, 48)
        self.dec2 = DecoderBlock(48,   64, 32)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            ConvBnRelu(16, 16),
        )
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        p  = self.pool(s1)
        s2 = self.enc2(p)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        s5 = self.enc5(s4)
        b  = self.bottleneck(s5)
        return self.head(self.dec1(self.dec2(self.dec3(self.dec4(self.dec5(b, s4), s3), s2), s1)))

    def get_param_groups(self, lr):
        """
        Differential LR : encodeur à lr/10, décodeur à lr plein.
        Évite de "désapprendre" les features ImageNet trop vite.
        """
        encoder_params = (list(self.enc1.parameters()) +
                          list(self.enc2.parameters()) +
                          list(self.enc3.parameters()) +
                          list(self.enc4.parameters()) +
                          list(self.enc5.parameters()))
        decoder_params = (list(self.bottleneck.parameters()) +
                          list(self.dec5.parameters()) +
                          list(self.dec4.parameters()) +
                          list(self.dec3.parameters()) +
                          list(self.dec2.parameters()) +
                          list(self.dec1.parameters()) +
                          list(self.head.parameters()))
        return [
            {"params": encoder_params, "lr": lr / 10},
            {"params": decoder_params, "lr": lr},
        ]


UNetResNet50 = UNetResNet18


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss pour segmentation sémantique multi-classe.
    FL(p) = -α * (1 - p)^γ * log(p)

    γ=2 : les exemples bien classifiés (p→1) contribuent peu à la loss.
          Le modèle se concentre sur les classes difficiles (verre, métal...).
    α   : poids inversement proportionnel à la fréquence de chaque classe.
    """
    def __init__(self, gamma=config.FOCAL_GAMMA, alpha=None,
                 num_classes=config.NUM_CLASSES, ignore_index=255):
        super().__init__()
        self.gamma  = gamma
        self.alpha  = alpha    # tensor (C,) ou None
        self.num_cls = num_classes
        self.ignore  = ignore_index

    def forward(self, logits, targets):
        # logits : (B, C, H, W)  targets : (B, H, W)
        B, C, H, W = logits.shape
        valid = targets != self.ignore
        logits_v  = logits.permute(0,2,3,1)[valid]       # (N, C)
        targets_v = targets[valid]                         # (N,)

        log_p  = F.log_softmax(logits_v, dim=1)           # (N, C)
        p      = log_p.exp()

        log_pt = log_p.gather(1, targets_v.unsqueeze(1)).squeeze(1)  # (N,)
        pt     = p.gather(1, targets_v.unsqueeze(1)).squeeze(1)

        focal_w = (1 - pt) ** self.gamma

        if self.alpha is not None:
            at = self.alpha.to(logits.device)[targets_v]
            focal_w = focal_w * at

        loss = -(focal_w * log_pt)
        return loss.mean()


class CombinedLoss(nn.Module):
    """Focal Loss + Dice Loss (50/50)."""
    def __init__(self, num_classes=config.NUM_CLASSES,
                 ce_weight=0.5, dice_weight=0.5,
                 class_weights=None):
        super().__init__()
        self.ce_w    = ce_weight
        self.dice_w  = dice_weight
        self.num_cls = num_classes

        if config.USE_FOCAL_LOSS:
            alpha = class_weights if config.FOCAL_ALPHA else None
            self.ce_loss = FocalLoss(gamma=config.FOCAL_GAMMA,
                                      alpha=alpha,
                                      num_classes=num_classes)
            print("  Loss : Focal Loss (γ={}) + Dice".format(config.FOCAL_GAMMA))
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights,
                                                ignore_index=255)
            print("  Loss : CrossEntropy + Dice")

    def forward(self, logits, targets):
        return (self.ce_w  * self.ce_loss(logits, targets) +
                self.dice_w * self._dice(logits, targets))

    def _dice(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        oh    = F.one_hot(targets.clamp(0, self.num_cls-1),
                           self.num_cls).permute(0,3,1,2).float()
        inter = (probs * oh).sum(dim=(0,2,3))
        union = probs.sum(dim=(0,2,3)) + oh.sum(dim=(0,2,3))
        return (1 - (2*inter + 1e-6)/(union + 1e-6))[1:].mean()


def compute_class_weights(scene_dir, device):
    import random, cv2, numpy as np
    from pathlib import Path
    counts = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    for level in config.DENSITY_LEVELS:
        msk_dir = Path(scene_dir) / level / "masks"
        files   = list(msk_dir.glob("*.png"))
        for f in random.sample(files, min(40, len(files))):
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            for c in range(config.NUM_CLASSES):
                counts[c] += (m == c).sum()
    freq = counts / (counts.sum() + 1e-9)
    w    = 1.0 / (freq + 1e-6)
    w    = w / w.sum() * config.NUM_CLASSES
    print(f"  Poids classes : {np.round(w, 2)}")
    return torch.tensor(w, dtype=torch.float32).to(device)


if __name__ == "__main__":
    m = UNetResNet18(freeze_encoder=False)
    x = torch.randn(2, 3, 128, 128)
    print(f"Output : {m(x).shape}")
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total : {total/1e6:.2f}M | Entraînables : {trainable/1e6:.2f}M")
