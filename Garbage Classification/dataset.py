"""
dataset.py
==========
PyTorch Dataset et DataLoaders pour les scènes composites.
Inclut les augmentations (albumentations) et la division train/val/test.
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentations
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5,
                           border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WasteSceneDataset(Dataset):
    """
    Dataset d'une scène (sparse / medium / dense).

    Attend la structure :
      root/
        images/  *.png
        masks/   *.png  (même stem, uint8, valeur = class_idx)
    """

    def __init__(self,
                 root: str,
                 density_level: str,
                 transform: Optional[A.Compose] = None):
        self.root     = Path(root) / density_level
        self.level    = density_level
        self.transform = transform

        img_dir = self.root / "images"
        msk_dir = self.root / "masks"

        self.img_paths  = sorted(img_dir.glob("*.png"))
        self.mask_paths = [msk_dir / p.name for p in self.img_paths]

        missing = [p for p in self.mask_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} masques manquants dans {msk_dir}. "
                "Lancez scene_composer.py d'abord."
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]          # (C, H, W) float32 tensor
            mask  = augmented["mask"].long()    # (H, W) int64 tensor
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask  = torch.from_numpy(mask).long()

        return image, mask, self.level


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers de split et DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(scene_dir: str = config.SCENE_DIR,
                     batch_size: int = config.BATCH_SIZE,
                     num_workers: int = config.NUM_WORKERS,
                     seed: int = config.RANDOM_SEED
                     ) -> tuple[DataLoader, DataLoader, dict[str, DataLoader]]:
    """
    Construit les DataLoaders :
      • train_loader (toutes densités mélangées)
      • val_loader   (toutes densités mélangées)
      • test_loaders : {density_level: DataLoader}  (pour analyse par densité)

    Retourne (train_loader, val_loader, test_loaders)
    """
    train_sets, val_sets = [], []
    test_sets_by_level   = {}

    train_tf = get_train_transforms()
    val_tf   = get_val_transforms()

    g = torch.Generator().manual_seed(seed)

    for level in config.DENSITY_LEVELS:
        full = WasteSceneDataset(scene_dir, level, transform=None)
        n    = len(full)
        n_tr = int(n * config.TRAIN_RATIO)
        n_vl = int(n * config.VAL_RATIO)
        n_te = n - n_tr - n_vl

        tr, vl, te = random_split(full, [n_tr, n_vl, n_te], generator=g)

        # Réappliquer les transformations (dataset wrappé)
        train_sets.append(_TransformWrapper(tr, train_tf))
        val_sets.append(_TransformWrapper(vl, val_tf))
        test_sets_by_level[level] = DataLoader(
            _TransformWrapper(te, val_tf),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_sets_by_level


class _TransformWrapper(Dataset):
    """Applique une transformation albumentations à un Subset existant."""
    def __init__(self, subset, transform: A.Compose):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path  = self.subset.dataset.img_paths[self.subset.indices[idx]]
        mask_path = self.subset.dataset.mask_paths[self.subset.indices[idx]]
        level     = self.subset.dataset.level

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        aug   = self.transform(image=image, mask=mask)
        return aug["image"], aug["mask"].long(), level
