"""
dataset.py — ZeroWaste complet + Garbage Classification, 5 classes
"""

import os, random, cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import config

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)


# ─── Augmentations ────────────────────────────────────────────────────────────

def _augment(img_rgb, mask):
    if random.random() < 0.5:
        img_rgb = img_rgb[:, ::-1].copy(); mask = mask[:, ::-1].copy()
    if random.random() < 0.3:
        img_rgb = img_rgb[::-1].copy();    mask = mask[::-1].copy()
    if random.random() < 0.4:
        k = random.randint(1, 3)
        img_rgb = np.rot90(img_rgb, k).copy(); mask = np.rot90(mask, k).copy()
    if random.random() < 0.6:
        img_rgb = np.clip(img_rgb * random.uniform(0.75, 1.25), 0, 255).astype(np.uint8)
    if random.random() < 0.2:
        img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), 0)
    return img_rgb, mask


def _to_tensor(img_rgb, mask):
    img_f = img_rgb.astype(np.float32) / 255.0
    img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
    return (torch.from_numpy(img_f.transpose(2, 0, 1)),
            torch.from_numpy(mask.copy()).long())


# ─── Dataset 1 : ZeroWaste complet ────────────────────────────────────────────

class ZeroWasteDataset(Dataset):
    """
    Utilise le dataset ZeroWaste ORIGINAL (4503 images).
    Sur train : 30 images par classe dominante pour éviter les frames
    consécutifs tout en gardant de la diversité inter-classe.
    Sur val/test : tout le split (évaluation complète).
    """
    def __init__(self, zerowaste_dir, split="train",
                 img_size=config.IMG_SIZE, n_per_class=config.N_PER_CLASS):
        self.img_dir  = Path(zerowaste_dir) / split / "data"
        self.mask_dir = Path(zerowaste_dir) / split / "sem_seg"
        self.size     = img_size
        self.train    = (split == "train")

        if not self.img_dir.exists():
            raise FileNotFoundError(f"ZeroWaste introuvable : {self.img_dir}")

        all_files = sorted([f for f in self.img_dir.iterdir()
                             if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

        if n_per_class and split == "train":
            self.files = self._sample_per_class(all_files, n_per_class)
        else:
            self.files = all_files

        print(f"  ZeroWaste {split:<6}: {len(self.files)} images")

    def _classes_present(self, mask_path):
        """Retourne l'ensemble des classes présentes dans le masque."""
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None: return {0}
        return set(np.unique(m).tolist())

    def _sample_per_class(self, all_files, n_per_class):
        """
        Groupe les images par classes PRÉSENTES (pas dominantes).
        Une image peut apparaître dans plusieurs groupes.
        Puis on tire n_per_class images par classe objet (1-4),
        et n_per_class images background parmi celles qui en contiennent.
        """
        groups = defaultdict(list)
        print(f"    Analyse des masques ({len(all_files)} images) …")
        for f in all_files:
            mp      = self.mask_dir / (f.stem + ".png")
            present = self._classes_present(mp)
            for cls in present:
                groups[cls].append(f)

        selected_set = set()
        selected     = []

        for cls in range(config.NUM_CLASSES):
            g      = groups[cls]
            limit  = n_per_class   # même limite pour toutes les classes
            chosen = random.sample(g, min(limit, len(g)))
            added  = 0
            for f in chosen:
                if f not in selected_set:
                    selected_set.add(f)
                    selected.append(f)
                    added += 1
            print(f"    ZW cls {cls} ({config.CLASS_NAMES[cls]:<16}): "
                  f"{len(g):>4} imgs contiennent cette classe → {added} ajoutées")

        random.shuffle(selected)
        return selected

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img  = cv2.cvtColor(cv2.imread(str(name)), cv2.COLOR_BGR2RGB)
        mp   = self.mask_dir / (name.stem + ".png")
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)

        img  = cv2.resize(img,  (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size),
                          interpolation=cv2.INTER_NEAREST)
        mask = np.clip(mask, 0, config.NUM_CLASSES - 1)

        if self.train:
            img, mask = _augment(img, mask)
        return _to_tensor(img, mask)


# ─── Dataset 2 : Garbage Classification ──────────────────────────────────────

class GarbageDataset(Dataset):
    """
    Images Garbage Classification (fond blanc) + masques auto générés.
    30 images par classe CIBLE (pas par dossier source) pour éviter que
    rigid_plastic soit surreprésenté (glass + plastic → même classe).

    Bordure fond blanc 30% ajoutée autour de chaque image pour que le
    modèle voie du vrai fond même depuis Garbage Classification.
    """
    def __init__(self, raw_dir=config.RAW_DATA_DIR,
                       mask_dir=config.MASK_DIR,
                       img_size=config.IMG_SIZE,
                       n_per_class=config.N_PER_CLASS,
                       train=True):
        self.size    = img_size
        self.train   = train
        self.samples = []
        self._load(raw_dir, mask_dir, n_per_class)
        print(f"  GarbageClassif   : {len(self.samples)} images "
              f"(≤{n_per_class}/classe cible)")

    def _load(self, raw_dir, mask_dir, n_per_class):
        by_class = defaultdict(list)
        for folder in sorted(Path(raw_dir).iterdir()):
            if not folder.is_dir(): continue
            cls_idx = config.FOLDER_TO_CLASS.get(folder.name.lower())
            if cls_idx is None: continue
            msk_folder = Path(mask_dir) / folder.name
            if not msk_folder.exists(): continue
            for img_f in sorted(folder.iterdir()):
                if img_f.suffix.lower() not in {".jpg",".jpeg",".png",".bmp",".webp"}:
                    continue
                mask_f = msk_folder / (img_f.stem + ".png")
                if mask_f.exists():
                    by_class[cls_idx].append((img_f, mask_f, cls_idx))

        for cls_idx, pairs in by_class.items():
            chosen = random.sample(pairs, min(n_per_class, len(pairs)))
            self.samples.extend(chosen)
            print(f"    GC cls {cls_idx} ({config.CLASS_NAMES[cls_idx]:<16}): "
                  f"{len(chosen):>3}/{len(pairs)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, _ = self.samples[idx]
        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)

        # Bordure fond blanc : force le modèle à voir du fond autour de l'objet
        img, mask = self._add_border(img, mask, ratio=0.25)

        img  = cv2.resize(img,  (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size),
                          interpolation=cv2.INTER_NEAREST)
        mask = np.clip(mask, 0, config.NUM_CLASSES - 1)

        if self.train:
            img, mask = _augment(img, mask)
        return _to_tensor(img, mask)

    @staticmethod
    def _add_border(img_rgb, mask, ratio=0.25):
        h, w   = img_rgb.shape[:2]
        bh, bw = int(h * ratio), int(w * ratio)
        canvas_img  = np.full((h+2*bh, w+2*bw, 3), 255, np.uint8)
        canvas_mask = np.zeros((h+2*bh, w+2*bw), np.uint8)
        canvas_img [bh:bh+h, bw:bw+w] = img_rgb
        canvas_mask[bh:bh+h, bw:bw+w] = mask
        return canvas_img, canvas_mask


# ─── Factory ──────────────────────────────────────────────────────────────────

def make_dataloaders(zerowaste_dir=config.ZEROWASTE_DIR,
                     raw_dir=config.RAW_DATA_DIR,
                     mask_dir=config.MASK_DIR,
                     batch_size=config.BATCH_SIZE,
                     num_workers=config.NUM_WORKERS,
                     n_per_class=config.N_PER_CLASS,
                     seed=config.RANDOM_SEED):
    random.seed(seed)

    zw_ok  = os.path.isdir(os.path.join(zerowaste_dir, "train", "data"))
    raw_ok = (config.USE_GARBAGE_DS and
              os.path.isdir(raw_dir) and os.path.isdir(mask_dir))

    print("\n═══ Chargement datasets ═══")

    # ── Train ─────────────────────────────────────────────────────────────────
    train_parts = []
    if zw_ok:
        print(f"  ZeroWaste depuis : {zerowaste_dir}")
        train_parts.append(
            ZeroWasteDataset(zerowaste_dir, "train", n_per_class=n_per_class)
        )
    else:
        print(f"  [WARN] ZeroWaste non trouvé : {zerowaste_dir}")

    if raw_ok:
        train_parts.append(
            GarbageDataset(raw_dir, mask_dir, n_per_class=n_per_class, train=True)
        )
    elif config.USE_GARBAGE_DS:
        print(f"  [WARN] Garbage Classification non trouvé : {raw_dir}")
    else:
        print("  [INFO] Garbage Classification désactivé (USE_GARBAGE_DS=False)")

    if not train_parts:
        raise RuntimeError("Aucune source de données trouvée.")

    train_ds = ConcatDataset(train_parts)

    # ── Val / Test ─────────────────────────────────────────────────────────────
    if zw_ok:
        val_ds  = ZeroWasteDataset(zerowaste_dir, "val",  n_per_class=None)
        test_ds = ZeroWasteDataset(zerowaste_dir, "test", n_per_class=None)
    else:
        from torch.utils.data import random_split
        full = GarbageDataset(raw_dir, mask_dir, n_per_class=None, train=False)
        n    = len(full)
        nv   = max(1, int(n*0.15)); nt = max(1, int(n*0.15))
        _, val_ds, test_ds = random_split(
            full, [n-nv-nt, nv, nt],
            generator=torch.Generator().manual_seed(seed)
        )

    print(f"\n  Total train : {len(train_ds)} images")
    print(f"  Val         : {len(val_ds)} images")
    print(f"  Test        : {len(test_ds)} images")

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                               num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False,
                               num_workers=num_workers)
    return train_loader, val_loader, test_loader
