"""
train_zerowaste.py
==================
Entraînement sur le dataset ZeroWaste-f (masques déjà fournis).

Structure ZeroWaste attendue :
  zerowaste-f-final/splits_final_deblurred/
    train/  data/*.jpg   sem_seg/*.png
    val/    data/*.jpg   sem_seg/*.png
    test/   data/*.jpg   sem_seg/*.png

Classes (5) : 0=background 1=rigid_plastic 2=cardboard 3=metal 4=soft_plastic
"""

import os, random, time, csv
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROOT       = "./zerowaste-f-final/splits_final_deblurred/"
OUTPUT_DIR = "./outputs_zerowaste"
CKPT_PATH  = "./outputs_zerowaste/best_model.pth"

IMG_SIZE    = 256
BATCH_SIZE  = 8
NUM_EPOCHS  = 40
LR          = 3e-4
WEIGHT_DECAY= 1e-4
NUM_WORKERS = 0        # 0 = stable sur Windows/CPU
PATIENCE    = 8
SEED        = 42

N_CLASSES   = 5
CLASS_NAMES = ["background", "rigid_plastic", "cardboard", "metal", "soft_plastic"]
CLASS_COLORS= [             # RGB pour matplotlib
    (30,  30,  30),         # background  — noir
    (231, 76,  60),         # rigid_plastic — rouge
    (139, 90,  43),         # cardboard   — brun
    (149, 165, 166),        # metal       — gris argent
    (52,  152, 219),        # soft_plastic — bleu
]

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if (hasattr(torch.backends,"mps") and torch.backends.mps.is_available())
          else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)


def _augment(img_rgb: np.ndarray, mask: np.ndarray, split: str):
    """Augmentations manuelles cohérentes image+masque."""
    if split != "train":
        return img_rgb, mask

    # Flip horizontal
    if random.random() < 0.5:
        img_rgb = img_rgb[:, ::-1].copy()
        mask    = mask[:, ::-1].copy()

    # Flip vertical
    if random.random() < 0.3:
        img_rgb = img_rgb[::-1].copy()
        mask    = mask[::-1].copy()

    # Rotation 90°
    if random.random() < 0.4:
        k = random.randint(1, 3)
        img_rgb = np.rot90(img_rgb, k).copy()
        mask    = np.rot90(mask,    k).copy()

    # ColorJitter (image seulement)
    if random.random() < 0.6:
        alpha = random.uniform(0.7, 1.3)   # brightness
        beta  = random.uniform(0.7, 1.3)   # contrast
        img_rgb = np.clip(img_rgb.astype(np.float32) * alpha * beta, 0, 255).astype(np.uint8)

    # Flou gaussien léger
    if random.random() < 0.2:
        img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), 0)

    return img_rgb, mask


class ZeroWasteDataset(Dataset):
    def __init__(self, root, split="train", img_size=IMG_SIZE,
                 n_per_class=30):
        """
        n_per_class : nb max d'images par classe dominante.
        La classe dominante d'une image = classe la plus fréquente dans son
        masque (hors background). Cela garantit une représentation équilibrée
        entre rigid_plastic, cardboard, metal et soft_plastic.
        """
        self.img_dir  = os.path.join(root, split, "data")
        self.mask_dir = os.path.join(root, split, "sem_seg")
        self.split    = split
        self.size     = img_size

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Dossier introuvable : {self.img_dir}")

        all_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # ── Sous-échantillonnage équilibré : n_per_class par classe ──────────
        if n_per_class is not None:
            self.files = self._balanced_sample(all_files, n_per_class)
        else:
            self.files = all_files

        total = len(self.files)
        print(f"  {split:<6} : {total} images ({n_per_class}/classe max)")

    def _dominant_class(self, mask_path: str) -> int:
        """Retourne la classe la plus représentée dans le masque (hors fond=0)."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 0
        counts = np.bincount(mask.flatten(), minlength=N_CLASSES)
        counts[0] = 0          # ignorer le fond
        return int(counts.argmax()) if counts.max() > 0 else 0

    def _balanced_sample(self, all_files, n_per_class):
        """
        Groupe les images par classe dominante et tire n_per_class au hasard
        dans chaque groupe. Affiche la distribution résultante.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        print(f"    Analyse des masques pour équilibrage …")
        for fname in all_files:
            mask_path = os.path.join(
                self.mask_dir, os.path.splitext(fname)[0] + ".png"
            )
            cls = self._dominant_class(mask_path)
            groups[cls].append(fname)

        selected = []
        for cls in range(N_CLASSES):
            g = groups[cls]
            n = min(n_per_class, len(g))
            chosen = random.sample(g, n)
            selected.extend(chosen)
            if n > 0:
                print(f"    Classe {cls} ({CLASS_NAMES[cls]:<16}) : {n}/{len(g)} images")

        random.shuffle(selected)
        return selected

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, name)),
                           cv2.COLOR_BGR2RGB)

        mask_name = os.path.splitext(name)[0] + ".png"
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name),
                          cv2.IMREAD_GRAYSCALE)

        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        img  = cv2.resize(img,  (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size),
                          interpolation=cv2.INTER_NEAREST)

        img, mask = _augment(img, mask, self.split)

        # Normalisation ImageNet
        img_f = img.astype(np.float32) / 255.0
        img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1))
        msk_t = torch.from_numpy(mask.copy()).long()
        return img_t, msk_t


# ─────────────────────────────────────────────────────────────────────────────
#  MODÈLE : U-Net ResNet-18
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, in_c // 2, 2, stride=2)
        self.conv = ConvBnRelu(in_c // 2 + skip_c, out_c)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet18(nn.Module):
    """U-Net avec encodeur ResNet-18 pré-entraîné sur ImageNet."""

    def __init__(self, num_classes=N_CLASSES, pretrained=True):
        super().__init__()
        bb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        self.enc1 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)  # /2   64ch
        self.pool = bb.maxpool
        self.enc2 = bb.layer1   # /4   64ch
        self.enc3 = bb.layer2   # /8  128ch
        self.enc4 = bb.layer3   # /16 256ch
        self.enc5 = bb.layer4   # /32 512ch

        self.bot  = ConvBnRelu(512, 256)
        self.dec5 = DecoderBlock(256, 256, 128)
        self.dec4 = DecoderBlock(128, 128, 64)
        self.dec3 = DecoderBlock(64,   64, 64)
        self.dec2 = DecoderBlock(64,   64, 32)
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
        b  = self.bot(s5)
        x  = self.dec5(b,  s4)
        x  = self.dec4(x,  s3)
        x  = self.dec3(x,  s2)
        x  = self.dec2(x,  s1)
        x  = self.dec1(x)
        return self.head(x)

    def param_groups(self, lr):
        """Differential LR : encodeur à lr/10."""
        enc = (list(self.enc1.parameters()) + list(self.enc2.parameters()) +
               list(self.enc3.parameters()) + list(self.enc4.parameters()) +
               list(self.enc5.parameters()))
        dec = (list(self.bot.parameters())  + list(self.dec5.parameters()) +
               list(self.dec4.parameters()) + list(self.dec3.parameters()) +
               list(self.dec2.parameters()) + list(self.dec1.parameters()) +
               list(self.head.parameters()))
        return [{"params": enc, "lr": lr/10, "initial_lr": lr/10},
                {"params": dec, "lr": lr,    "initial_lr": lr}]


# ─────────────────────────────────────────────────────────────────────────────
#  LOSS : Focal + Dice
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=255):
        super().__init__()
        self.gamma  = gamma
        self.alpha  = alpha
        self.ignore = ignore_index

    def forward(self, logits, targets):
        valid = targets != self.ignore
        lv    = logits.permute(0,2,3,1)[valid]
        tv    = targets[valid]
        log_p = F.log_softmax(lv, dim=1)
        pt    = log_p.exp().gather(1, tv.unsqueeze(1)).squeeze(1)
        log_pt= log_p.gather(1, tv.unsqueeze(1)).squeeze(1)
        fw    = (1 - pt) ** self.gamma
        if self.alpha is not None:
            fw = fw * self.alpha.to(logits.device)[tv]
        return -(fw * log_pt).mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, num_classes=N_CLASSES):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=class_weights)
        self.nc    = num_classes

    def forward(self, logits, targets):
        return 0.5 * self.focal(logits, targets) + 0.5 * self._dice(logits, targets)

    def _dice(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        oh    = F.one_hot(targets.clamp(0, self.nc-1), self.nc).permute(0,3,1,2).float()
        inter = (probs * oh).sum(dim=(0,2,3))
        union = probs.sum(dim=(0,2,3)) + oh.sum(dim=(0,2,3))
        return (1 - (2*inter + 1e-6) / (union + 1e-6))[1:].mean()


def compute_class_weights(dataset, num_classes=N_CLASSES, max_samples=200):
    print("  Calcul poids de classe …")
    counts = np.zeros(num_classes, np.float64)
    indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))
    for i in indices:
        _, mask = dataset[i]
        for c in range(num_classes):
            counts[c] += (mask.numpy() == c).sum()
    freq = counts / (counts.sum() + 1e-9)
    w    = 1.0 / (freq + 1e-6)
    w    = w / w.sum() * num_classes
    print(f"  Poids : {np.round(w, 2)}")
    return torch.tensor(w, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds, targets, num_classes=N_CLASSES):
    """preds, targets : np.ndarray (N,H,W)"""
    p = preds.flatten()
    t = targets.flatten()
    iou_list, f1_list = [], []
    for c in range(num_classes):
        tp = ((p==c)&(t==c)).sum()
        fp = ((p==c)&(t!=c)).sum()
        fn = ((p!=c)&(t==c)).sum()
        iou = tp/(tp+fp+fn+1e-9)
        f1  = 2*tp/(2*tp+fp+fn+1e-9)
        if (t==c).sum()>0 or (p==c).sum()>0:
            iou_list.append(float(iou))
        else:
            iou_list.append(float("nan"))
        f1_list.append(float(f1))
    miou = float(np.nanmean(iou_list))
    acc  = float((p==t).sum()/(len(t)+1e-9))
    miou_no_bg = float(np.nanmean(iou_list[1:]))
    return {"miou": miou, "miou_no_bg": miou_no_bg, "acc": acc,
            "iou": iou_list, "f1": f1_list}


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosine:
    def __init__(self, opt, warmup, total, min_lr=1e-6):
        self.opt=opt; self.w=warmup; self.T=total; self.min_lr=min_lr
        self.base = opt.param_groups[-1]["initial_lr"]

    def step(self, epoch):
        if epoch < self.w:
            scale = (epoch+1)/self.w
        else:
            p     = (epoch-self.w)/max(1,self.T-self.w)
            scale = self.min_lr/self.base + 0.5*(1-self.min_lr/self.base)*(1+np.cos(np.pi*p))
        for pg in self.opt.param_groups:
            pg["lr"] = pg["initial_lr"] * scale
        return self.opt.param_groups[-1]["lr"]


# ─────────────────────────────────────────────────────────────────────────────
#  BOUCLE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, leave=False, desc="train" if train else "val  ",
                    dynamic_ncols=True)
        for img, mask in pbar:
            img  = img.to(device)
            mask = mask.to(device)
            logits = model(img)
            loss   = criterion(logits, mask)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            all_preds.append(logits.detach().argmax(1).cpu().numpy())
            all_targets.append(mask.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    m = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))
    m["loss"] = total_loss / len(loader)
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────

def train():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(DEVICE)
    if device.type == "cpu":
        n = os.cpu_count() or 1
        torch.set_num_threads(n)
        print(f"CPU : {n} threads")
    else:
        torch.backends.cudnn.benchmark = True
    print(f"Device : {device}")

    print("Chargement datasets …")
    # 30 images par classe sur train ; val et test sans limite (évaluation complète)
    train_ds = ZeroWasteDataset(ROOT, "train", n_per_class=30)
    val_ds   = ZeroWasteDataset(ROOT, "val",   n_per_class=None)
    test_ds  = ZeroWasteDataset(ROOT, "test",  n_per_class=None)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS)

    cls_w     = compute_class_weights(train_ds).to(device)
    model     = UNetResNet18(num_classes=N_CLASSES, pretrained=True).to(device)
    criterion = CombinedLoss(class_weights=cls_w).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modèle : {total/1e6:.1f}M params | {trainable/1e6:.1f}M entraînables")

    param_groups = model.param_groups(LR)
    optimizer    = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler    = WarmupCosine(optimizer, warmup=4, total=NUM_EPOCHS)

    # Log CSV
    log_f  = open(os.path.join(OUTPUT_DIR, "log.csv"), "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=[
        "epoch","lr","train_loss","val_loss",
        "train_miou","val_miou","train_miou_no_bg","val_miou_no_bg","time_s"
    ])
    writer.writeheader()

    best_miou = 0.0; patience_ct = 0; epoch_times = []
    history   = {"tl":[],"vl":[],"tm":[],"vm":[]}

    sep = "─"*80
    print(f"\n{sep}")
    print(f"{'Ep':>4} {'LR':>8} {'TrLoss':>8} {'VlLoss':>8} "
          f"{'TrmIoU':>8} {'VlmIoU':>8} {'VlFG':>7} {'Time':>6} {'ETA':>7}")
    print(sep)

    for epoch in range(1, NUM_EPOCHS+1):
        t0     = time.time()
        lr_now = scheduler.step(epoch-1)
        tr     = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl     = run_epoch(model, val_loader,   criterion, None,      device, False)
        elapsed = time.time()-t0
        epoch_times.append(elapsed)
        eta_s   = np.mean(epoch_times[-5:]) * (NUM_EPOCHS-epoch)
        eta_str = f"{int(eta_s//60)}m{int(eta_s%60):02d}s"

        mark = "★" if vl["miou"] > best_miou else " "
        print(f"{epoch:>4} {lr_now:>8.2e} {tr['loss']:>8.4f} {vl['loss']:>8.4f} "
              f"{tr['miou']:>8.4f} {vl['miou']:>8.4f} {vl['miou_no_bg']:>7.4f} "
              f"{elapsed:>5.1f}s {eta_str:>7} {mark}")

        writer.writerow({"epoch":epoch,"lr":round(lr_now,8),
                         "train_loss":round(tr["loss"],5),"val_loss":round(vl["loss"],5),
                         "train_miou":round(tr["miou"],5),"val_miou":round(vl["miou"],5),
                         "train_miou_no_bg":round(tr["miou_no_bg"],5),
                         "val_miou_no_bg":round(vl["miou_no_bg"],5),
                         "time_s":round(elapsed,1)})
        log_f.flush()

        for k,v in [("tl",tr["loss"]),("vl",vl["loss"]),
                    ("tm",tr["miou"]),("vm",vl["miou"])]:
            history[k].append(v)

        if vl["miou"] > best_miou:
            best_miou = vl["miou"]; patience_ct = 0
            torch.save({"epoch":epoch,"state":model.state_dict(),
                        "miou":best_miou}, CKPT_PATH)
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"\nEarly stopping (epoch {epoch})")
                break

    log_f.close()
    total_min = sum(epoch_times)/60
    print(f"{sep}")
    print(f"Terminé en {total_min:.1f} min | Meilleur val mIoU = {best_miou:.4f}")

    _plot_curves(history)
    evaluate(model, test_loader, device)
    visualize(model, test_ds, device, n=8)


# ─────────────────────────────────────────────────────────────────────────────
#  ÉVALUATION FINALE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for img, mask in tqdm(loader, desc="test ", leave=False):
        logits = model(img.to(device))
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_targets.append(mask.numpy())

    m = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))

    print("\n═══ Résultats Test ═══")
    print(f"  mIoU global    : {m['miou']:.4f}")
    print(f"  mIoU sans fond : {m['miou_no_bg']:.4f}")
    print(f"  Pixel accuracy : {m['acc']:.4f}")
    print(f"\n  {'Classe':<16} {'IoU':>7}  {'F1':>7}")
    print("  " + "─"*35)
    for name, iou, f1 in zip(CLASS_NAMES, m["iou"], m["f1"]):
        bar = "█" * int((iou if not np.isnan(iou) else 0) * 20)
        print(f"  {name:<16} {iou:>7.4f}  {f1:>7.4f}  {bar}")

    # Sauvegarde
    with open(os.path.join(OUTPUT_DIR, "test_results.txt"), "w") as f:
        f.write(f"mIoU: {m['miou']:.4f}\nmIoU sans fond: {m['miou_no_bg']:.4f}\n"
                f"Accuracy: {m['acc']:.4f}\n\n")
        for name, iou, f1 in zip(CLASS_NAMES, m["iou"], m["f1"]):
            f.write(f"{name}: IoU={iou:.4f} F1={f1:.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def visualize(model, dataset, device, n=8):
    model.eval()
    color_arr = np.array(CLASS_COLORS, dtype=np.uint8)
    indices   = random.sample(range(len(dataset)), min(n, len(dataset)))

    fig, axes = plt.subplots(n, 4, figsize=(18, 4*n))
    if n == 1: axes = axes[np.newaxis]

    for row, idx in enumerate(indices):
        img_t, mask_t = dataset[idx]
        logits = model(img_t.unsqueeze(0).to(device))
        pred   = logits.argmax(1)[0].cpu().numpy()
        mask_np= mask_t.numpy()

        # Dénormaliser l'image pour affichage
        img_np = img_t.permute(1,2,0).numpy()
        img_np = np.clip(img_np * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

        gt_color   = color_arr[mask_np.clip(0, N_CLASSES-1)]
        pred_color = color_arr[pred.clip(0, N_CLASSES-1)]
        overlay    = (img_np*255*0.5 + pred_color*0.5).astype(np.uint8)

        iou = compute_metrics(pred[None], mask_np[None])["miou_no_bg"]

        axes[row,0].imshow(img_np);              axes[row,0].set_title("Image")
        axes[row,1].imshow(gt_color);            axes[row,1].set_title("GT")
        axes[row,2].imshow(pred_color);          axes[row,2].set_title(f"Préd (mIoU={iou:.3f})")
        axes[row,3].imshow(overlay);             axes[row,3].set_title("Overlay")
        for ax in axes[row]: ax.axis("off")

    patches = [mpatches.Patch(color=np.array(c)/255, label=n)
               for c,n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=N_CLASSES,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    plt.suptitle("Prédictions sur le test set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "predictions.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Visualisations → {path}")


def _plot_curves(h):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h["tl"], label="train", lw=2); axes[0].plot(h["vl"], label="val", lw=2)
    axes[0].set_title("Loss", fontweight="bold"); axes[0].legend()
    axes[1].plot(h["tm"], label="train", lw=2); axes[1].plot(h["vm"], label="val", lw=2)
    axes[1].set_title("mIoU", fontweight="bold"); axes[1].legend()
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=110)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  INFÉRENCE SEULE (après entraînement)
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(image_path: str, checkpoint: str = CKPT_PATH):
    """Prédit et affiche la segmentation d'une image individuelle."""
    device = torch.device(DEVICE)
    model  = UNetResNet18(pretrained=False).to(device)
    ckpt   = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img_r = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    inp   = torch.from_numpy(
        ((img_r/255.0 - IMAGENET_MEAN)/IMAGENET_STD).transpose(2,0,1)
    ).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(inp).argmax(1)[0].cpu().numpy()

    color_arr  = np.array(CLASS_COLORS, dtype=np.uint8)
    pred_color = color_arr[pred]
    overlay    = (img_r*0.5 + pred_color*0.5).astype(np.uint8)

    dominant   = CLASS_NAMES[np.bincount(pred.flatten(), minlength=N_CLASSES)[1:].argmax() + 1]
    print(f"Classe dominante : {dominant}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(img_r);       axes[0].set_title("Original")
    axes[1].imshow(pred_color);  axes[1].set_title("Segmentation")
    axes[2].imshow(overlay);     axes[2].set_title(f"Overlay — {dominant}")
    for ax in axes: ax.axis("off")
    patches = [mpatches.Patch(color=np.array(c)/255, label=n)
               for c,n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=N_CLASSES, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "single_pred.png"), dpi=110)
    plt.close()
    return pred


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Mode inférence : python train_zerowaste.py chemin/image.jpg
        predict_image(sys.argv[1])
    else:
        train()
