"""
evaluate.py
===========
Métriques de segmentation sémantique et analyse de l'impact de la densité.

Métriques calculées :
  • mIoU     — mean Intersection over Union (principale métrique)
  • IoU par classe
  • Pixel Accuracy globale
  • Precision, Recall, F1 par classe
  • Dice coefficient moyen

Analyse densité :
  • Comparaison mIoU / F1 entre sparse / medium / dense
  • Matrice de confusion agrégée
  • Visualisations prédictions vs vérité terrain
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix

import config


# ─────────────────────────────────────────────────────────────────────────────
#  Métriques pixel-wise
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray,
                    num_classes: int = config.NUM_CLASSES,
                    ignore_index: int = 255) -> dict:
    """
    Calcule mIoU, pixel accuracy, precision, recall, F1, Dice.

    preds   : (N, H, W) int64
    targets : (N, H, W) int64
    """
    valid      = (targets != ignore_index)
    preds_flat  = preds[valid].flatten()
    targets_flat = targets[valid].flatten()

    # ── IoU par classe ────────────────────────────────────────────────────────
    iou_per_class = []
    precision_list, recall_list, f1_list, dice_list = [], [], [], []

    for c in range(num_classes):
        tp = ((preds_flat == c) & (targets_flat == c)).sum()
        fp = ((preds_flat == c) & (targets_flat != c)).sum()
        fn = ((preds_flat != c) & (targets_flat == c)).sum()

        union = tp + fp + fn
        iou   = tp / (union + 1e-9)

        prec  = tp / (tp + fp + 1e-9)
        rec   = tp / (tp + fn + 1e-9)
        f1    = 2 * prec * rec / (prec + rec + 1e-9)
        dice  = 2 * tp / (2 * tp + fp + fn + 1e-9)

        # N'inclure dans le mIoU que si la classe existe dans le GT
        if (targets_flat == c).sum() > 0 or (preds_flat == c).sum() > 0:
            iou_per_class.append(float(iou))
        else:
            iou_per_class.append(float("nan"))

        precision_list.append(float(prec))
        recall_list.append(float(rec))
        f1_list.append(float(f1))
        dice_list.append(float(dice))

    miou   = float(np.nanmean(iou_per_class))
    acc    = float((preds_flat == targets_flat).sum() / (len(targets_flat) + 1e-9))
    m_dice = float(np.nanmean([d for d, i in zip(dice_list, iou_per_class)
                                if not np.isnan(i)]))

    return {
        "miou":      miou,
        "acc":       acc,
        "m_dice":    m_dice,
        "iou":       iou_per_class,
        "precision": precision_list,
        "recall":    recall_list,
        "f1":        f1_list,
        "dice":      dice_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Évaluation d'un DataLoader
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict:
    model.eval()
    all_preds, all_targets = [], []

    for images, masks, _ in tqdm(loader, leave=False, desc="eval"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(masks.numpy())

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return compute_metrics(all_preds, all_targets)


# ─────────────────────────────────────────────────────────────────────────────
#  Analyse par densité
# ─────────────────────────────────────────────────────────────────────────────

def analyze_density_impact(model, test_loaders: dict, device,
                            save_dir: str = config.OUTPUT_DIR) -> pd.DataFrame:
    """
    Évalue le modèle sur chaque niveau de densité et produit :
      - Tableau comparatif mIoU / F1 / acc
      - Graphiques barres par densité
      - Matrice de confusion par densité

    Retourne un DataFrame de synthèse.
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []

    for level, loader in test_loaders.items():
        print(f"\n── Évaluation densité '{level}' ──")
        m = evaluate_loader(model, loader, device)
        row = {"density": level, "miou": m["miou"], "acc": m["acc"], "m_dice": m["m_dice"]}
        for i, cls in enumerate(config.CLASS_NAMES):
            row[f"iou_{cls}"]  = m["iou"][i]  if i < len(m["iou"])  else float("nan")
            row[f"f1_{cls}"]   = m["f1"][i]   if i < len(m["f1"])   else float("nan")
        rows.append(row)
        print(f"  mIoU={m['miou']:.4f}  acc={m['acc']:.4f}  Dice={m['m_dice']:.4f}")
        for cls, iou in zip(config.CLASS_NAMES, m["iou"]):
            print(f"    {cls:<12} IoU = {iou:.4f}")

        # Matrice de confusion pour ce niveau
        _plot_confusion(model, loader, device, level, save_dir)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "density_analysis.csv"), index=False)

    # ── Graphiques de comparaison ─────────────────────────────────────────────
    _plot_density_comparison(df, save_dir)
    _plot_per_class_iou(df, save_dir)

    print(f"\nRésultats sauvegardés dans {save_dir}")
    return df


def _plot_density_comparison(df: pd.DataFrame, save_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics   = ["miou", "acc", "m_dice"]
    titles    = ["mIoU", "Pixel Accuracy", "Dice moyen"]
    colors    = ["#3498db", "#2ecc71", "#e74c3c"]

    for ax, met, title, col in zip(axes, metrics, titles, colors):
        bars = ax.bar(df["density"], df[met], color=col, width=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Densité")
        ax.bar_label(bars, fmt="%.3f", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Impact de la densité sur les performances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "density_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()


def _plot_per_class_iou(df: pd.DataFrame, save_dir: str) -> None:
    iou_cols  = [c for c in df.columns if c.startswith("iou_")]
    class_names = [c.replace("iou_", "") for c in iou_cols]

    x = np.arange(len(class_names))
    w = 0.25
    colors = ["#3498db", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, row in df.iterrows():
        vals = [row.get(c, 0) for c in iou_cols]
        ax.bar(x + i * w, vals, w, label=row["density"], color=colors[i % len(colors)])

    ax.set_xticks(x + w)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1)
    ax.set_title("IoU par classe et par densité", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_iou.png"), dpi=120, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def _plot_confusion(model, loader, device, level: str, save_dir: str) -> None:
    """Génère la matrice de confusion normalisée pour un niveau de densité."""
    model.eval()
    all_preds, all_targets = [], []
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        preds  = model(images).argmax(dim=1).cpu().numpy()
        all_preds.append(preds.flatten())
        all_targets.append(masks.numpy().flatten())

    p = np.concatenate(all_preds)
    t = np.concatenate(all_targets)

    valid = t != 255
    cm    = confusion_matrix(t[valid], p[valid],
                              labels=list(range(config.NUM_CLASSES)),
                              normalize="true")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", ax=ax,
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES,
                cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Prédit");  ax.set_ylabel("Vrai")
    ax.set_title(f"Matrice de confusion — densité '{level}' (normalisée)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confusion_{level}.png"), dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation qualitative
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def visualize_predictions(model, loader, device,
                           n: int = 6, save_dir: str = config.OUTPUT_DIR) -> None:
    """
    Affiche n exemples : image | masque GT | prédiction | overlay.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    color_arr = np.array(config.CLASS_COLORS, dtype=np.uint8)   # (C, 3)

    count = 0
    for images, masks, levels in loader:
        images_d = images.to(device, non_blocking=True)
        logits   = model(images_d)
        preds    = logits.argmax(dim=1).cpu().numpy()

        for i in range(images.shape[0]):
            if count >= n:
                return

            # Dénormaliser l'image
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img_np = images[i].permute(1, 2, 0).numpy()
            img_np = np.clip(img_np * std + mean, 0, 1)

            gt_mask   = masks[i].numpy()
            pred_mask = preds[i]

            # Carte couleur
            gt_color   = color_arr[gt_mask.clip(0, len(color_arr) - 1)]
            pred_color = color_arr[pred_mask.clip(0, len(color_arr) - 1)]

            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            axes[0].imshow(img_np);                 axes[0].set_title("Image")
            axes[1].imshow(gt_color);               axes[1].set_title("GT sémantique")
            axes[2].imshow(pred_color);             axes[2].set_title("Prédiction")
            axes[3].imshow(img_np)
            axes[3].imshow(pred_color, alpha=0.5);  axes[3].set_title("Overlay")
            for ax in axes: ax.axis("off")

            patches = [mpatches.Patch(color=np.array(c) / 255, label=n)
                       for c, n in zip(config.CLASS_COLORS, config.CLASS_NAMES)]
            fig.legend(handles=patches, loc="lower center", ncol=config.NUM_CLASSES,
                       fontsize=8, bbox_to_anchor=(0.5, -0.05))

            level_str = levels[i] if isinstance(levels[i], str) else levels[i]
            plt.suptitle(f"Densité : {level_str}", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"pred_vis_{count:03d}.png"),
                        dpi=100, bbox_inches="tight")
            plt.close()
            count += 1


# ─────────────────────────────────────────────────────────────────────────────
#  Script principal
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluation(checkpoint_path: str = None) -> None:
    from model import UNetResNet50
    from dataset import make_dataloaders

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    model = UNetResNet50(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Modèle chargé depuis {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    _, val_loader, test_loaders = make_dataloaders(config.SCENE_DIR)

    print("\n═══ Analyse par densité ═══")
    df = analyze_density_impact(model, test_loaders, device)
    print("\n", df.to_string(index=False))

    print("\n═══ Visualisations ═══")
    for level, loader in test_loaders.items():
        visualize_predictions(model, loader, device, n=3,
                               save_dir=os.path.join(config.OUTPUT_DIR, f"vis_{level}"))


if __name__ == "__main__":
    full_evaluation()
