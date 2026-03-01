"""
evaluate.py — métriques et évaluation, 5 classes ZeroWaste
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import config


def compute_metrics(preds: np.ndarray, targets: np.ndarray,
                    num_classes: int = config.NUM_CLASSES) -> dict:
    """
    preds, targets : np.ndarray (N, H, W) ou (N*H*W,)
    Retourne mIoU, accuracy, IoU/F1 par classe.
    """
    p = preds.flatten()
    t = targets.flatten()

    iou_list, f1_list = [], []
    for c in range(num_classes):
        tp = ((p == c) & (t == c)).sum()
        fp = ((p == c) & (t != c)).sum()
        fn = ((p != c) & (t == c)).sum()
        iou = tp / (tp + fp + fn + 1e-9)
        f1  = 2 * tp / (2 * tp + fp + fn + 1e-9)
        iou_list.append(float(iou) if ((t==c).sum()+(p==c).sum()) > 0 else float("nan"))
        f1_list.append(float(f1))

    miou      = float(np.nanmean(iou_list))
    miou_fg   = float(np.nanmean(iou_list[1:]))   # sans fond
    acc       = float((p == t).sum() / (len(t) + 1e-9))

    return {
        "miou":    miou,
        "miou_fg": miou_fg,
        "acc":     acc,
        "iou":     iou_list,
        "f1":      f1_list,
        "loss":    0.0,   # rempli par run_epoch si besoin
    }


@torch.no_grad()
def full_evaluation(checkpoint_path: str = None) -> None:
    from model import UNetResNet18
    from dataset import make_dataloaders

    device = torch.device("cpu")
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"[ERREUR] Checkpoint introuvable : {checkpoint_path}")
        print("  → Lancez d'abord : python main.py --step train")
        return

    model = UNetResNet18(pretrained=False).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Modèle chargé (epoch {ckpt.get('epoch','?')}, mIoU={ckpt.get('miou',0):.4f})")

    _, _, test_loader = make_dataloaders()

    all_preds, all_targets = [], []
    for img, mask in tqdm(test_loader, desc="test"):
        logits = model(img.to(device))
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_targets.append(mask.numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    m = compute_metrics(preds, targets)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n═══ Résultats sur le test set ═══")
    print(f"  mIoU global    : {m['miou']:.4f}")
    print(f"  mIoU sans fond : {m['miou_fg']:.4f}")
    print(f"  Pixel accuracy : {m['acc']:.4f}")
    print(f"\n  {'Classe':<16} {'IoU':>7}  {'F1':>7}  Barre")
    print("  " + "─"*50)
    for name, iou, f1 in zip(config.CLASS_NAMES, m["iou"], m["f1"]):
        bar  = "█" * int((iou if not np.isnan(iou) else 0) * 25)
        flag = "" if not np.isnan(iou) else " (absent du test)"
        print(f"  {name:<16} {iou:>7.4f}  {f1:>7.4f}  {bar}{flag}")

    # Sauvegarde texte
    with open(os.path.join(config.OUTPUT_DIR, "test_results.txt"), "w") as f:
        f.write(f"mIoU: {m['miou']:.4f}\n")
        f.write(f"mIoU sans fond: {m['miou_fg']:.4f}\n")
        f.write(f"Accuracy: {m['acc']:.4f}\n\n")
        for name, iou, f1 in zip(config.CLASS_NAMES, m["iou"], m["f1"]):
            f.write(f"{name}: IoU={iou:.4f}  F1={f1:.4f}\n")

    # Matrice de confusion
    _plot_confusion(preds, targets)

    # Visualisations
    _visualize(model, test_loader, device)


def _plot_confusion(preds, targets):
    p = preds.flatten(); t = targets.flatten()
    cm = confusion_matrix(t, p, labels=list(range(config.NUM_CLASSES)), normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", ax=ax,
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES,
                cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
    ax.set_title("Matrice de confusion (normalisée)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix → {config.OUTPUT_DIR}/confusion_matrix.png")


@torch.no_grad()
def _visualize(model, loader, device, n=6):
    model.eval()
    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])
    color_arr = np.array(config.CLASS_COLORS, dtype=np.uint8)

    collected = []
    for img, mask in loader:
        logits = model(img.to(device))
        preds  = logits.argmax(1).cpu()
        for i in range(img.shape[0]):
            collected.append((img[i], mask[i], preds[i]))
            if len(collected) >= n: break
        if len(collected) >= n: break

    fig, axes = plt.subplots(n, 4, figsize=(18, 4 * n))
    if n == 1: axes = axes[np.newaxis]

    for row, (img_t, mask_t, pred_t) in enumerate(collected):
        img_np  = np.clip(img_t.permute(1,2,0).numpy() * STD + MEAN, 0, 1)
        mask_np = mask_t.numpy()
        pred_np = pred_t.numpy()

        gt_col  = color_arr[mask_np.clip(0, config.NUM_CLASSES-1)]
        pr_col  = color_arr[pred_np.clip(0, config.NUM_CLASSES-1)]
        overlay = (img_np * 255 * 0.5 + pr_col * 0.5).astype(np.uint8)

        iou = compute_metrics(pred_np[None], mask_np[None])["miou_fg"]
        axes[row,0].imshow(img_np);   axes[row,0].set_title("Image")
        axes[row,1].imshow(gt_col);   axes[row,1].set_title("GT")
        axes[row,2].imshow(pr_col);   axes[row,2].set_title(f"Préd (FG-IoU={iou:.3f})")
        axes[row,3].imshow(overlay);  axes[row,3].set_title("Overlay")
        for ax in axes[row]: ax.axis("off")

    patches = [mpatches.Patch(color=np.array(c)/255, label=nm)
               for c, nm in zip(config.CLASS_COLORS, config.CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=config.NUM_CLASSES,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    plt.suptitle("Prédictions — test set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "predictions.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Visualisations → {path}")


if __name__ == "__main__":
    full_evaluation()
