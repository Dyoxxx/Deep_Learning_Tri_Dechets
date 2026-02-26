"""
diagnose.py
===========
Diagnostics rapides AVANT de relancer l'entraînement.
Lance : python diagnose.py

Vérifie :
  1. Distribution des classes dans les masques générés
  2. Ce que le modèle prédit réellement (histogramme des prédictions)
  3. Qualité des masques par classe (ratio objet/image)
"""

import os, cv2, numpy as np, torch
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config


# ─── 1. Distribution réelle dans les masques ─────────────────────────────────

def check_mask_distribution(scene_dir=config.SCENE_DIR):
    print("\n═══ 1. Distribution des classes dans les masques ═══")
    counts = np.zeros(config.NUM_CLASSES, dtype=np.int64)
    n_masks = 0
    for level in config.DENSITY_LEVELS:
        msk_dir = Path(scene_dir) / level / "masks"
        for f in list(msk_dir.glob("*.png"))[:50]:
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            for c in range(config.NUM_CLASSES):
                counts[c] += (m == c).sum()
            n_masks += 1

    total = counts.sum()
    print(f"  ({n_masks} masques analysés)")
    print(f"  {'Classe':<12} {'Pixels':>10} {'%':>7}  {'Poids idéal':>12}")
    print("  " + "─" * 50)
    weights = []
    for i, (name, cnt) in enumerate(zip(config.CLASS_NAMES, counts)):
        pct  = cnt / total * 100
        w    = (total / (config.NUM_CLASSES * cnt)) if cnt > 0 else 0
        weights.append(w)
        bar  = "█" * int(pct / 2)
        print(f"  {name:<12} {cnt:>10,} {pct:>6.2f}%  w={w:>6.2f}  {bar}")

    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 4))
    colors  = [np.array(c)/255 for c in config.CLASS_COLORS]
    ax.bar(config.CLASS_NAMES, counts / total * 100, color=colors, edgecolor="white")
    ax.set_ylabel("% pixels"); ax.set_title("Distribution des classes dans les masques d'entraînement")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "diag_class_distribution.png"), dpi=110)
    plt.close()
    print(f"  → diag_class_distribution.png")
    return counts


# ─── 2. Ce que le modèle prédit réellement ───────────────────────────────────

def check_model_predictions(checkpoint=None):
    print("\n═══ 2. Biais de prédiction du modèle ═══")
    ckpt_path = checkpoint or os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("  [SKIP] Pas de checkpoint trouvé.")
        return

    from model import UNetResNet18
    device = torch.device("cpu")
    model  = UNetResNet18(pretrained=False, freeze_encoder=False).to(device)
    ckpt   = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pred_counts = np.zeros(config.NUM_CLASSES, dtype=np.int64)
    gt_counts   = np.zeros(config.NUM_CLASSES, dtype=np.int64)
    n = 0

    MEAN = np.array([0.485,0.456,0.406], np.float32)
    STD  = np.array([0.229,0.224,0.225], np.float32)

    with torch.no_grad():
        for level in config.DENSITY_LEVELS:
            img_dir = Path(config.SCENE_DIR) / level / "images"
            msk_dir = Path(config.SCENE_DIR) / level / "masks"
            files   = list(img_dir.glob("*.png"))[:20]
            for f in files:
                img  = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                img  = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
                inp  = torch.from_numpy(
                    ((img/255.0 - MEAN)/STD).transpose(2,0,1)
                ).unsqueeze(0).float()
                pred = model(inp).argmax(1).numpy().flatten()
                gt   = cv2.imread(str(msk_dir/f.name), cv2.IMREAD_GRAYSCALE)
                gt   = cv2.resize(gt, (config.IMG_SIZE, config.IMG_SIZE),
                                  interpolation=cv2.INTER_NEAREST).flatten()
                for c in range(config.NUM_CLASSES):
                    pred_counts[c] += (pred == c).sum()
                    gt_counts[c]   += (gt   == c).sum()
                n += 1

    print(f"  ({n} images testées)")
    print(f"  {'Classe':<12} {'GT%':>7} {'Prédit%':>9}  {'Biais':>8}")
    print("  " + "─" * 50)
    gt_pct   = gt_counts   / gt_counts.sum()   * 100
    pred_pct = pred_counts / pred_counts.sum() * 100
    for name, gp, pp in zip(config.CLASS_NAMES, gt_pct, pred_pct):
        biais = pp - gp
        flag  = "  ← SURPRÉDIT ⚠" if biais > 10 else ("  ← sous-prédit" if biais < -5 else "")
        print(f"  {name:<12} {gp:>6.1f}%  {pp:>7.1f}%  {biais:>+7.1f}%{flag}")

    # Graphique comparatif
    x = np.arange(config.NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, gt_pct,   0.4, label="Vérité terrain", alpha=0.8)
    ax.bar(x + 0.2, pred_pct, 0.4, label="Prédictions modèle", alpha=0.8, color="tomato")
    ax.set_xticks(x); ax.set_xticklabels(config.CLASS_NAMES, rotation=20)
    ax.set_ylabel("% pixels"); ax.set_title("GT vs Prédictions — biais de classe")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "diag_prediction_bias.png"), dpi=110)
    plt.close()
    print(f"  → diag_prediction_bias.png")


# ─── 3. Qualité des masques de génération par classe ────────────────────────

def check_mask_quality(raw_dir=config.RAW_DATA_DIR, n_per_class=15):
    print("\n═══ 3. Qualité des masques par classe ═══")
    from mask_generation import generate_binary_mask

    raw_path = Path(raw_dir)
    stats    = {}

    for folder in sorted(raw_path.iterdir()):
        if not folder.is_dir(): continue
        cls = folder.name.lower()
        if cls not in config.FOLDER_TO_CLASS: continue

        files    = sorted([f for f in folder.iterdir()
                           if f.suffix.lower() in {".jpg",".jpeg",".png"}])[:n_per_class]
        ratios   = []
        for f in files:
            img  = cv2.imread(str(f))
            if img is None: continue
            mask = generate_binary_mask(img)
            h, w = img.shape[:2]
            ratio = (mask > 127).sum() / (h * w)
            ratios.append(ratio)

        if ratios:
            stats[cls] = {"mean": np.mean(ratios), "std": np.std(ratios),
                           "min": np.min(ratios),  "max": np.max(ratios)}
            quality = "✓ OK" if 0.05 < np.mean(ratios) < 0.90 else "⚠ PROBLÈME"
            print(f"  {cls:<12}  ratio objet/image : "
                  f"{np.mean(ratios):.1%} ± {np.std(ratios):.1%}  "
                  f"[{np.min(ratios):.1%} – {np.max(ratios):.1%}]  {quality}")
            if np.mean(ratios) > 0.85:
                print(f"              ↑ Masque trop grand : tout est prédit objet (fond mal soustrait)")
            elif np.mean(ratios) < 0.05:
                print(f"              ↑ Masque trop petit : objet non détecté (couleur proche du fond)")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    counts = check_mask_distribution()
    check_model_predictions()
    check_mask_quality()
    print("\n═══ Recommandations ═══")
    bg_pct = counts[0] / counts.sum() * 100
    if bg_pct > 80:
        print(f"  ⚠ Fond = {bg_pct:.0f}% des pixels → poids fond trop faible dans la loss")
        print("    → Augmentez FOCAL_GAMMA dans config.py (voir fix ci-dessous)")
    dominant = int(np.argmax(counts[1:])) + 1
    dom_pct  = counts[dominant] / counts.sum() * 100
    if dom_pct > 30:
        print(f"  ⚠ Classe '{config.CLASS_NAMES[dominant]}' domine ({dom_pct:.0f}% des pixels objet)")
        print("    → Les poids de classe doivent être recalculés avant de re-entraîner")
    print("\n  → Lancez ensuite : python main.py --step train")
