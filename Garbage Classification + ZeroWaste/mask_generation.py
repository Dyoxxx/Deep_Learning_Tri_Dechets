"""
mask_generation.py
==================
Génère des masques sémantiques pour le dataset Garbage Classification
(images de déchets individuels sur fond blanc).

Les masques produits utilisent les 5 classes ZeroWaste :
  0=background  1=rigid_plastic  2=cardboard  3=metal  4=soft_plastic

Seules les classes mappables sont traitées (trash=None est ignoré).
"""

import os, cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config


def generate_binary_mask(img_bgr: np.ndarray,
                          class_name: str = None) -> np.ndarray:
    """Masque binaire 255=objet / 0=fond par seuillage BGR direct."""
    thresh = config.BGR_THRESHOLDS.get(
        class_name.lower() if class_name else "",
        config.DEFAULT_BGR_THRESH
    )
    smooth = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    b, g, r = cv2.split(smooth)
    bg  = ((b.astype(np.int16) > thresh) &
           (g.astype(np.int16) > thresh) &
           (r.astype(np.int16) > thresh)).astype(np.uint8) * 255
    obj = cv2.bitwise_not(bg)

    # Fermeture : bouche les trous intérieurs (reflets, zones claires)
    k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, k, iterations=3)
    k2  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, k2, iterations=1)

    # Remplissage des trous via flood fill inversé
    h, w   = obj.shape
    inv    = cv2.bitwise_not(obj)
    canvas = np.zeros((h+2, w+2), np.uint8)
    inv_c  = inv.copy()
    cv2.floodFill(inv_c, canvas, (0, 0), 128)
    obj = cv2.bitwise_not((inv_c == 128).astype(np.uint8) * 255)

    # Forcer les bords à fond
    obj[:3,:] = obj[-3:,:] = obj[:,:3] = obj[:,-3:] = 0

    # Garder le plus grand composant
    n, labels, stats, _ = cv2.connectedComponentsWithStats(obj, connectivity=8)
    if n > 1:
        lg  = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out = np.zeros_like(obj)
        out[labels == lg] = 255
        obj = out

    # Fallback Canny si masque vide
    if (obj > 127).sum() / (h * w) < 0.01:
        obj = _canny_fallback(img_bgr, obj)

    return obj


def _canny_fallback(img_bgr, current):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 20, 80)
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    edges = cv2.dilate(edges, k, iterations=4)
    h, w  = edges.shape
    canvas= np.zeros((h+2,w+2), np.uint8)
    inv   = cv2.bitwise_not(edges)
    cv2.floodFill(inv, canvas, (0,0), 128)
    filled = cv2.bitwise_not((inv==128).astype(np.uint8)*255)
    filled = cv2.morphologyEx(filled,cv2.MORPH_CLOSE,k,iterations=3)
    n,labels,stats,_ = cv2.connectedComponentsWithStats(filled,connectivity=8)
    if n > 1:
        lg = 1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
        out = np.zeros_like(filled); out[labels==lg]=255; filled=out
    ratio = (filled>127).sum()/max(filled.size,1)
    return filled if 0.01 < ratio < 0.95 else current


def generate_masks_for_dataset(raw_dir=config.RAW_DATA_DIR,
                                mask_dir=config.MASK_DIR,
                                n_per_class=config.N_PER_CLASS,
                                visualize_n=2):
    """
    Parcourt raw_dir (un sous-dossier par classe) et génère les masques
    sémantiques dans mask_dir.
    Limite à n_per_class images par classe. Ignore les dossiers avec FOLDER_TO_CLASS=None.
    """
    import random
    raw_path  = Path(raw_dir)
    mask_path = Path(mask_dir)
    mask_path.mkdir(parents=True, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    summary = {}
    for class_folder in sorted(raw_path.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name.lower()
        class_idx  = config.FOLDER_TO_CLASS.get(class_name)

        if class_idx is None:
            print(f"[SKIP] '{class_folder.name}' → pas de classe équivalente")
            continue

        all_files = sorted([
            f for f in class_folder.iterdir()
            if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}
        ])
        # Limite à n_per_class
        files = random.sample(all_files, min(n_per_class, len(all_files)))

        out_folder = mask_path / class_folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n── {class_name} → classe {class_idx} "
              f"({config.CLASS_NAMES[class_idx]}) | {len(files)}/{len(all_files)} images")

        ratios = []; vis_count = 0
        for img_file in tqdm(files, desc=class_name):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            binary   = generate_binary_mask(img, class_name)
            semantic = np.where(binary > 127, class_idx, 0).astype(np.uint8)
            cv2.imwrite(str(out_folder / (img_file.stem + ".png")), semantic)

            ratios.append((binary > 127).sum() / (img.shape[0]*img.shape[1]))
            if vis_count < visualize_n:
                _save_vis(img, binary, semantic, class_name, img_file.stem, class_idx)
                vis_count += 1

        mean_r = np.mean(ratios) if ratios else 0
        flag   = "✓" if 0.03 < mean_r < 0.90 else "⚠"
        print(f"   ratio objet moyen : {mean_r:.1%} {flag}")
        summary[class_name] = (class_idx, mean_r, len(files))

    print("\n═══ Résumé ═══")
    for cls,(idx,r,n) in summary.items():
        bar  = "█" * int(r*30)
        flag = "✓" if 0.03<r<0.90 else "⚠"
        print(f"  {cls:<12} → {config.CLASS_NAMES[idx]:<16} {n:>3} imgs  {r:>6.1%} {flag}  {bar}")


def _save_vis(img_bgr, binary, semantic, class_name, stem, class_idx):
    import matplotlib.pyplot as plt
    overlay = img_bgr.copy()
    color   = config.CLASS_COLORS[class_idx]
    overlay[semantic > 0] = (overlay[semantic>0]*0.3 + np.array(color)*0.7).astype(np.uint8)
    ratio = (binary>127).sum()/(img_bgr.shape[0]*img_bgr.shape[1])*100

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(cv2.cvtColor(img_bgr,  cv2.COLOR_BGR2RGB)); axes[0].set_title("Original")
    axes[1].imshow(binary, cmap="gray");                        axes[1].set_title(f"Masque ({ratio:.1f}%)")
    axes[2].imshow(cv2.cvtColor(overlay,  cv2.COLOR_BGR2RGB)); axes[2].set_title(f"Classe : {config.CLASS_NAMES[class_idx]}")
    for ax in axes: ax.axis("off")
    plt.suptitle(f"{class_name} — {stem}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, f"mask_{class_name}_{stem}.png"),
                dpi=90, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    generate_masks_for_dataset()
