"""
debug_mask.py — teste différents seuils BGR sur une image
Usage : python debug_mask.py data/raw/glass/img001.jpg
"""

import sys, os, cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config


def test_bgr_thresh(img_bgr, thresh, close_k=21, close_iter=3):
    smooth = cv2.GaussianBlur(img_bgr, (3,3), 0)
    b, g, r = cv2.split(smooth)
    bg   = ((b.astype(np.int16)>thresh) &
            (g.astype(np.int16)>thresh) &
            (r.astype(np.int16)>thresh)).astype(np.uint8)*255
    obj  = cv2.bitwise_not(bg)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(close_k,close_k))
    obj  = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, k, iterations=close_iter)
    k2   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    obj  = cv2.morphologyEx(obj, cv2.MORPH_OPEN,  k2, iterations=1)
    # fill holes
    h,w  = obj.shape
    inv  = cv2.bitwise_not(obj)
    canvas = np.zeros((h+2,w+2),np.uint8)
    inv_c  = inv.copy()
    cv2.floodFill(inv_c, canvas, (0,0), 128)
    real_bg = (inv_c==128).astype(np.uint8)*255
    obj  = cv2.bitwise_not(real_bg)
    # nettoyer bords
    obj[:3,:]=obj[-3:,:]=obj[:,:3]=obj[:,-3:]=0
    # plus grand composant
    n,labels,stats,_ = cv2.connectedComponentsWithStats(obj,connectivity=8)
    if n>1:
        lg = 1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
        out = np.zeros_like(obj); out[labels==lg]=255; obj=out
    return obj


def run_debug(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERREUR] Image introuvable : {image_path}"); sys.exit(1)

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image : {Path(image_path).name}  —  {w}×{h} px")

    # Grille de seuils BGR à tester
    thresholds = [180, 190, 200, 210, 218, 225, 230, 235]

    n_cols = len(thresholds) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))

    axes[0,0].imshow(img_rgb); axes[0,0].set_title("ORIGINAL", fontweight="bold")
    axes[1,0].axis("off"); axes[0,0].axis("off")

    best_thresh = None; best_ratio_dist = float("inf")

    for j, thresh in enumerate(thresholds):
        mask  = test_bgr_thresh(img, thresh)
        ratio = (mask>127).sum() / (h*w) * 100
        ok    = 5 < ratio < 85
        color = "green" if ok else "red"

        axes[0,j+1].imshow(mask, cmap="gray")
        axes[0,j+1].set_title(f"seuil={thresh}\n{ratio:.1f}% objet",
                               fontsize=8, fontweight="bold", color=color)
        axes[0,j+1].axis("off")

        overlay = img_rgb.copy()
        overlay[mask>127] = (overlay[mask>127]*0.4 + np.array([0,220,80])*0.6).astype(np.uint8)
        axes[1,j+1].imshow(overlay); axes[1,j+1].axis("off")

        dist = abs(ratio - 40)
        if ok and dist < best_ratio_dist:
            best_ratio_dist = dist; best_thresh = thresh

    if best_thresh is not None:
        j = thresholds.index(best_thresh)
        axes[0,j+1].set_title(axes[0,j+1].get_title()+"\n← RECOMMANDÉ",
                               fontsize=8, fontweight="bold", color="blue")
        print(f"\n  Paramètre recommandé pour cette image :")
        print(f"  BGR_THRESHOLD = {best_thresh}")
        print(f"\n  → Modifier dans mask_generation.py :")
        cls = Path(image_path).parent.name.lower()
        print(f"    BGR_THRESHOLDS['{cls}'] = {best_thresh}")
    else:
        print("\n⚠  Aucun seuil satisfaisant (5-85%) trouvé.")
        print("   Images peut-être trop similaires au fond blanc.")
        print("   → Essayez avec use_grabcut=True dans generate_binary_mask()")

    plt.suptitle(f"Debug seuils BGR — {Path(image_path).name}\n"
                 "Vert=bon (5-85%)  Rouge=mauvais", fontsize=11)
    plt.tight_layout()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out = os.path.join(config.OUTPUT_DIR, f"debug_{Path(image_path).stem}.png")
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print(f"\n  → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python debug_mask.py chemin/vers/image.jpg")
        sys.exit(0)
    run_debug(sys.argv[1])
