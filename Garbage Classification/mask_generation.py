"""
mask_generation.py — approche directe BGR
==========================================
Abandon du flood fill (trop sensible à la position de l'objet dans l'image).

Approche retenue : seuillage direct sur les valeurs BGR
  Un pixel est FOND si R > seuil ET G > seuil ET B > seuil.
  Un fond blanc pur aura R=G=B=255. On tolère jusqu'à ~20 pts de variation
  due à la compression JPEG ou à l'ombre légère des bords.

  Cette méthode est robuste pour :
    - Papier blanc  (objet blanc sur fond blanc) → on baisse le seuil
    - Verre transparent → fallback Canny
    - Métal brillant → seuil intermédiaire
    - Plastique coloré → fonctionne parfaitement

Pipeline par image :
  1. Seuillage direct BGR  → masque fond grossier
  2. Morphologie (close)   → bouche les trous à l'intérieur de l'objet
  3. Remplissage de trous  → si des pixels fond sont piégés à l'intérieur
  4. Suppression bordures  → retire le fond résiduel sur les 2px de bord
  5. Fallback Canny        → si masque toujours vide (verre transparent)
"""

import os, cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config

# ─── Seuils BGR par classe ────────────────────────────────────────────────────
# Seuil = valeur minimale de chaque canal pour qu'un pixel soit "fond blanc"
# Plus bas = moins de fond détecté (utile pour objets clairs comme le papier)
# Plus haut = plus de fond détecté (utile pour objets sombres/colorés)
BGR_THRESHOLDS = {
    "glass":       230,   # verre : souvent transparent, seuil bas + Canny en fallback
    "cardboard":   220,   # carton brun : bien distinct du fond blanc
    "paper":       200,   # ← CLEF : papier blanc, on doit être très strict
                          #   sinon on efface l'objet. Descendre encore si vide.
    "plastic":     215,   # coloré : facile à détecter
    "metal":       210,   # brillant mais avec reflets gris
    "trash":       215,   # divers
    "white-glass": 225,
    "green-glass": 225,
    "brown-glass": 220,
}
DEFAULT_THRESHOLD = 218


def generate_binary_mask(img_bgr: np.ndarray,
                          class_name: str = None,
                          use_grabcut: bool = False) -> np.ndarray:
    """
    Retourne masque binaire uint8 : 255=objet, 0=fond.
    """
    thresh = BGR_THRESHOLDS.get(
        class_name.lower() if class_name else "", DEFAULT_THRESHOLD
    )
    h, w = img_bgr.shape[:2]

    # ── Étape 1 : Léger flou pour atténuer le bruit JPEG ─────────────────────
    smooth = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    b, g, r = cv2.split(smooth)

    # ── Étape 2 : Masque fond = tous les canaux > seuil ──────────────────────
    bg_mask  = ((b.astype(np.int16) > thresh) &
                (g.astype(np.int16) > thresh) &
                (r.astype(np.int16) > thresh)).astype(np.uint8) * 255
    obj_mask = cv2.bitwise_not(bg_mask)

    # ── Étape 3 : Fermeture morphologique → soude les trous intérieurs ───────
    # (pixels clairs à l'intérieur de l'objet qui ont été classés "fond")
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, k_close, iterations=3)

    # ── Étape 4 : Ouverture → supprime petits artefacts isolés ───────────────
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # ── Étape 5 : Remplissage des trous résiduels ─────────────────────────────
    obj_mask = _fill_holes(obj_mask)

    # ── Étape 6 : Forcer les 3 pixels de bord à 0 (toujours fond) ────────────
    border = 3
    obj_mask[:border, :]  = 0
    obj_mask[-border:, :] = 0
    obj_mask[:, :border]  = 0
    obj_mask[:, -border:] = 0

    # ── Étape 7 : Garder uniquement le plus grand composant ───────────────────
    obj_mask = _keep_largest(obj_mask)

    # ── Étape 8 : GrabCut optionnel ───────────────────────────────────────────
    if use_grabcut and (obj_mask > 127).sum() > 200:
        obj_mask = _grabcut_refine(img_bgr, obj_mask)

    # ── Étape 9 : Fallback Canny si masque toujours vide ─────────────────────
    ratio = (obj_mask > 127).sum() / (h * w)
    if ratio < 0.01:
        obj_mask = _canny_fallback(img_bgr, obj_mask)

    return obj_mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Remplit les trous INTÉRIEURS (pixels fond encerclés par l'objet).
    Méthode : flood fill depuis le coin supérieur gauche sur masque INVERSÉ.
    Seuls les pixels accessibles depuis le bord sont vrais fond.
    """
    h, w    = mask.shape
    inv     = cv2.bitwise_not(mask)          # 255=fond, 0=objet
    canvas  = np.zeros((h+2, w+2), np.uint8)

    # Flood fill depuis le coin : marque le fond accessible depuis le bord
    # On travaille sur une copie car floodFill modifie l'image
    inv_copy = inv.copy()
    cv2.floodFill(inv_copy, canvas, (0, 0), 128)   # fond accessible → 128

    # Pixels de fond NON atteints = trous intérieurs → les mettre à 0 (objet)
    real_bg = (inv_copy == 128).astype(np.uint8) * 255
    return cv2.bitwise_not(real_bg)   # 255=objet (y compris les trous comblés)


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out


def _grabcut_refine(img_bgr, init_obj, n_iter=3):
    gc = np.where(init_obj > 127, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc[0,:]=gc[-1,:]=gc[:,0]=gc[:,-1] = cv2.GC_BGD
    bgd = np.zeros((1,65), np.float64)
    fgd = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(img_bgr, gc, None, bgd, fgd, n_iter, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return init_obj
    return np.where((gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def _canny_fallback(img_bgr, current_mask):
    """Dernier recours : contours Canny pour objets transparents (verre)."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), 20, 80)
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    edges = cv2.dilate(edges, k, iterations=4)
    filled = _fill_holes(edges)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k, iterations=3)
    filled = _keep_largest(filled)
    ratio  = (filled>127).sum() / max(gray.size, 1)
    return filled if 0.01 < ratio < 0.95 else current_mask


def binary_to_semantic(binary_mask, class_idx):
    sem = np.zeros_like(binary_mask, dtype=np.uint8)
    sem[binary_mask > 127] = class_idx
    return sem


# ─── Pipeline dataset complet ─────────────────────────────────────────────────

def generate_masks_for_dataset(raw_dir=config.RAW_DATA_DIR,
                                mask_dir=config.MASK_DIR,
                                use_grabcut=False,
                                visualize_n=3):
    raw_path  = Path(raw_dir)
    mask_path = Path(mask_dir)
    mask_path.mkdir(parents=True, exist_ok=True)

    all_ratios = {}
    for class_folder in sorted(raw_path.iterdir()):
        if not class_folder.is_dir(): continue
        class_name = class_folder.name.lower()
        class_idx  = config.FOLDER_TO_CLASS.get(class_name)
        if class_idx is None:
            print(f"[WARN] '{class_folder.name}' non reconnu → ignoré")
            continue

        out_folder = mask_path / class_folder.name
        out_folder.mkdir(parents=True, exist_ok=True)
        files = sorted([f for f in class_folder.iterdir()
                        if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}])

        print(f"\n── {class_name} (classe {class_idx}, seuil BGR={BGR_THRESHOLDS.get(class_name, DEFAULT_THRESHOLD)}) — {len(files)} images")
        failed=0; ratios=[]; vis_count=0

        for img_file in tqdm(files, desc=class_name):
            img = cv2.imread(str(img_file))
            if img is None:
                failed += 1; continue

            binary   = generate_binary_mask(img, class_name, use_grabcut)
            semantic = binary_to_semantic(binary, class_idx)
            cv2.imwrite(str(out_folder / (img_file.stem + ".png")), semantic)

            ratio = (binary > 127).sum() / (img.shape[0] * img.shape[1])
            ratios.append(ratio)

            if vis_count < visualize_n:
                _save_vis(img, binary, semantic, class_name, img_file.stem)
                vis_count += 1

        mean_r = np.mean(ratios) if ratios else 0
        ok     = len(files) - failed
        quality = "✓" if 0.03 < mean_r < 0.90 else "⚠ VÉRIFIER"
        print(f"   {ok}/{len(files)} masques | ratio moyen {mean_r:.1%} {quality}")
        all_ratios[class_name] = mean_r

    print("\n═══ Résumé qualité masques ═══")
    for cls, r in all_ratios.items():
        bar  = "█" * int(r * 40)
        flag = "✓" if 0.03 < r < 0.90 else "⚠"
        print(f"  {cls:<12}  {r:>6.1%}  {flag}  {bar}")
    
    # Conseils automatiques
    print("\n═══ Conseils ajustement seuils ═══")
    for cls, r in all_ratios.items():
        curr = BGR_THRESHOLDS.get(cls, DEFAULT_THRESHOLD)
        if r > 0.90:
            print(f"  {cls:<12} ratio {r:.0%} trop élevé  → baisser BGR_THRESHOLDS['{cls}'] "
                  f"de {curr} vers {curr - 15} ou {curr - 25}")
        elif r < 0.03:
            print(f"  {cls:<12} ratio {r:.0%} trop faible → monter BGR_THRESHOLDS['{cls}'] "
                  f"de {curr} vers {curr + 15} ou {curr + 25}")


def _save_vis(img_bgr, binary, semantic, class_name, stem):
    import matplotlib.pyplot as plt
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    overlay = img_bgr.copy()
    color   = config.CLASS_COLORS[config.FOLDER_TO_CLASS.get(class_name, 0)]
    overlay[semantic > 0] = (overlay[semantic>0]*0.3 + np.array(color)*0.7).astype(np.uint8)
    contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_bgr.copy()
    cv2.drawContours(contour_img, contours, -1, (0,0,255), 2)
    ratio = (binary>127).sum() / (img_bgr.shape[0]*img_bgr.shape[1])*100

    fig, axes = plt.subplots(1, 4, figsize=(18,4))
    axes[0].imshow(cv2.cvtColor(img_bgr,     cv2.COLOR_BGR2RGB)); axes[0].set_title("Original")
    axes[1].imshow(binary, cmap="gray");                           axes[1].set_title(f"Masque ({ratio:.1f}%)")
    axes[2].imshow(cv2.cvtColor(overlay,     cv2.COLOR_BGR2RGB)); axes[2].set_title("Overlay")
    axes[3].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)); axes[3].set_title("Contours")
    for ax in axes: ax.axis("off")
    plt.suptitle(f"{class_name} — {stem}", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, f"mask_vis_{class_name}_{stem}.png"),
                dpi=100, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        from debug_mask import run_debug
        run_debug(sys.argv[1])
    else:
        generate_masks_for_dataset(visualize_n=3)
