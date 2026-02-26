"""
scene_composer.py
=================
Compose des scènes synthétiques à partir d'images de déchets individuels
(fond blanc) en plaçant plusieurs objets sur un canvas blanc.

Trois niveaux de densité :
  • sparse  — objets bien espacés, peu de chevauchement
  • medium  — espacement modéré, léger chevauchement possible
  • dense   — nombreux objets, chevauchements fréquents

Sortie :
  scenes/
    sparse/  images/  masks/
    medium/  images/  masks/
    dense/   images/  masks/
"""

import os
import random
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import config
from mask_generation import generate_binary_mask


# ─────────────────────────────────────────────────────────────────────────────
#  Chargement des données sources
# ─────────────────────────────────────────────────────────────────────────────

def load_source_images(raw_dir: str = config.RAW_DATA_DIR):
    """
    Retourne un dict {class_idx: [{"image": np.ndarray, "mask": np.ndarray}]}
    en chargeant les images brutes et en générant leur masque binaire à la volée.
    """
    raw_path = Path(raw_dir)
    sources = {}

    for class_folder in sorted(raw_path.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name.lower()
        class_idx  = config.FOLDER_TO_CLASS.get(class_name)
        if class_idx is None:
            continue

        entries = []
        files = sorted([
            f for f in class_folder.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ])
        print(f"  Chargement {class_name} ({len(files)} images) …", end="", flush=True)
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            mask = generate_binary_mask(img)
            entries.append({"image": img, "mask": mask, "class_idx": class_idx})
        sources[class_idx] = entries
        print(f" OK ({len(entries)} chargées)")

    return sources


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaires de placement
# ─────────────────────────────────────────────────────────────────────────────

def _resize_object(img_bgr, binary_mask, target_size: tuple):
    """Redimensionne objet + masque en conservant le ratio."""
    h, w = img_bgr.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img_r  = cv2.resize(img_bgr,    (nw, nh), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(binary_mask,(nw, nh), interpolation=cv2.INTER_NEAREST)
    return img_r, mask_r


def _paste_object(canvas_img, canvas_mask,
                  obj_img, obj_bin_mask, class_idx,
                  cx, cy):
    """
    Colle obj_img sur canvas_img aux coordonnées (cx, cy) du coin supérieur gauche.
    Utilise le masque binaire pour ne coller que les pixels de l'objet.
    Retourne True si au moins un pixel a été collé.
    """
    oh, ow = obj_img.shape[:2]
    ch, cw = canvas_img.shape[:2]

    # Clipping
    x1c, y1c = max(cx, 0), max(cy, 0)
    x2c, y2c = min(cx + ow, cw), min(cy + oh, ch)
    if x2c <= x1c or y2c <= y1c:
        return False

    x1o = x1c - cx
    y1o = y1c - cy
    x2o = x1o + (x2c - x1c)
    y2o = y1o + (y2c - y1c)

    obj_crop  = obj_img[y1o:y2o, x1o:x2o]
    mask_crop = obj_bin_mask[y1o:y2o, x1o:x2o]

    fg = mask_crop > 127
    canvas_img[y1c:y2c, x1c:x2c][fg]  = obj_crop[fg]
    canvas_mask[y1c:y2c, x1c:x2c][fg] = class_idx
    return True


def _sample_position(cx_range, cy_range, existing_boxes, ow, oh, overlap_ratio=0.3):
    """
    Tire une position aléatoire en limitant le chevauchement.
    overlap_ratio : part maximale autorisée de recouvrement par rapport à un objet existant.
    """
    for _ in range(50):
        cx = random.randint(*cx_range)
        cy = random.randint(*cy_range)
        box = (cx, cy, cx + ow, cy + oh)

        valid = True
        for bx in existing_boxes:
            ix1 = max(box[0], bx[0]); iy1 = max(box[1], bx[1])
            ix2 = min(box[2], bx[2]); iy2 = min(box[3], bx[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area  = ow * oh
            if area > 0 and inter / area > overlap_ratio:
                valid = False
                break

        if valid:
            return cx, cy, box
    # fallback sans contrainte
    cx = random.randint(*cx_range)
    cy = random.randint(*cy_range)
    return cx, cy, (cx, cy, cx + ow, cy + oh)


# ─────────────────────────────────────────────────────────────────────────────
#  Composition d'une scène
# ─────────────────────────────────────────────────────────────────────────────

def compose_scene(sources: dict,
                  n_objects: int,
                  scene_w: int = config.SCENE_WIDTH,
                  scene_h: int = config.SCENE_HEIGHT,
                  overlap_ratio: float = 0.3) -> tuple:
    """
    Compose une scène aléatoire.

    Retourne (scene_bgr, semantic_mask, metadata)
    """
    canvas_img  = np.ones((scene_h, scene_w, 3), dtype=np.uint8) * 255
    canvas_mask = np.zeros((scene_h, scene_w), dtype=np.uint8)
    metadata    = []

    available_classes = [c for c, items in sources.items() if len(items) > 0]
    if not available_classes:
        raise ValueError("Aucune image source disponible.")

    existing_boxes = []

    for _ in range(n_objects):
        class_idx = random.choice(available_classes)
        item      = random.choice(sources[class_idx])

        # Taille cible
        scale  = random.uniform(*config.OBJECT_SCALE_RANGE)
        target = (int(scene_w * scale), int(scene_h * scale))

        obj_img, obj_mask = _resize_object(item["image"], item["mask"], target)
        oh, ow = obj_img.shape[:2]

        # Position
        cx_range = (0, max(0, scene_w - ow))
        cy_range = (0, max(0, scene_h - oh))
        cx, cy, box = _sample_position(cx_range, cy_range, existing_boxes, ow, oh, overlap_ratio)
        existing_boxes.append(box)

        ok = _paste_object(canvas_img, canvas_mask, obj_img, obj_mask, class_idx, cx, cy)
        if ok:
            metadata.append({"class": config.CLASS_NAMES[class_idx], "box": box})

    return canvas_img, canvas_mask, metadata


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline de génération complet
# ─────────────────────────────────────────────────────────────────────────────

def generate_scenes(raw_dir: str = config.RAW_DATA_DIR,
                    scene_dir: str = config.SCENE_DIR,
                    n_per_level: int = config.N_SCENES_PER_LEVEL,
                    seed: int = config.RANDOM_SEED) -> None:
    """
    Génère n_per_level scènes pour chaque niveau de densité défini dans
    config.DENSITY_LEVELS.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("═══ Chargement des images sources ═══")
    sources = load_source_images(raw_dir)

    # Paramètres de chevauchement par densité
    overlap_map = {"sparse": 0.05, "medium": 0.30, "dense": 0.60}

    for level, n_objects in config.DENSITY_LEVELS.items():
        print(f"\n═══ Génération scènes '{level}' ({n_objects} objets/scène) ═══")
        img_dir  = Path(scene_dir) / level / "images"
        msk_dir  = Path(scene_dir) / level / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        all_meta = []
        for i in tqdm(range(n_per_level), desc=level):
            scene_img, scene_mask, meta = compose_scene(
                sources, n_objects,
                overlap_ratio=overlap_map[level]
            )
            fname = f"{level}_{i:04d}"
            cv2.imwrite(str(img_dir / f"{fname}.png"), scene_img)
            cv2.imwrite(str(msk_dir / f"{fname}.png"), scene_mask)
            all_meta.append({"file": fname, "objects": meta})

        # Sauvegarde des métadonnées
        meta_path = Path(scene_dir) / level / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(all_meta, f, indent=2)

        print(f"  → {n_per_level} scènes sauvegardées dans {img_dir.parent}")


def visualize_scenes(scene_dir: str = config.SCENE_DIR, n: int = 2) -> None:
    """Sauvegarde n exemples de scènes avec overlay de couleurs sémantiques."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    color_arr = np.array(config.CLASS_COLORS, dtype=np.uint8)  # (7,3) RGB

    for level in config.DENSITY_LEVELS:
        img_dir = Path(scene_dir) / level / "images"
        msk_dir = Path(scene_dir) / level / "masks"
        files   = sorted(img_dir.glob("*.png"))[:n]

        for f in files:
            img  = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(msk_dir / f.name), cv2.IMREAD_GRAYSCALE)

            # Overlay coloré
            overlay = img.copy()
            for cls_idx, color in enumerate(config.CLASS_COLORS):
                if cls_idx == 0: continue
                overlay[mask == cls_idx] = color

            blended = (0.55 * img + 0.45 * overlay).astype(np.uint8)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img);                            axes[0].set_title("Scène")
            axes[1].imshow(mask, cmap="tab10", vmin=0, vmax=6); axes[1].set_title("Masque sémantique")
            axes[2].imshow(blended);                        axes[2].set_title("Overlay")
            for ax in axes: ax.axis("off")

            patches = [mpatches.Patch(color=np.array(c)/255, label=n)
                       for c, n in zip(config.CLASS_COLORS[1:], config.CLASS_NAMES[1:])]
            fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=8)
            plt.suptitle(f"Densité : {level} — {f.stem}", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(config.OUTPUT_DIR, f"scene_{level}_{f.stem}.png"),
                        dpi=100, bbox_inches="tight")
            plt.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_scenes()
    visualize_scenes(n=2)
