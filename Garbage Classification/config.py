"""
config.py — Paramètres globaux du projet
CPU-optimized : images 128×128, encodeur gelé, 20 epochs max
"""

import os

# ─── Chemins ────────────────────────────────────────────────────────────────
ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR    = os.path.join(ROOT_DIR, "data", "raw")
MASK_DIR        = os.path.join(ROOT_DIR, "data", "masks")
SCENE_DIR       = os.path.join(ROOT_DIR, "data", "scenes")
OUTPUT_DIR      = os.path.join(ROOT_DIR, "outputs")
CHECKPOINT_DIR  = os.path.join(ROOT_DIR, "checkpoints")

# ─── Classes ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["background", "glass", "cardboard", "paper", "plastic", "metal", "trash"]
CLASS_COLORS = [
    (0,   0,   0),
    (0,   255, 255),
    (139, 69,  19),
    (200, 200, 200),
    (255, 0,   0),
    (128, 128, 128),
    (0,   128, 0),
]
NUM_CLASSES = len(CLASS_NAMES)

FOLDER_TO_CLASS = {
    "glass": 1, "cardboard": 2, "paper": 3,
    "plastic": 4, "metal": 5, "trash": 6,
    "white-glass": 1, "green-glass": 1, "brown-glass": 1,
}

# ─── Génération de masques ────────────────────────────────────────────────────
WHITE_THRESH  = 240
MASK_BLUR_K   = 5
MORPH_ITER    = 2

# ─── Scènes composites ────────────────────────────────────────────────────────
SCENE_WIDTH   = 256          # ↓ 512 → 256
SCENE_HEIGHT  = 256
SCENE_BG      = (255, 255, 255)
DENSITY_LEVELS = {
    "sparse":  3,
    "medium":  6,
    "dense":  12,
}
OBJECT_SCALE_RANGE  = (0.15, 0.30)
N_SCENES_PER_LEVEL  = 80     # ↓ 200 → 80  (240 scènes total, rapide à générer)

# ─── Entraînement ─────────────────────────────────────────────────────────────
IMG_SIZE      = 128          # ↓ 512 → 128  : 16× moins de pixels !
BATCH_SIZE    = 32           # ↑ batch élevé = moins de steps par epoch
NUM_EPOCHS    = 20           # ↓ 30 → 20
LR            = 5e-4         # LR plus élevé, converge plus vite
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 0            # 0 = pas de multiprocessing (plus stable et souvent plus rapide sur CPU)
DEVICE        = "cpu"
USE_AMP       = False        # AMP uniquement utile sur GPU NVIDIA
FREEZE_ENCODER = True        # ← Clé : geler ResNet-18, n'entraîner que le décodeur
                             #    Réduit le nb de paramètres entraînés de 11M → ~2M
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
RANDOM_SEED   = 42

# ─── Perte Focal (anti-effondrement vers classe majoritaire) ──────────────────
USE_FOCAL_LOSS = True    # remplace CrossEntropy classique
FOCAL_GAMMA    = 2.0     # plus élevé = pénalise davantage les classes faciles
FOCAL_ALPHA    = True    # pondération inverse de fréquence (comme class_weights)

# ─── Stratégie d'entraînement corrigée ────────────────────────────────────────
FREEZE_ENCODER = False   # dégeler l'encodeur : nécessaire pour distinguer verre/métal/papier
LR             = 1e-4    # LR plus faible car encodeur dégelé
BATCH_SIZE     = 16
NUM_EPOCHS     = 25
