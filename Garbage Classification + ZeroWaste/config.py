"""
config.py — 5 classes ZeroWaste + Garbage Classification fusionnés
"""

import os

# ─── Chemins ─────────────────────────────────────────────────────────────────
ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR   = os.path.join(ROOT_DIR, "data", "raw")
MASK_DIR       = os.path.join(ROOT_DIR, "data", "masks")
#ZEROWASTE_DIR  = os.path.join(ROOT_DIR, "data", "zerowaste")   # ← dataset ORIGINAL complet
ZEROWASTE_DIR  = os.path.join(ROOT_DIR, "data", "zerowaste_curated") # ← dataset CURATED avec des images différentes des autres
OUTPUT_DIR     = os.path.join(ROOT_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# ─── Classes ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["background", "rigid_plastic", "cardboard", "metal", "soft_plastic"]
CLASS_COLORS = [
    (30,  30,  30),
    (231, 76,  60),
    (139, 90,  43),
    (149, 165, 166),
    (52,  152, 219),
]
NUM_CLASSES = len(CLASS_NAMES)  # 5

# ─── Mapping Garbage Classification → classes ZeroWaste ──────────────────────
FOLDER_TO_CLASS = {
    "glass":       1,    # → rigid_plastic
    "cardboard":   2,    # → cardboard
    "paper":       2,    # → cardboard
    "plastic":     1,    # → rigid_plastic
    "metal":       3,    # → metal
    "trash":       None, # ignoré
    "white-glass": 1,
    "green-glass": 1,
    "brown-glass": 1,
}

# ─── Masques Garbage Classification ──────────────────────────────────────────
BGR_THRESHOLDS = {
    "glass":       230,
    "cardboard":   220,
    "paper":       200,
    "plastic":     215,
    "metal":       210,
    "white-glass": 225,
    "green-glass": 225,
    "brown-glass": 220,
}
DEFAULT_BGR_THRESH = 218

# ─── Entraînement ─────────────────────────────────────────────────────────────
IMG_SIZE        = 256
BATCH_SIZE      = 16
NUM_EPOCHS      = 30
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 0
FREEZE_ENCODER  = False
RANDOM_SEED     = 42

# ZeroWaste : 30 images par classe dominante (évite les frames consécutifs)
# Garbage   : 30 images par classe cible
N_PER_CLASS     = 30
USE_GARBAGE_DS  = True

# ─── Loss ─────────────────────────────────────────────────────────────────────
USE_FOCAL_LOSS = True
FOCAL_GAMMA    = 2.0
