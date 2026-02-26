"""
main.py
=======
Point d'entrée principal du projet.
Lance séquentiellement :
  1. Vérification / téléchargement des données
  2. Génération des masques individuels
  3. Composition des scènes synthétiques
  4. Entraînement du modèle
  5. Évaluation et analyse de densité

Usage :
  python main.py                      # pipeline complet
  python main.py --step masks         # génération de masques uniquement
  python main.py --step scenes        # composition de scènes uniquement
  python main.py --step train         # entraînement uniquement
  python main.py --step eval          # évaluation uniquement
  python main.py --step download      # aide au téléchargement
"""

import os
import argparse
import sys

import config


# ─────────────────────────────────────────────────────────────────────────────
#  Aide au téléchargement
# ─────────────────────────────────────────────────────────────────────────────

def print_download_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║              INSTRUCTIONS DE TÉLÉCHARGEMENT DES DATASETS                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Dataset TrashNet (recommandé, 2 527 images)                             ║
║  ─────────────────────────────────────────────                           ║
║  1. Télécharger depuis GitHub :                                          ║
║     https://github.com/garythung/trashnet                                ║
║     → data/data.zip (ou trashnet-data-v1.zip)                            ║
║  2. Extraire dans : data/raw/                                            ║
║     Structure attendue :                                                 ║
║       data/raw/glass/      *.jpg                                         ║
║       data/raw/cardboard/  *.jpg                                         ║
║       data/raw/paper/      *.jpg                                         ║
║       data/raw/plastic/    *.jpg                                         ║
║       data/raw/metal/      *.jpg                                         ║
║       data/raw/trash/      *.jpg                                         ║
║                                                                          ║
║  Dataset Kaggle Garbage Classification (2 467 images)                    ║
║  ────────────────────────────────────────────────────                    ║
║  Option A — CLI Kaggle :                                                 ║
║    pip install kaggle                                                    ║
║    # Placez kaggle.json dans ~/.kaggle/                                  ║
║    kaggle datasets download -d asdasdasasdas/garbage-classification      ║
║    unzip garbage-classification.zip -d data/raw_kaggle/                  ║
║                                                                          ║
║  Option B — Interface web :                                              ║
║    https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification  ║
║                                                                          ║
║  Remarque : si les deux datasets sont utilisés, fusionnez les dossiers   ║
║  homonymes (glass/, plastic/, etc.) dans data/raw/.                      ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


def download_with_kaggle_api():
    """Téléchargement automatique via l'API Kaggle."""
    try:
        import kaggle  # noqa
    except ImportError:
        print("[ERREUR] Le package 'kaggle' n'est pas installé.")
        print("  → pip install kaggle")
        return False

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("[ERREUR] Fichier kaggle.json introuvable dans ~/.kaggle/")
        print("  → Téléchargez-le depuis https://www.kaggle.com/account")
        return False

    os.makedirs(os.path.join(config.ROOT_DIR, "data"), exist_ok=True)
    os.system(
        f"kaggle datasets download -d asdasdasasdas/garbage-classification "
        f"--path {os.path.join(config.ROOT_DIR, 'data')} --unzip"
    )
    print("Téléchargement terminé.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Vérifications préalables
# ─────────────────────────────────────────────────────────────────────────────

def check_data_exists():
    raw = config.RAW_DATA_DIR
    if not os.path.isdir(raw):
        print(f"[ERREUR] Répertoire de données introuvable : {raw}")
        print_download_instructions()
        sys.exit(1)

    valid_classes = set(config.FOLDER_TO_CLASS.keys())
    found = [d for d in os.listdir(raw)
             if os.path.isdir(os.path.join(raw, d)) and d.lower() in valid_classes]
    if len(found) == 0:
        print(f"[ERREUR] Aucun dossier de classe reconnu dans {raw}")
        print_download_instructions()
        sys.exit(1)

    total_imgs = sum(
        len([f for f in os.listdir(os.path.join(raw, d))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for d in found
    )
    print(f"✓ {len(found)} classe(s) trouvée(s) | {total_imgs} images au total")


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline complet
# ─────────────────────────────────────────────────────────────────────────────

def run_masks():
    from mask_generation import generate_masks_for_dataset
    print("\n═══ ÉTAPE 1 : Génération des masques ═══")
    generate_masks_for_dataset(visualize_n=3)


def run_scenes():
    from scene_composer import generate_scenes, visualize_scenes
    print("\n═══ ÉTAPE 2 : Composition des scènes ═══")
    generate_scenes()
    visualize_scenes(n=2)


def run_train():
    from train import train
    print("\n═══ ÉTAPE 3 : Entraînement ═══")
    train()


def run_eval():
    from evaluate import full_evaluation
    print("\n═══ ÉTAPE 4 : Évaluation ═══")
    full_evaluation()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de segmentation sémantique de déchets"
    )
    parser.add_argument(
        "--step",
        choices=["download", "masks", "scenes", "train", "eval", "all"],
        default="all",
        help="Étape à exécuter (défaut : all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for d in [config.RAW_DATA_DIR, config.MASK_DIR, config.SCENE_DIR,
              config.OUTPUT_DIR, config.CHECKPOINT_DIR]:
        os.makedirs(d, exist_ok=True)

    if args.step == "download":
        print_download_instructions()
        download_with_kaggle_api()
        return

    if args.step in ("all", "masks", "scenes", "train", "eval"):
        check_data_exists()

    if args.step == "masks" or args.step == "all":
        run_masks()

    if args.step == "scenes" or args.step == "all":
        run_scenes()

    if args.step == "train" or args.step == "all":
        run_train()

    if args.step == "eval" or args.step == "all":
        run_eval()

    print("\n✓ Pipeline terminé.")


if __name__ == "__main__":
    main()
