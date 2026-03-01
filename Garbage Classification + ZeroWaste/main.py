"""
main.py — Point d'entrée
========================
Usage :
  python main.py --step masks    # génère masques Garbage Classification
  python main.py --step train    # entraîne sur ZeroWaste + GarbageClassif
  python main.py --step eval     # évalue le meilleur checkpoint
  python main.py --step all      # masks → train → eval
"""

import argparse, os, sys
import config


def print_setup():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                 STRUCTURE DES DONNÉES ATTENDUE                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  data/                                                           ║
║    zerowaste/                 ← ZeroWaste-f dataset              ║
║      train/  data/  sem_seg/                                     ║
║      val/    data/  sem_seg/                                     ║
║      test/   data/  sem_seg/                                     ║
║    raw/                       ← Garbage Classification           ║
║      glass/      *.jpg                                           ║
║      cardboard/  *.jpg                                           ║
║      paper/      *.jpg                                           ║
║      plastic/    *.jpg                                           ║
║      metal/      *.jpg                                           ║
║      trash/      *.jpg  (ignoré, pas de classe équivalente)      ║
║                                                                  ║
║  Classes unifiées (5) :                                          ║
║    0 background   1 rigid_plastic   2 cardboard                  ║
║    3 metal        4 soft_plastic                                 ║
║                                                                  ║
║  Mapping Garbage → ZeroWaste :                                   ║
║    glass, plastic → rigid_plastic                                ║
║    cardboard, paper → cardboard                                  ║
║    metal → metal                                                 ║
║    trash → ignoré                                                ║
╚══════════════════════════════════════════════════════════════════╝
""")


def check_data():
    zw  = os.path.isdir(os.path.join(config.ZEROWASTE_DIR,"train","data"))
    raw = os.path.isdir(config.RAW_DATA_DIR)
    if not zw and not raw:
        print("[ERREUR] Aucune source de données trouvée.")
        print_setup(); sys.exit(1)
    if not zw:
        print(f"[INFO] ZeroWaste non trouvé dans {config.ZEROWASTE_DIR}")
    if not raw:
        print(f"[INFO] Garbage Classification non trouvé dans {config.RAW_DATA_DIR}")


def run_masks():
    from mask_generation import generate_masks_for_dataset
    print("\n═══ Génération des masques (Garbage Classification) ═══")
    generate_masks_for_dataset()


def run_train():
    from train import train
    print("\n═══ Entraînement ═══")
    train()


def run_eval():
    from evaluate import full_evaluation
    print("\n═══ Évaluation ═══")
    full_evaluation()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--step",
                   choices=["masks","train","eval","all","setup"],
                   default="all")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for d in [config.RAW_DATA_DIR, config.MASK_DIR, config.OUTPUT_DIR,
              config.CHECKPOINT_DIR]:
        os.makedirs(d, exist_ok=True)

    if args.step == "setup":
        print_setup(); sys.exit(0)

    check_data()

    if args.step in ("masks","all"):
        if os.path.isdir(config.RAW_DATA_DIR):
            run_masks()
        else:
            print("[SKIP] Pas de Garbage Classification → masques non générés")

    if args.step in ("train","all"):
        run_train()

    if args.step in ("eval","all"):
        run_eval()

    print("\n✓ Terminé.")
