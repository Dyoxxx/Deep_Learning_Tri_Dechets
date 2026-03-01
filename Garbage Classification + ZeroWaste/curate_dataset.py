"""
curate_dataset.py
=================
Sélection d'images ZeroWaste diversifiées (évite les frames consécutifs).

IMPORTANT : toujours pointer vers le dataset ORIGINAL, pas le dossier curé.
  python curate_dataset.py --auto --n 200 --zw-dir data/zerowaste/ --out data/zerowaste_curated/

Usage :
  python curate_dataset.py --analyze  --zw-dir data/zerowaste/
  python curate_dataset.py --auto --n 200 --zw-dir data/zerowaste/ --out data/zerowaste_curated/
  python curate_dataset.py --grid --n 200 --zw-dir data/zerowaste/
"""

import os, sys, shutil, argparse, cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config


# ─── Hash perceptuel ──────────────────────────────────────────────────────────

def phash(img_bgr, size=16):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).flatten()

def hamming(h1, h2):
    return int((h1 != h2).sum())


# ─── Analyse ─────────────────────────────────────────────────────────────────

def analyze_dataset(split_dir, sample_n=200):
    img_dir = Path(split_dir) / "data"
    files   = sorted([f for f in img_dir.iterdir()
                       if f.suffix.lower() in {".jpg",".jpeg",".png"}])[:sample_n]
    print(f"Analyse de {len(files)} images consécutives …")

    hashes    = [(f.name, phash(cv2.imread(str(f)))) for f in files
                 if cv2.imread(str(f)) is not None]
    distances = [hamming(hashes[i][1], hashes[i+1][1])
                 for i in range(len(hashes)-1)]

    print(f"  Distance Hamming consécutive — moy:{np.mean(distances):.1f}  "
          f"min:{np.min(distances)}  max:{np.max(distances)}")
    print(f"  % quasi-identiques (dist<20) : "
          f"{100*sum(d<20 for d in distances)/len(distances):.1f}%")
    print(f"  % similaires       (dist<40) : "
          f"{100*sum(d<40 for d in distances)/len(distances):.1f}%")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(distances, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(20, color="red",    linestyle="--", label="Quasi-identique (20)")
    ax.axvline(40, color="orange", linestyle="--", label="Similaire (40)")
    ax.set_xlabel("Distance Hamming"); ax.set_ylabel("Nb paires")
    ax.set_title("Similarité entre frames consécutifs")
    ax.legend()
    plt.tight_layout()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out = os.path.join(config.OUTPUT_DIR, "dataset_similarity.png")
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print(f"  → {out}")


# ─── Sélection automatique ────────────────────────────────────────────────────

def auto_select(zerowaste_dir, n=200, min_hamming=30, out_dir=None):
    zw_path  = Path(zerowaste_dir)
    all_imgs = []
    for split in ["train", "val", "test"]:
        img_dir  = zw_path / split / "data"
        mask_dir = zw_path / split / "sem_seg"
        if not img_dir.exists(): continue
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in {".jpg",".jpeg",".png"}:
                mp = mask_dir / (f.stem + ".png")
                all_imgs.append((f, mp if mp.exists() else None, split))

    print(f"Dataset complet : {len(all_imgs)} images")
    print("Calcul des hashes perceptuels …")

    entries = []
    for img_path, mask_path, split in all_imgs:
        img = cv2.imread(str(img_path))
        if img is None: continue
        entries.append({"path": img_path, "mask": mask_path,
                        "split": split, "hash": phash(img)})
    print(f"  {len(entries)} images chargées")

    # Greedy diversité
    selected  = [entries[0]]
    remaining = entries[1:]

    while len(selected) < n and remaining:
        best_entry = None; best_score = -1
        for cand in remaining:
            min_dist = min(hamming(cand["hash"], s["hash"]) for s in selected)
            if min_dist > best_score:
                best_score = min_dist; best_entry = cand
        if best_score < min_hamming:
            print(f"  Stop : plus aucune image assez différente "
                  f"(dist max = {best_score} < {min_hamming})")
            break
        selected.append(best_entry)
        remaining = [e for e in remaining if e != best_entry]
        if len(selected) % 10 == 0:
            print(f"  Sélectionnées : {len(selected)}/{n}  (dernière dist={best_score})")

    from collections import Counter
    print(f"\nSélection terminée : {len(selected)} images")
    print(f"  Provenance : {dict(Counter(e['split'] for e in selected))}")

    if out_dir:
        _save_selection(selected, out_dir)
    _show_grid(selected[:20], "20 premières images sélectionnées")
    return selected


# ─── Sauvegarde ──────────────────────────────────────────────────────────────

def _save_selection(selected, out_dir):
    import random, time
    out_path = Path(out_dir)

    # Supprimer l'ancien dossier proprement avant de recopier
    if out_path.exists():
        print(f"  Suppression de l'ancien dossier {out_dir} …")
        # Sur Windows, attendre que les fichiers soient libérés
        for attempt in range(5):
            try:
                shutil.rmtree(out_path)
                break
            except PermissionError:
                time.sleep(0.5)
        else:
            # Si toujours verrouillé, écrire dans un sous-dossier horodaté
            import datetime
            ts      = datetime.datetime.now().strftime("%H%M%S")
            out_dir = str(out_path.parent / f"{out_path.name}_{ts}")
            out_path = Path(out_dir)
            print(f"  [WARN] Dossier verrouillé → sortie dans {out_dir}")

    n       = len(selected)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    splits  = (["train"] * n_train +
               ["val"]   * n_val   +
               ["test"]  * (n - n_train - n_val))
    random.shuffle(selected)

    counts = {"train": 0, "val": 0, "test": 0}
    for entry, split in zip(selected, splits):
        for sub, src, suffix in [
            ("data",    entry["path"], entry["path"].suffix),
            ("sem_seg", entry["mask"], ".png"),
        ]:
            if src is None: continue
            dst_dir = out_path / split / sub
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / (entry["path"].stem + suffix)
            for attempt in range(3):
                try:
                    shutil.copy2(src, dst); break
                except PermissionError:
                    import time; time.sleep(0.2)
            else:
                print(f"  [WARN] Copie impossible : {src.name}")
        counts[split] += 1

    print(f"\nDataset curé → {out_dir}")
    print(f"  train : {counts['train']}  val : {counts['val']}  test : {counts['test']}")
    print(f"\n  → Mettez à jour config.py :")
    print(f'    ZEROWASTE_DIR = "{out_dir}"')


def _show_grid(entries, title="", cols=5):
    n    = len(entries)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten()
    for i, e in enumerate(entries):
        img = cv2.cvtColor(cv2.imread(str(e["path"])), cv2.COLOR_BGR2RGB)
        axes[i].imshow(cv2.resize(img, (200, 150)))
        axes[i].set_title(f"#{i+1} {e['split']}", fontsize=7)
        axes[i].axis("off")
    for i in range(len(entries), len(axes)):
        axes[i].axis("off")
    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out = os.path.join(config.OUTPUT_DIR, "curated_selection.png")
    plt.savefig(out, dpi=100, bbox_inches="tight"); plt.close()
    print(f"  Grille → {out}")


# ─── Mode grille manuelle ─────────────────────────────────────────────────────

def grid_select(zerowaste_dir, n=200, out_dir=None):
    zw_path   = Path(zerowaste_dir)
    all_files = []
    for split in ["train", "val", "test"]:
        img_dir  = zw_path / split / "data"
        mask_dir = zw_path / split / "sem_seg"
        if not img_dir.exists(): continue
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in {".jpg",".jpeg",".png"}:
                mp = mask_dir / (f.stem + ".png")
                all_files.append((f, mp if mp.exists() else None, split))

    step       = max(1, len(all_files) // (n * 2))
    candidates = all_files[::step]
    batch_size = 25
    n_batches  = (len(candidates) + batch_size - 1) // batch_size
    print(f"  {len(candidates)} candidates (1/{step}) → {n_batches} grilles")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    for b in range(n_batches):
        batch = candidates[b*batch_size:(b+1)*batch_size]
        cols  = 5; rows = (len(batch)+cols-1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = axes.flatten()
        for i, (img_path, _, split) in enumerate(batch):
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            axes[i].imshow(cv2.resize(img, (200,150)))
            axes[i].set_title(f"#{b*batch_size+i:03d} {split}", fontsize=7)
            axes[i].axis("off")
        for i in range(len(batch), len(axes)):
            axes[i].axis("off")
        plt.suptitle(f"Lot {b+1}/{n_batches}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(config.OUTPUT_DIR, f"grid_batch_{b+1:02d}.png")
        plt.savefig(out, dpi=100, bbox_inches="tight"); plt.close()
        print(f"  Grille {b+1}/{n_batches} → {out}")

    print(f"\n→ Notez les numéros (#XXX) à garder dans les grilles")
    print(f"→ Puis : python curate_dataset.py --indices 003,007,... "
          f"--zw-dir {zerowaste_dir} --out {out_dir or 'data/zerowaste_curated/'}")


def select_by_indices(zerowaste_dir, indices_str, out_dir):
    zw_path   = Path(zerowaste_dir)
    all_files = []
    for split in ["train", "val", "test"]:
        img_dir  = zw_path / split / "data"
        mask_dir = zw_path / split / "sem_seg"
        if not img_dir.exists(): continue
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in {".jpg",".jpeg",".png"}:
                mp = mask_dir / (f.stem + ".png")
                all_files.append((f, mp if mp.exists() else None, split))

    step       = max(1, len(all_files) // 400)
    candidates = all_files[::step]
    indices    = [int(i.strip()) for i in indices_str.split(",") if i.strip()]
    selected   = [{"path": candidates[i][0], "mask": candidates[i][1],
                   "split": candidates[i][2], "hash": None}
                  for i in indices if 0 <= i < len(candidates)]
    print(f"{len(selected)} images sélectionnées")
    _save_selection(selected, out_dir)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Curation dataset ZeroWaste",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT : toujours utiliser --zw-dir pour pointer vers le dataset ORIGINAL
  (et non le dossier curé, qui n'a que 120 images)

Exemples :
  python curate_dataset.py --analyze --zw-dir data/zerowaste/
  python curate_dataset.py --auto --n 200 --zw-dir data/zerowaste/ --out data/zerowaste_curated/
  python curate_dataset.py --grid  --n 200 --zw-dir data/zerowaste/
  python curate_dataset.py --indices 003,012,045 --zw-dir data/zerowaste/ --out data/zerowaste_curated/
        """
    )
    parser.add_argument("--analyze",  action="store_true")
    parser.add_argument("--auto",     action="store_true")
    parser.add_argument("--grid",     action="store_true")
    parser.add_argument("--indices",  type=str, default=None)
    parser.add_argument("--n",        type=int, default=200)
    parser.add_argument("--min-dist", type=int, default=30)
    parser.add_argument("--out",      type=str, default=None)
    parser.add_argument("--zw-dir",   type=str, default=None,
                        help="Dossier ZeroWaste ORIGINAL (obligatoire)")
    args = parser.parse_args()

    if not any([args.analyze, args.auto, args.grid, args.indices]):
        parser.print_help(); sys.exit(0)

    # Résoudre le dossier source
    zw_dir = args.zw_dir or config.ZEROWASTE_DIR
    n_imgs = sum(
        len(list((Path(zw_dir)/split/"data").glob("*")))
        for split in ["train","val","test"]
        if (Path(zw_dir)/split/"data").exists()
    )
    print(f"Source : {zw_dir}  ({n_imgs} images)")

    if n_imgs < 500:
        print(f"\n[ATTENTION] Seulement {n_imgs} images — c'est peut-être le dossier curé.")
        print(f"  Précisez le dataset original avec --zw-dir, par exemple :")
        print(f"  python curate_dataset.py --auto --n 200 "
              f"--zw-dir data/zerowaste/ --out data/zerowaste_curated/")
        rep = input("Continuer quand même ? (o/n) : ").strip().lower()
        if rep != "o":
            sys.exit(0)

    if args.analyze:
        for split in ["train", "val", "test"]:
            sd = os.path.join(zw_dir, split)
            if os.path.isdir(os.path.join(sd, "data")):
                print(f"\n── {split} ──"); analyze_dataset(sd)

    if args.auto:
        auto_select(zw_dir, n=args.n, min_hamming=args.min_dist, out_dir=args.out)

    if args.grid:
        grid_select(zw_dir, n=args.n, out_dir=args.out)

    if args.indices:
        if not args.out:
            print("[ERREUR] --out requis avec --indices"); sys.exit(1)
        select_by_indices(zw_dir, args.indices, args.out)


if __name__ == "__main__":
    main()
