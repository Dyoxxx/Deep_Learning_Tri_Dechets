"""
predict.py
==========
Prédiction de segmentation sémantique sur une ou plusieurs images de déchets.

Deux modes :
  1. Image seule sur fond blanc (déchet individuel)
     → On détecte automatiquement le déchet, on le segmente, on affiche la classe.

  2. Scène composée (plusieurs déchets)
     → Segmentation complète pixel par pixel.

Usages :
    # Image unique
    python predict.py --image data/raw/plastic/img001.jpg

    # Dossier complet
    python predict.py --folder data/raw/glass/

    # Avec spécification de la classe réelle (pour évaluation)
    python predict.py --image img.jpg --true-class plastic

    # Mode batch sur toutes les classes (génère un rapport)
    python predict.py --eval-folder data/raw/
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import config
from mask_generation import generate_binary_mask
from model import UNetResNet18


# ─────────────────────────────────────────────────────────────────────────────
#  Chargement du modèle
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str = None,
               device: torch.device = None) -> torch.nn.Module:
    if device is None:
        device = torch.device("cpu")
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}\n"
            "→ Lancez d'abord : python main.py --step train"
        )

    model = UNetResNet18(num_classes=config.NUM_CLASSES,
                          pretrained=False,
                          freeze_encoder=False).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Modèle chargé (epoch {ckpt.get('epoch','?')}, "
          f"mIoU={ckpt.get('best_miou', 0):.4f})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Pré-traitement d'une image
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr: np.ndarray,
               size: int = config.IMG_SIZE) -> torch.Tensor:
    """
    Redimensionne + normalise une image BGR → tensor (1, 3, H, W).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    img_f   = img_res.astype(np.float32) / 255.0
    img_n   = (img_f - IMAGENET_MEAN) / IMAGENET_STD
    tensor  = torch.from_numpy(img_n.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
#  Inférence sur une image
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single(img_bgr: np.ndarray,
                   model: torch.nn.Module,
                   device: torch.device) -> dict:
    """
    Prédit le masque sémantique d'une image.

    Retourne un dict avec :
      • pred_mask    : (H_orig, W_orig) uint8 — indices de classe
      • confidence   : (H_orig, W_orig) float32 — probabilité max par pixel
      • class_scores : {class_name: score} — score global par classe (hors fond)
      • dominant     : classe dominante (hors fond)
    """
    h_orig, w_orig = img_bgr.shape[:2]
    tensor = preprocess(img_bgr).to(device)

    logits = model(tensor)                          # (1, C, H, W)
    probs  = F.softmax(logits, dim=1)[0]            # (C, H, W)
    pred   = probs.argmax(dim=0).cpu().numpy()      # (H, W)
    conf   = probs.max(dim=0).values.cpu().numpy()  # (H, W)

    # Redimensionner à la taille originale
    pred_orig = cv2.resize(pred.astype(np.uint8),
                           (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    conf_orig = cv2.resize(conf,
                           (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # Score par classe : fraction de pixels prédits comme cette classe (hors fond)
    total_fg = max((pred_orig > 0).sum(), 1)
    class_scores = {}
    for i, name in enumerate(config.CLASS_NAMES[1:], start=1):
        px = (pred_orig == i).sum()
        class_scores[name] = float(px / total_fg) if total_fg > 0 else 0.0

    dominant = max(class_scores, key=class_scores.get)

    return {
        "pred_mask":    pred_orig,
        "confidence":   conf_orig,
        "class_scores": class_scores,
        "dominant":     dominant,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Mode "déchet individuel sur fond blanc"
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_individual_waste(img_bgr: np.ndarray,
                              model: torch.nn.Module,
                              device: torch.device) -> dict:
    """
    Spécialisé pour les images de déchets SEULS sur fond blanc.

    Pipeline :
      1. Détecte automatiquement le masque objet (flood fill + morpho)
      2. Recadre sur le bounding box du déchet (zoom naturel)
      3. Applique le modèle sur le crop (meilleure résolution effective)
      4. Recolle le résultat dans l'image originale

    Avantage par rapport au mode générique :
      Le déchet occupe 100 % de l'image 128×128 → bien meilleure segmentation.
    """
    h, w = img_bgr.shape[:2]

    # Étape 1 : Détecter l'objet
    obj_mask = generate_binary_mask(img_bgr)
    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Aucun objet détecté → inférence sur l'image complète
        return predict_single(img_bgr, model, device)

    # Étape 2 : Bounding box avec marge
    x, y, bw, bh = cv2.boundingRect(np.vstack(contours))
    margin = 10
    x1 = max(0, x - margin);  y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin); y2 = min(h, y + bh + margin)

    crop = img_bgr[y1:y2, x1:x2]

    # Étape 3 : Inférence sur le crop
    result_crop = predict_single(crop, model, device)

    # Étape 4 : Recoller dans l'image originale
    pred_full = np.zeros((h, w), dtype=np.uint8)   # tout = background
    conf_full = np.zeros((h, w), dtype=np.float32)

    # N'assigner la prédiction que là où l'objet a été détecté (obj_mask)
    crop_h, crop_w = (y2 - y1), (x2 - x1)
    pred_crop_resized = cv2.resize(result_crop["pred_mask"],
                                    (crop_w, crop_h),
                                    interpolation=cv2.INTER_NEAREST)
    conf_crop_resized = cv2.resize(result_crop["confidence"],
                                    (crop_w, crop_h),
                                    interpolation=cv2.INTER_LINEAR)

    # Masque objet dans la zone du crop
    obj_crop_mask = obj_mask[y1:y2, x1:x2] > 127

    pred_full[y1:y2, x1:x2][obj_crop_mask] = pred_crop_resized[obj_crop_mask]
    conf_full[y1:y2, x1:x2][obj_crop_mask] = conf_crop_resized[obj_crop_mask]

    # Score global inchangé (calculé sur le crop)
    result_crop["pred_mask"]  = pred_full
    result_crop["confidence"] = conf_full
    result_crop["obj_mask"]   = obj_mask
    result_crop["crop_box"]   = (x1, y1, x2, y2)

    return result_crop


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_result(img_bgr: np.ndarray,
                     result: dict,
                     true_class: str = None,
                     save_path: str = None,
                     show: bool = False) -> None:
    """
    Génère une figure de résultat complète :
      • Image originale
      • Masque de segmentation coloré
      • Overlay semi-transparent
      • Carte de confiance
      • Barres de scores par classe
    """
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pred_mask = result["pred_mask"]
    confidence= result["confidence"]
    scores    = result["class_scores"]
    dominant  = result["dominant"]

    # Masque coloré
    color_arr  = np.array(config.CLASS_COLORS, dtype=np.uint8)
    pred_color = color_arr[pred_mask.clip(0, len(color_arr) - 1)]

    # Overlay
    overlay = (img_rgb.astype(float) * 0.5 +
               pred_color.astype(float) * 0.5).astype(np.uint8)

    # Figure
    fig = plt.figure(figsize=(18, 9))
    gs  = fig.add_gridspec(2, 5, hspace=0.35, wspace=0.35)

    ax_img  = fig.add_subplot(gs[:, 0])
    ax_mask = fig.add_subplot(gs[:, 1])
    ax_over = fig.add_subplot(gs[:, 2])
    ax_conf = fig.add_subplot(gs[0, 3])
    ax_bar  = fig.add_subplot(gs[1, 3:])

    ax_img.imshow(img_rgb);              ax_img.set_title("Image originale", fontweight="bold")
    ax_mask.imshow(pred_color);          ax_mask.set_title("Segmentation prédite", fontweight="bold")
    ax_over.imshow(overlay);             ax_over.set_title("Overlay", fontweight="bold")
    ax_conf.imshow(confidence, cmap="viridis", vmin=0, vmax=1)
    ax_conf.set_title("Confiance", fontweight="bold")
    plt.colorbar(ax_conf.images[0], ax=ax_conf, fraction=0.046, pad=0.04)

    for ax in [ax_img, ax_mask, ax_over, ax_conf]:
        ax.axis("off")

    # Légende couleurs
    patches = [mpatches.Patch(color=np.array(c)/255, label=n)
               for c, n in zip(config.CLASS_COLORS[1:], config.CLASS_NAMES[1:])]
    ax_mask.legend(handles=patches, loc="lower center",
                   bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7)

    # Barres de scores
    names  = list(scores.keys())
    values = list(scores.values())
    colors_bar = [np.array(config.CLASS_COLORS[i+1]) / 255
                  for i in range(len(names))]
    bars   = ax_bar.bar(names, values, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Score (fraction pixels)")
    ax_bar.set_title("Score par classe", fontweight="bold")
    ax_bar.tick_params(axis="x", rotation=25)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for bar, val in zip(bars, values):
        if val > 0.02:
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.01, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=8)

    # Titre principal
    correct = (true_class is not None and true_class.lower() == dominant.lower())
    status  = ""
    if true_class:
        status = f"  ✓ Correct" if correct else f"  ✗ Attendu : {true_class}"
    conf_mean = float(confidence[pred_mask > 0].mean()) if (pred_mask > 0).any() else 0.0
    fig.suptitle(
        f"Prédiction : {dominant.upper()}  (confiance moy. {conf_mean:.2%}){status}",
        fontsize=14, fontweight="bold",
        color="green" if (not true_class or correct) else "red"
    )

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  → Sauvegardé : {save_path}")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Mode batch / rapport
# ─────────────────────────────────────────────────────────────────────────────

def predict_folder(folder: str, model, device,
                   true_class: str = None,
                   out_dir: str = None,
                   max_images: int = 20) -> None:
    """Prédit toutes les images d'un dossier et génère un rapport."""
    import pandas as pd

    folder_path = Path(folder)
    files = sorted([f for f in folder_path.iterdir()
                    if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}])[:max_images]

    if not files:
        print(f"[ERREUR] Aucune image trouvée dans {folder}")
        return

    out_dir = out_dir or os.path.join(config.OUTPUT_DIR, "predictions", folder_path.name)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        result = predict_individual_waste(img, model, device)
        dominant = result["dominant"]
        conf_mean = float(result["confidence"][result["pred_mask"] > 0].mean()) \
                    if (result["pred_mask"] > 0).any() else 0.0

        save_path = os.path.join(out_dir, f"{f.stem}_pred.png")
        visualize_result(img, result, true_class=true_class,
                          save_path=save_path)

        correct = true_class is not None and true_class.lower() == dominant.lower()
        rows.append({
            "file": f.name,
            "predicted": dominant,
            "true_class": true_class or "?",
            "correct": correct if true_class else "N/A",
            "confidence": round(conf_mean, 4),
            **{f"score_{k}": round(v, 4) for k, v in result["class_scores"].items()}
        })
        print(f"  {f.name:<30} → {dominant:<12} (conf={conf_mean:.2%})"
              + (f"  {'✓' if correct else '✗'}" if true_class else ""))

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "predictions.csv")
        df.to_csv(csv_path, index=False)
        if true_class:
            acc = df["correct"].mean() * 100
            print(f"\nPrécision sur {len(df)} images : {acc:.1f}%")
        print(f"Rapport → {csv_path}")


def eval_all_classes(raw_dir: str, model, device) -> None:
    """Évalue le modèle sur toutes les classes et affiche un tableau de synthèse."""
    raw_path = Path(raw_dir)
    results  = {}

    for folder in sorted(raw_path.iterdir()):
        if not folder.is_dir():
            continue
        cls = folder.name.lower()
        if cls not in config.FOLDER_TO_CLASS:
            continue
        print(f"\n── Évaluation classe : {cls}")
        files = sorted([f for f in folder.iterdir()
                        if f.suffix.lower() in {".jpg",".jpeg",".png"}])[:30]
        correct = 0
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            res = predict_individual_waste(img, model, device)
            if res["dominant"].lower() == cls.lower():
                correct += 1
        acc = correct / len(files) * 100 if files else 0
        results[cls] = {"n": len(files), "correct": correct, "accuracy": acc}
        print(f"   {correct}/{len(files)} correctes = {acc:.1f}%")

    print("\n═══ SYNTHÈSE ═══")
    for cls, r in results.items():
        bar = "█" * int(r["accuracy"] / 5)
        print(f"  {cls:<12} {r['accuracy']:>5.1f}%  {bar}")
    overall = sum(r["correct"] for r in results.values()) / \
              max(sum(r["n"] for r in results.values()), 1) * 100
    print(f"  {'GLOBAL':<12} {overall:>5.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prédiction de segmentation sémantique sur déchets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python predict.py --image data/raw/plastic/img001.jpg
  python predict.py --image data/raw/paper/img005.jpg --true-class paper
  python predict.py --folder data/raw/glass/
  python predict.py --eval-folder data/raw/
        """
    )
    parser.add_argument("--image",       help="Chemin vers une image unique")
    parser.add_argument("--folder",      help="Dossier d'images à prédire")
    parser.add_argument("--eval-folder", help="Dossier racine (une sous-dossier par classe)")
    parser.add_argument("--true-class",  help="Classe réelle (optionnel, pour évaluation)")
    parser.add_argument("--checkpoint",  help="Chemin vers le checkpoint .pth")
    parser.add_argument("--show",        action="store_true",
                                          help="Afficher les figures (requiert display)")
    parser.add_argument("--max",         type=int, default=20,
                                          help="Nb max d'images par dossier")
    args = parser.parse_args()

    if not any([args.image, args.folder, args.eval_folder]):
        parser.print_help()
        sys.exit(0)

    device = torch.device("cpu")
    model  = load_model(args.checkpoint, device)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"[ERREUR] Impossible de lire : {args.image}")
            sys.exit(1)
        result   = predict_individual_waste(img, model, device)
        dominant = result["dominant"]
        conf     = float(result["confidence"][result["pred_mask"] > 0].mean()) \
                   if (result["pred_mask"] > 0).any() else 0.0
        print(f"\n Prédiction : {dominant.upper()}  (confiance : {conf:.2%})")
        print(" Scores par classe :")
        for cls, score in sorted(result["class_scores"].items(),
                                   key=lambda x: -x[1]):
            bar = "█" * int(score * 30)
            print(f"   {cls:<12} {score:>5.1%}  {bar}")

        stem      = Path(args.image).stem
        save_path = os.path.join(config.OUTPUT_DIR, f"pred_{stem}.png")
        visualize_result(img, result, args.true_class, save_path, args.show)

    elif args.folder:
        predict_folder(args.folder, model, device,
                        true_class=args.true_class, max_images=args.max)

    elif args.eval_folder:
        eval_all_classes(args.eval_folder, model, device)


if __name__ == "__main__":
    main()
