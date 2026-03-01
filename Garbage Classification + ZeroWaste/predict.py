"""
predict.py — Inférence sur images ou dossiers
==============================================
Usage :
  python predict.py --image chemin/image.jpg
  python predict.py --folder chemin/dossier/
  python predict.py --image img.jpg --true-class rigid_plastic
  python predict.py --eval-folder data/zerowaste/test/data/
"""

import os, sys, argparse, random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import config
from model import UNetResNet18

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)


# ─── Chargement modèle ───────────────────────────────────────────────────────

def load_model(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"[ERREUR] Checkpoint introuvable : {checkpoint_path}")
        print("  → Lancez d'abord : python main.py --step train")
        sys.exit(1)

    device = torch.device("cpu")
    model  = UNetResNet18(num_classes=config.NUM_CLASSES,
                           pretrained=False).to(device)
    ckpt   = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Modèle chargé — epoch {ckpt.get('epoch','?')} | "
          f"val mIoU = {ckpt.get('miou', 0):.4f}")
    return model, device


# ─── Pré-traitement ──────────────────────────────────────────────────────────

def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (config.IMG_SIZE, config.IMG_SIZE))
    img_f   = (img_res.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0), img_rgb


# ─── Inférence ───────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(img_bgr, model, device):
    """Retourne pred_mask (H,W uint8), confidence (H,W float), scores par classe."""
    h, w = img_bgr.shape[:2]
    tensor, _ = preprocess(img_bgr)
    tensor    = tensor.to(device)

    logits = model(tensor)                          # (1, C, H, W)
    probs  = F.softmax(logits, dim=1)[0]            # (C, H, W)
    pred   = probs.argmax(0).cpu().numpy()          # (H, W)
    conf   = probs.max(0).values.cpu().numpy()      # (H, W)

    # Redimensionner à la taille originale
    pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    conf = cv2.resize(conf,                  (w, h), interpolation=cv2.INTER_LINEAR)

    # Score par classe = fraction de pixels prédits (hors fond)
    total_fg = max((pred > 0).sum(), 1)
    scores   = {
        name: float((pred == i).sum() / total_fg)
        for i, name in enumerate(config.CLASS_NAMES[1:], start=1)
    }
    dominant = max(scores, key=scores.get)
    conf_fg  = float(conf[pred > 0].mean()) if (pred > 0).any() else 0.0

    return {
        "pred_mask": pred,
        "confidence": conf,
        "scores": scores,
        "dominant": dominant,
        "conf_fg": conf_fg,
    }


# ─── Visualisation ───────────────────────────────────────────────────────────

def visualize(img_bgr, result, true_class=None, save_path=None, show=False):
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pred      = result["pred_mask"]
    conf      = result["confidence"]
    scores    = result["scores"]
    dominant  = result["dominant"]
    color_arr = np.array(config.CLASS_COLORS, dtype=np.uint8)

    pred_color = color_arr[pred.clip(0, config.NUM_CLASSES - 1)]
    overlay    = (img_rgb.astype(float) * 0.5 + pred_color.astype(float) * 0.5).astype(np.uint8)

    correct = true_class and true_class.lower() == dominant.lower()
    status  = ""
    if true_class:
        status = "  ✓" if correct else f"  ✗ attendu : {true_class}"
    title_color = "green" if (not true_class or correct) else "red"

    fig = plt.figure(figsize=(18, 8))
    gs  = fig.add_gridspec(2, 5, hspace=0.4, wspace=0.35)

    ax_img  = fig.add_subplot(gs[:, 0])
    ax_pred = fig.add_subplot(gs[:, 1])
    ax_over = fig.add_subplot(gs[:, 2])
    ax_conf = fig.add_subplot(gs[0, 3])
    ax_bar  = fig.add_subplot(gs[1, 3:])

    ax_img.imshow(img_rgb);                     ax_img.set_title("Image originale")
    ax_pred.imshow(pred_color);                 ax_pred.set_title("Segmentation")
    ax_over.imshow(overlay);                    ax_over.set_title("Overlay")
    im = ax_conf.imshow(conf, cmap="viridis", vmin=0, vmax=1)
    ax_conf.set_title("Confiance")
    plt.colorbar(im, ax=ax_conf, fraction=0.046, pad=0.04)
    for ax in [ax_img, ax_pred, ax_over, ax_conf]:
        ax.axis("off")

    # Légende
    patches = [mpatches.Patch(color=np.array(c)/255, label=n)
               for c, n in zip(config.CLASS_COLORS[1:], config.CLASS_NAMES[1:])]
    ax_pred.legend(handles=patches, loc="lower center",
                   bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7)

    # Barplot scores
    names  = list(scores.keys())
    values = list(scores.values())
    colors_b = [np.array(config.CLASS_COLORS[i+1])/255 for i in range(len(names))]
    bars = ax_bar.bar(names, values, color=colors_b, edgecolor="white")
    ax_bar.set_ylim(0, 1); ax_bar.set_ylabel("Fraction pixels FG")
    ax_bar.set_title("Score par classe"); ax_bar.tick_params(axis="x", rotation=20)
    ax_bar.spines["top"].set_visible(False); ax_bar.spines["right"].set_visible(False)
    for bar, val in zip(bars, values):
        if val > 0.02:
            ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                        f"{val:.2f}", ha="center", fontsize=8)

    fig.suptitle(
        f"Prédiction : {dominant.upper()}  (conf. {result['conf_fg']:.1%}){status}",
        fontsize=13, fontweight="bold", color=title_color
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
        print(f"  → {save_path}")
    if show:
        plt.show()
    plt.close()


# ─── Modes ───────────────────────────────────────────────────────────────────

def predict_single(image_path, model, device, true_class=None,
                   out_dir=config.OUTPUT_DIR, show=False):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERREUR] Image illisible : {image_path}"); return

    result   = predict(img, model, device)
    dominant = result["dominant"]

    print(f"\n  Fichier   : {Path(image_path).name}")
    print(f"  Prédiction: {dominant.upper()}  (conf. {result['conf_fg']:.1%})")
    if true_class:
        ok = true_class.lower() == dominant.lower()
        print(f"  Vérité    : {true_class}  {'✓' if ok else '✗'}")
    print("  Scores :")
    for cls, sc in sorted(result["scores"].items(), key=lambda x: -x[1]):
        bar = "█" * int(sc * 30)
        print(f"    {cls:<16} {sc:>5.1%}  {bar}")

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"pred_{Path(image_path).stem}.png")
    visualize(img, result, true_class, save_path, show)


def predict_folder(folder_path, model, device, true_class=None,
                   max_images=50, show=False):
    import pandas as pd
    folder = Path(folder_path)
    files  = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}
    ])[:max_images]

    if not files:
        print(f"[ERREUR] Aucune image dans {folder_path}"); return

    out_dir = os.path.join(config.OUTPUT_DIR, "predictions", folder.name)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    print(f"\n{'─'*60}")
    print(f"  {'Fichier':<30} {'Prédit':<16} {'Conf':>6}  {'OK':>4}")
    print(f"{'─'*60}")

    for f in files:
        img = cv2.imread(str(f))
        if img is None: continue

        result   = predict(img, model, device)
        dominant = result["dominant"]
        conf     = result["conf_fg"]
        correct  = true_class.lower() == dominant.lower() if true_class else None

        mark = ("✓" if correct else "✗") if true_class else ""
        print(f"  {f.name:<30} {dominant:<16} {conf:>5.1%}  {mark}")

        save_path = os.path.join(out_dir, f"{f.stem}_pred.png")
        visualize(img, result, true_class, save_path, show)
        rows.append({"file": f.name, "predicted": dominant,
                     "true_class": true_class or "?",
                     "correct": correct, "confidence": round(conf, 4),
                     **{f"score_{k}": round(v, 4) for k, v in result["scores"].items()}})

    print(f"{'─'*60}")
    df = pd.DataFrame(rows)
    if true_class:
        acc = df["correct"].mean() * 100
        print(f"  Précision : {acc:.1f}%  ({int(df['correct'].sum())}/{len(df)} correctes)")
    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Résultats CSV → {csv_path}")
    print(f"  Visualisations → {out_dir}/")


def eval_all_classes(root_dir, model, device, max_per_class=30):
    """Évalue sur toutes les sous-dossiers (un dossier = une classe)."""
    root = Path(root_dir)
    results = {}

    for folder in sorted(root.iterdir()):
        if not folder.is_dir(): continue
        cls_name = folder.name.lower()

        files = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in {".jpg",".jpeg",".png"}
        ])
        sample = random.sample(files, min(max_per_class, len(files)))

        correct = 0
        for f in sample:
            img = cv2.imread(str(f))
            if img is None: continue
            res = predict(img, model, device)
            if res["dominant"].lower() == cls_name:
                correct += 1

        acc = correct / len(sample) * 100 if sample else 0
        results[cls_name] = {"n": len(sample), "correct": correct, "acc": acc}
        print(f"  {cls_name:<20} {acc:>5.1f}%  ({correct}/{len(sample)})")

    print(f"\n{'─'*40}")
    total_ok = sum(r["correct"] for r in results.values())
    total_n  = sum(r["n"]       for r in results.values())
    global_acc = total_ok / max(total_n, 1) * 100
    print(f"  {'GLOBAL':<20} {global_acc:>5.1f}%  ({total_ok}/{total_n})")

    # Barplot synthèse
    fig, ax = plt.subplots(figsize=(10, 4))
    names  = list(results.keys())
    accs   = [r["acc"] for r in results.values()]
    colors = [np.array(config.CLASS_COLORS[
        next((i for i,n in enumerate(config.CLASS_NAMES) if n==k), 0)
    ])/255 for k in names]
    bars = ax.bar(names, accs, color=colors, edgecolor="white")
    ax.axhline(global_acc, color="red", linestyle="--", label=f"Global {global_acc:.1f}%")
    ax.set_ylim(0, 105); ax.set_ylabel("Précision (%)")
    ax.set_title("Précision par classe (dominant pixel)", fontweight="bold")
    ax.legend(); ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, "eval_per_class.png")
    plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close()
    print(f"  Graphique → {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inférence segmentation déchets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python predict.py --image data/zerowaste/test/data/img001.jpg
  python predict.py --image img.jpg --true-class metal
  python predict.py --folder data/zerowaste/test/data/
  python predict.py --folder data/raw/glass/ --true-class rigid_plastic
  python predict.py --eval-folder data/raw/
        """
    )
    parser.add_argument("--image",        help="Image unique")
    parser.add_argument("--folder",       help="Dossier d'images")
    parser.add_argument("--eval-folder",  help="Dossier racine (sous-dossier = classe)")
    parser.add_argument("--true-class",   help=f"Classe réelle parmi : {config.CLASS_NAMES[1:]}")
    parser.add_argument("--checkpoint",   help="Chemin checkpoint .pth (défaut : best_model.pth)")
    parser.add_argument("--max",          type=int, default=50)
    parser.add_argument("--show",         action="store_true")
    args = parser.parse_args()

    if not any([args.image, args.folder, args.eval_folder]):
        parser.print_help(); sys.exit(0)

    model, device = load_model(args.checkpoint)

    if args.image:
        predict_single(args.image, model, device,
                        true_class=args.true_class, show=args.show)

    elif args.folder:
        predict_folder(args.folder, model, device,
                        true_class=args.true_class,
                        max_images=args.max, show=args.show)

    elif args.eval_folder:
        print(f"\n═══ Évaluation par classe sur {args.eval_folder} ═══")
        eval_all_classes(args.eval_folder, model, device, max_per_class=args.max)


if __name__ == "__main__":
    main()
