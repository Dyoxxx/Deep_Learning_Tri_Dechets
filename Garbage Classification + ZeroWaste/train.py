"""
train.py — U-Net ResNet-18, ZeroWaste + Garbage Classification, 5 classes
"""

import os, csv, time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

import config
from dataset import make_dataloaders
from model import UNetResNet18, CombinedLoss
from evaluate import compute_metrics


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    n = os.cpu_count() or 1
    torch.set_num_threads(n)
    torch.set_num_interop_threads(max(1, n//2))
    print(f"  CPU : {n} threads")
    return torch.device("cpu")


class WarmupCosine:
    def __init__(self, opt, warmup, total, min_lr=1e-6):
        self.opt = opt; self.w = warmup; self.T = total; self.min_lr = min_lr
        self.base = opt.param_groups[-1]["initial_lr"]

    def step(self, epoch):
        if epoch < self.w:
            scale = (epoch+1) / self.w
        else:
            p     = (epoch-self.w) / max(1, self.T-self.w)
            scale = (self.min_lr/self.base +
                     0.5*(1-self.min_lr/self.base)*(1+np.cos(np.pi*p)))
        for pg in self.opt.param_groups:
            pg["lr"] = pg["initial_lr"] * scale
        return self.opt.param_groups[-1]["lr"]


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, leave=False,
                    desc="train" if train else "val  ", dynamic_ncols=True)
        for img, mask in pbar:
            img  = img.to(device); mask = mask.to(device)
            logits = model(img)
            loss   = criterion(logits, mask)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            all_preds.append(logits.detach().argmax(1).cpu().numpy())
            all_targets.append(mask.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    m = compute_metrics(np.concatenate(all_preds),
                         np.concatenate(all_targets), config.NUM_CLASSES)
    m["loss"] = total_loss / len(loader)
    return m


def compute_class_weights(loader, device, max_batches=50):
    counts = np.zeros(config.NUM_CLASSES, np.float64)
    for i, (_, masks) in enumerate(loader):
        if i >= max_batches: break
        for c in range(config.NUM_CLASSES):
            counts[c] += (masks.numpy() == c).sum()

    total = counts.sum()
    print("  Distribution (% pixels) :")
    for name, cnt in zip(config.CLASS_NAMES, counts):
        print(f"    {name:<16} {cnt/total*100:>5.1f}%")

    freq = counts / (total + 1e-9)

    print("  Distribution réelle :")
    for name, fr in zip(config.CLASS_NAMES, freq):
        print(f"    {name:<16} {fr*100:.1f}%")

    # Poids équilibrés — ni trop bas (tout background) ni trop haut (tout cardboard)
    # Règle : toutes les classes objet entre 2.0 et 5.0
    # background légèrement pénalisé pour forcer la détection des objets
    w = np.array([
        0.5,   # background  — réduit pour forcer la détection d'objets
        4.0,   # rigid_plastic
        2.0,   # cardboard   — le plus fréquent des objets
        5.0,   # metal       — le plus rare
        3.5,   # soft_plastic
    ], dtype=np.float64)

    print("  Poids appliqués :")
    for name, wi in zip(config.CLASS_NAMES, w):
        print(f"    {name:<16} {wi:.2f}")
    return torch.tensor(w, dtype=torch.float32).to(device)


def train(num_epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE,
          lr=config.LR, patience=8):

    device = get_device()
    print(f"Device : {device}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR,     exist_ok=True)

    train_loader, val_loader, test_loader = make_dataloaders()

    print("\nCalcul poids de classe …")
    cls_w = compute_class_weights(train_loader, device)

    model     = UNetResNet18(num_classes=config.NUM_CLASSES, pretrained=True,
                              freeze_encoder=config.FREEZE_ENCODER).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modèle : {total/1e6:.1f}M params | {trainable/1e6:.1f}M entraînables")

    criterion = CombinedLoss(num_classes=config.NUM_CLASSES,
                              class_weights=cls_w).to(device)
    groups    = model.get_param_groups(lr)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = WarmupCosine(optimizer, warmup=3, total=num_epochs)

    log_f  = open(os.path.join(config.OUTPUT_DIR, "training_log.csv"), "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=[
        "epoch","lr","train_loss","val_loss",
        "train_miou","val_miou","val_miou_fg","time_s"
    ])
    writer.writeheader()

    best_miou = 0.0; patience_ct = 0; times = []
    history   = {"tl":[],"vl":[],"tm":[],"vm":[]}

    sep = "─"*78
    print(f"\n{sep}")
    print(f"{'Ep':>4} {'LR':>8} {'TrLoss':>8} {'VlLoss':>8} "
          f"{'TrmIoU':>8} {'VlmIoU':>8} {'FG-IoU':>7} {'ETA':>8}")
    print(sep)

    for epoch in range(1, num_epochs+1):
        t0     = time.time()
        lr_now = scheduler.step(epoch-1)
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl = run_epoch(model, val_loader,   criterion, None,      device, False)
        elapsed = time.time()-t0; times.append(elapsed)
        eta_s   = np.mean(times[-5:]) * (num_epochs-epoch)
        eta_str = f"{int(eta_s//60)}m{int(eta_s%60):02d}s"

        vl_fg = float(np.nanmean(vl["iou"][1:]))
        mark  = "★" if vl["miou"] > best_miou else " "
        print(f"{epoch:>4} {lr_now:>8.2e} {tr['loss']:>8.4f} {vl['loss']:>8.4f} "
              f"{tr['miou']:>8.4f} {vl['miou']:>8.4f} {vl_fg:>7.4f} "
              f"{eta_str:>8} {mark}")

        writer.writerow({
            "epoch": epoch, "lr": round(lr_now,8),
            "train_loss": round(tr["loss"],5), "val_loss": round(vl["loss"],5),
            "train_miou": round(tr["miou"],5), "val_miou": round(vl["miou"],5),
            "val_miou_fg": round(vl_fg,5), "time_s": round(elapsed,1)
        })
        log_f.flush()

        for k,v in [("tl",tr["loss"]),("vl",vl["loss"]),
                    ("tm",tr["miou"]),("vm",vl["miou"])]:
            history[k].append(v)

        if vl["miou"] > best_miou:
            best_miou = vl["miou"]; patience_ct = 0
            torch.save({"epoch": epoch, "state": model.state_dict(),
                        "miou": best_miou},
                       os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"\nEarly stopping (epoch {epoch})")
                break

    log_f.close()
    print(f"{sep}")
    print(f"Terminé en {sum(times)/60:.1f} min | Meilleur val mIoU = {best_miou:.4f}")

    _plot_curves(history)

    # Évaluation finale
    print("\n═══ Test set ═══")
    ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                      map_location=device)
    model.load_state_dict(ckpt["state"])
    te = run_epoch(model, test_loader, criterion, None, device, False)
    te_fg = float(np.nanmean(te["iou"][1:]))
    print(f"  mIoU global    : {te['miou']:.4f}")
    print(f"  mIoU sans fond : {te_fg:.4f}")
    print(f"  Pixel accuracy : {te['acc']:.4f}")
    print(f"\n  {'Classe':<16} {'IoU':>7}  {'F1':>7}")
    print("  " + "─"*35)
    for name, iou, f1 in zip(config.CLASS_NAMES, te["iou"], te["f1"]):
        bar = "█" * int((iou if not np.isnan(iou) else 0)*20)
        print(f"  {name:<16} {iou:>7.4f}  {f1:>7.4f}  {bar}")

    visualize(model, test_loader, device)
    return model


def _plot_curves(h):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax,(k1,k2),title in zip(axes, [("tl","vl"),("tm","vm")], ["Loss","mIoU"]):
        ax.plot(h[k1], label="train", lw=2); ax.plot(h[k2], label="val", lw=2)
        ax.set_title(title, fontweight="bold"); ax.legend()
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "training_curves.png"), dpi=110)
    plt.close()


@torch.no_grad()
def visualize(model, loader, device, n=6):
    model.eval()
    MEAN      = np.array([0.485, 0.456, 0.406])
    STD       = np.array([0.229, 0.224, 0.225])
    color_arr = np.array(config.CLASS_COLORS, dtype=np.uint8)

    collected = []
    for img, mask in loader:
        logits = model(img.to(device))
        preds  = logits.argmax(1).cpu()
        for i in range(img.shape[0]):
            collected.append((img[i], mask[i], preds[i]))
            if len(collected) >= n: break
        if len(collected) >= n: break

    fig, axes = plt.subplots(n, 4, figsize=(18, 4*n))
    if n == 1: axes = axes[np.newaxis]
    for row, (img_t, mask_t, pred_t) in enumerate(collected):
        img_np  = np.clip(img_t.permute(1,2,0).numpy()*STD+MEAN, 0, 1)
        gt_col  = color_arr[mask_t.numpy().clip(0, config.NUM_CLASSES-1)]
        pr_col  = color_arr[pred_t.numpy().clip(0, config.NUM_CLASSES-1)]
        overlay = (img_np*255*0.5 + pr_col*0.5).astype(np.uint8)
        axes[row,0].imshow(img_np);  axes[row,0].set_title("Image")
        axes[row,1].imshow(gt_col);  axes[row,1].set_title("GT")
        axes[row,2].imshow(pr_col);  axes[row,2].set_title("Prédiction")
        axes[row,3].imshow(overlay); axes[row,3].set_title("Overlay")
        for ax in axes[row]: ax.axis("off")

    patches = [mpatches.Patch(color=np.array(c)/255, label=nm)
               for c, nm in zip(config.CLASS_COLORS, config.CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center",
               ncol=config.NUM_CLASSES, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    plt.suptitle("Prédictions — test set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "predictions.png")
    plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()
    print(f"Visualisations → {path}")


if __name__ == "__main__":
    train()
