"""
train.py — avec Differential LR + Focal Loss
"""

import os, csv, time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from dataset import make_dataloaders
from model import UNetResNet18, CombinedLoss, compute_class_weights
from evaluate import compute_metrics


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    n = os.cpu_count() or 1
    torch.set_num_threads(n)
    torch.set_num_interop_threads(max(1, n//2))
    print(f"  CPU : {n} threads")
    return torch.device("cpu")


class WarmupCosineScheduler:
    def __init__(self, opt, warmup, total, min_lr=1e-6):
        self.opt=opt; self.warmup=warmup; self.total=total
        self.min_lr=min_lr; self.base=opt.param_groups[-1]["lr"]  # LR décodeur

    def step(self, epoch):
        if epoch < self.warmup:
            scale = (epoch+1)/self.warmup
        else:
            p     = (epoch-self.warmup)/max(1, self.total-self.warmup)
            scale = self.min_lr/self.base + 0.5*(1-self.min_lr/self.base)*(1+np.cos(np.pi*p))
        for pg in self.opt.param_groups:
            pg["lr"] = pg["initial_lr"] * scale
        return self.opt.param_groups[-1]["lr"]


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, leave=False, desc="train" if train else "val  ",
                    dynamic_ncols=True)
        for images, masks, _ in pbar:
            images = images.to(device); masks = masks.to(device)
            logits = model(images)
            loss   = criterion(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            all_preds.append(logits.detach().argmax(1).cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    p = np.concatenate(all_preds);  t = np.concatenate(all_targets)
    m = compute_metrics(p, t, config.NUM_CLASSES)
    m["loss"] = total_loss / len(loader)
    return m


def train(scene_dir=config.SCENE_DIR, num_epochs=config.NUM_EPOCHS,
          batch_size=config.BATCH_SIZE, lr=config.LR, patience=7):

    device = get_device()
    print(f"Device : {device}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR,     exist_ok=True)

    print("Chargement DataLoaders …")
    train_loader, val_loader, _ = make_dataloaders(scene_dir, batch_size)
    print(f"  Train:{len(train_loader.dataset)} | Val:{len(val_loader.dataset)}"
          f" | {len(train_loader)} steps/epoch")

    print("Calcul poids de classe …")
    cls_w = compute_class_weights(scene_dir, str(device))

    model = UNetResNet18(pretrained=True,
                          freeze_encoder=config.FREEZE_ENCODER).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modèle : {total/1e6:.1f}M total | {trainable/1e6:.1f}M entraînables")

    criterion = CombinedLoss(class_weights=cls_w).to(device)

    # Differential LR : encodeur à lr/10
    param_groups = model.get_param_groups(lr)
    for pg in param_groups:
        pg["initial_lr"] = pg["lr"]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, warmup=3, total=num_epochs)

    log_path = os.path.join(config.OUTPUT_DIR, "training_log.csv")
    f_csv    = open(log_path, "w", newline="")
    writer   = csv.DictWriter(f_csv, fieldnames=[
        "epoch","lr_decoder","train_loss","val_loss",
        "train_miou","val_miou","epoch_time_s"
    ])
    writer.writeheader()

    best_miou = 0.0; patience_ct = 0
    epoch_times = []
    history = {"train_loss":[],"val_loss":[],"train_miou":[],"val_miou":[]}

    sep = "─"*75
    print(f"\n{sep}")
    print(f"{'Ep':>4} {'LR-dec':>8} {'TrLoss':>8} {'VlLoss':>8} "
          f"{'TrIoU':>7} {'VlIoU':>7} {'Time':>6}  {'ETA':>7}")
    print(sep)

    for epoch in range(1, num_epochs+1):
        t0     = time.time()
        lr_now = scheduler.step(epoch-1)
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl = run_epoch(model, val_loader,   criterion, None,      device, False)

        elapsed = time.time()-t0
        epoch_times.append(elapsed)
        eta_s   = np.mean(epoch_times[-5:]) * (num_epochs-epoch)
        eta_str = f"{int(eta_s//60)}m{int(eta_s%60):02d}s"

        mark = "★" if vl["miou"] > best_miou else " "
        print(f"{epoch:>4} {lr_now:>8.2e} {tr['loss']:>8.4f} {vl['loss']:>8.4f} "
              f"{tr['miou']:>7.4f} {vl['miou']:>7.4f} {elapsed:>5.1f}s  {eta_str:>7} {mark}")

        writer.writerow({"epoch":epoch,"lr_decoder":round(lr_now,8),
                         "train_loss":round(tr["loss"],5),"val_loss":round(vl["loss"],5),
                         "train_miou":round(tr["miou"],5),"val_miou":round(vl["miou"],5),
                         "epoch_time_s":round(elapsed,1)})
        f_csv.flush()

        for k in history:
            src = tr if "train" in k else vl
            history[k].append(src[k.split("_")[1]])

        if vl["miou"] > best_miou:
            best_miou = vl["miou"]; patience_ct = 0
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                        "best_miou":best_miou},
                       os.path.join(config.CHECKPOINT_DIR,"best_model.pth"))
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"\nEarly stopping à l'epoch {epoch}")
                break

    f_csv.close()
    total_min = sum(epoch_times)/60
    print(f"{sep}")
    print(f"Terminé en {total_min:.1f} min | Meilleur val mIoU = {best_miou:.4f}")
    _plot(history)
    return model


def _plot(h):
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for ax, keys, title in zip(axes,
        [("train_loss","val_loss"),("train_miou","val_miou")],["Loss","mIoU"]):
        ax.plot(h[keys[0]], label="train", lw=2)
        ax.plot(h[keys[1]], label="val",   lw=2)
        ax.set_title(title, fontweight="bold"); ax.legend()
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p = os.path.join(config.OUTPUT_DIR,"training_curves.png")
    plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close()
    print(f"Courbes → {p}")


if __name__ == "__main__":
    train()
