"""
train.py — Training Loop for BraTS2020 3D U-Net
=================================================
Connects dataset → model → loss → optimizer into a full training pipeline.

Run:
    python train.py

Checkpoints saved to: checkpoints/best_model.pth
TensorBoard logs:     checkpoints/logs/
"""
from dotenv import load_dotenv
import os
load_dotenv()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np

from dataset import BraTSDataset
from model import UNet3D



# ─── Config ───────────────────────────────────────────────────────────────────
# All training hyperparameters in one place — easy to change without
# hunting through the code.

CONFIG = {
    "data_root":    os.getenv("DATA_ROOT"),
    "output_dir":   os.getenv("CHECKPOINT_PATH"),
    "epochs":       110,
    "batch_size":   1,       # 1 is the max for 128³ on ~10GB VRAM
    "lr":           1e-4,    # AdamW learning rate
    "num_workers":  2,       # parallel data loading — set to 0 on Windows if errors
    "base_filters": 32,
    "depth":        4,
    "seed":         42,
}


# ─── Loss Functions ───────────────────────────────────────────────────────────
# DiceLoss: computed per tumor class independently — handles class imbalance.
# CombinedLoss: Dice + CrossEntropy equally weighted.
#   Dice handles imbalance, CE provides stable per-voxel gradients.

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits:  (B, C, H, W, D) — raw model output
        # targets: (B, H, W, D)    — integer labels {0,1,2,3}
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, H, W, D) → (B, C, H, W, D)
        targets_oh = F.one_hot(targets.long(), num_classes)
        targets_oh = targets_oh.permute(0, 4, 1, 2, 3).float()

        # Skip class 0 (background) — we only care about tumor Dice
        dice_scores = []
        for c in range(1, num_classes):
            p = probs[:, c]
            t = targets_oh[:, c]
            intersection = (p * t).sum()
            dsc = (2 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_scores.append(dsc)

        # Return loss = 1 - mean Dice (minimizing loss = maximizing Dice)
        return 1 - torch.stack(dice_scores).mean()


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.ce   = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + \
               0.5 * self.ce(logits, targets.long())


# ─── BraTS Dice Metrics ───────────────────────────────────────────────────────
# Computes the three official BraTS evaluation region Dice scores.
# Called during validation — not used in the loss, only for monitoring.
#
# WT (Whole Tumor)   = labels {1,2,3}
# TC (Tumor Core)    = labels {1,3}
# ET (Enhancing)     = label  {3}

def compute_brats_dice(pred, target, smooth=1e-5):
    # pred, target: (H, W, D) numpy arrays with values {0,1,2,3}
    regions = {
        "WT": (pred > 0,              target > 0),
        "TC": (np.isin(pred, [1, 3]), np.isin(target, [1, 3])),
        "ET": (pred == 3,             target == 3),
    }
    scores = {}
    for name, (p, t) in regions.items():
        intersection = (p & t).sum()
        scores[name] = float(2 * intersection + smooth) / \
                       float(p.sum() + t.sum() + smooth)
    return scores


# ─── Training Loop (one epoch) ────────────────────────────────────────────────
# AMP (Automatic Mixed Precision): runs forward pass in float16 where safe,
# keeps weights in float32. Roughly 2× faster and halves VRAM usage.
# GradScaler prevents float16 underflow during backprop.

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0

    for step, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()

        with torch.amp.autocast("cuda"):                       # float16 forward pass
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()           # scaled backprop
        scaler.unscale_(optimizer)
        # Gradient clipping: prevents exploding gradients in deep 3D networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % 10 == 0:
            print(f"    step {step:3d}/{len(loader)}  loss: {loss.item():.4f}")

    return total_loss / len(loader)


# ─── Validation Loop ──────────────────────────────────────────────────────────
# Runs inference on the val set with no gradients (torch.no_grad saves memory).
# Computes mean Dice across WT/TC/ET — this is what we save the best model on.

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_dice   = {"WT": [], "TC": [], "ET": []}

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss   = criterion(logits, masks)

        total_loss += loss.item()

        # Argmax over class dim → predicted label map
        pred = torch.argmax(logits, dim=1).cpu().numpy()   # (B, H, W, D)
        gt   = masks.cpu().numpy()                          # (B, H, W, D)

        # Compute BraTS Dice per sample in batch
        for b in range(pred.shape[0]):
            scores = compute_brats_dice(pred[b], gt[b])
            for region, score in scores.items():
                all_dice[region].append(score)

    mean_dice = {r: float(np.mean(v)) for r, v in all_dice.items()}
    mean_dice["mean"] = float(np.mean(list(mean_dice.values())))
    return total_loss / len(loader), mean_dice


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device:     {device}")
    print(f"Output dir: {output_dir}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = BraTSDataset(CONFIG["data_root"], split="train", seed=CONFIG["seed"])
    val_ds   = BraTSDataset(CONFIG["data_root"], split="val",   seed=CONFIG["seed"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=CONFIG["num_workers"],
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"],
                              pin_memory=True)

    print(f"Train: {len(train_ds)} cases  |  Val: {len(val_ds)} cases")

    # ── Model ────────────────────────────────────────────────────────────────
    model = UNet3D(in_channels=4, out_channels=4,
                   base_filters=CONFIG["base_filters"],
                   depth=CONFIG["depth"]).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # ── Training components ──────────────────────────────────────────────────
    criterion = CombinedLoss()
    # AdamW: Adam + weight decay. Weight decay regularizes weights,
    # preventing overfitting on a 295-case dataset.
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-5)
    # CosineAnnealingLR: smoothly decays LR from lr → eta_min over all epochs.
    # Avoids the sharp drops of step schedulers that can destabilize training.
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")                # for AMP
    writer    = SummaryWriter(output_dir / "logs")  # TensorBoard

    best_dice  = 0.0

    # ── Resume from checkpoint ────────────────────────────────────────────────
    RESUME = "checkpoints/best_model.pth"   # set to None to start fresh
    start_epoch = 0

    if RESUME and Path(RESUME).exists():
        ckpt = torch.load(RESUME, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt["best_dice"]
        print(f"Resumed from epoch {ckpt['epoch']}  best Dice: {best_dice:.4f}")

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CONFIG["epochs"]):
        print(f"\nEpoch {epoch:03d}/{CONFIG['epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Val Dice — WT: {val_dice['WT']:.3f}  "
              f"TC: {val_dice['TC']:.3f}  "
              f"ET: {val_dice['ET']:.3f}  "
              f"Mean: {val_dice['mean']:.3f}")

        # TensorBoard logging — run: tensorboard --logdir checkpoints/logs
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        for region, score in val_dice.items():
            writer.add_scalar(f"Dice/{region}", score, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Save best model based on mean val Dice across WT/TC/ET
        if val_dice["mean"] > best_dice:
            best_dice = val_dice["mean"]
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice":             val_dice,
                "best_dice":            best_dice,
                "config":               CONFIG,
            }, output_dir / "best_model.pth")
            print(f"  ✅ Best model saved  (mean Dice: {best_dice:.4f})")

        # Periodic checkpoint every 50 epochs — lets you resume if training crashes
        if epoch % 50 == 0:
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
            }, output_dir / f"epoch_{epoch:03d}.pth")

    writer.close()
    print(f"\nTraining complete. Best mean Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()