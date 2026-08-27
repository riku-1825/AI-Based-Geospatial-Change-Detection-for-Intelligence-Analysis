"""
train.py -- train the Siamese U-Net change detection model.

Usage:
    python train.py --config path/to/config.yaml
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import ChangeDetectionDataset
from models.siamese_unet import build_model
from utils.losses import BCEDiceLoss
from utils.metrics import ConfusionMeter
from utils.visualize import plot_training_curves


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True, amp=True):
    model.train() if train else model.eval()
    meter = ConfusionMeter()
    total_loss = 0.0
    n_batches = 0

    torch.set_grad_enabled(train)
    pbar = tqdm(loader, desc="train" if train else "val")
    for batch in pbar:
        img_a = batch["A"].to(device, non_blocking=True)
        img_b = batch["B"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
            logits = model(img_a, img_b)
            loss = criterion(logits, mask)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        meter.update(logits, mask)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = meter.compute()
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["log_dir"])

    train_ds = ChangeDetectionDataset(cfg["data_root"], split="train", img_size=cfg["img_size"])
    val_ds = ChangeDetectionDataset(cfg["data_root"], split="val", img_size=None, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    model = build_model(backbone=cfg["backbone"], pretrained=cfg["pretrained"]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M")

    criterion = BCEDiceLoss(bce_weight=cfg["bce_weight"], pos_weight=cfg.get("pos_weight"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["amp"] and device.type == "cuda"))

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_iou": []}
    best_f1 = -1.0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                   train=True, amp=cfg["amp"])
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, device,
                                 train=False, amp=cfg["amp"])
        scheduler.step()
        dt = time.time() - t0

        print(
            f"[Epoch {epoch:03d}/{cfg['epochs']}] "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_prec={val_metrics['precision']:.4f} "
            f"val_rec={val_metrics['recall']:.4f} | {dt:.1f}s"
        )

        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_iou"].append(val_metrics["iou"])

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], "last.pt"))
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], "best.pt"))
            print(f"  -> new best model saved (F1={best_f1:.4f})")

    plot_training_curves(history, os.path.join(cfg["checkpoint_dir"], "training_curves.png"))
    writer.close()
    print("Training complete. Best val F1:", best_f1)


if __name__ == "__main__":
    main()
