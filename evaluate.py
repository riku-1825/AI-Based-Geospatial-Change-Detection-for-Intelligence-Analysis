"""
evaluate.py -- run the trained model over the test split and report
Precision / Recall / F1 / IoU / Overall Accuracy, plus save a handful of
qualitative before/after/mask figures.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --config configs/config.yaml \
        --n_qualitative 6 --out_dir outputs/eval
"""
import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ChangeDetectionDataset
from models.siamese_unet import build_model
from utils.metrics import ConfusionMeter, changed_area_stats
from utils.visualize import save_comparison_figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_qualitative", type=int, default=6)
    parser.add_argument("--out_dir", default="outputs/eval")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(backbone=cfg["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val F1={ckpt['val_metrics']['f1']:.4f})")

    ds = ChangeDetectionDataset(cfg["data_root"], split=args.split, img_size=None, train=False)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    print(f"Evaluating on {len(ds)} pairs from split='{args.split}'")

    meter = ConfusionMeter(threshold=cfg["threshold"])
    all_changed_pcts = []

    saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            img_a = batch["A"].to(device)
            img_b = batch["B"].to(device)
            mask = batch["mask"].to(device)

            logits = model(img_a, img_b)
            meter.update(logits, mask)

            probs = torch.sigmoid(logits)
            preds = (probs > cfg["threshold"]).float().cpu().numpy()

            for i in range(preds.shape[0]):
                stats = changed_area_stats(preds[i, 0])
                all_changed_pcts.append(stats["changed_pct"])

                if saved < args.n_qualitative:
                    save_path = os.path.join(args.out_dir, f"sample_{saved:03d}.png")
                    save_comparison_figure(
                        img_a[i].cpu(), img_b[i].cpu(),
                        mask[i, 0].cpu().numpy(), preds[i, 0],
                        save_path, changed_pct=stats["changed_pct"],
                    )
                    saved += 1

    metrics = meter.compute()
    print("\n===== Test Set Results =====")
    print(f"Precision        : {metrics['precision']*100:.2f}%")
    print(f"Recall           : {metrics['recall']*100:.2f}%")
    print(f"F1 Score         : {metrics['f1']*100:.2f}%")
    print(f"IoU              : {metrics['iou']*100:.2f}%")
    print(f"Overall Accuracy : {metrics['overall_accuracy']*100:.2f}%")
    print(f"\nMean predicted changed area over test set: {np.mean(all_changed_pcts):.2f}%")
    print(f"Mean predicted unchanged area over test set: {100 - np.mean(all_changed_pcts):.2f}%")
    print(f"\nQualitative figures saved to: {args.out_dir}")

    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"mean_changed_pct: {np.mean(all_changed_pcts):.2f}\n")


if __name__ == "__main__":
    main()
