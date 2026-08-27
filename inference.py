"""
inference.py -- run change detection on a whole folder of before/after image
pairs (any size; sliding-window if larger than training crop size).

Before/after images are matched by filename, e.g.:
    before_folder/tile_001.png  <->  after_folder/tile_001.png
    before_folder/tile_002.png  <->  after_folder/tile_002.png

Usage:
    python inference.py --before path/to/before_folder --after path/to/after_folder \
        --checkpoint checkpoints/best.pt --config configs/config.yaml \
        --out outputs/inference_results

All results (comparison figure, binary mask, per-image stats, and a
summary CSV across the whole folder) are written into the --out directory.
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from models.siamese_unet import build_model
from utils.metrics import changed_area_stats
from utils.visualize import save_comparison_figure, IMAGENET_MEAN, IMAGENET_STD

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

PIPELINE_DIAGRAM = """
   Before Image
        |
        v
   After Image
        |
        v
   Feature Extraction   (shared ResNet encoder, Siamese)
        |
        v
   Feature Difference   (|F_before - F_after| at 4 scales)
        |
        v
   Change Detection     (U-Net decoder -> per-pixel logits)
        |
        v
   Change Mask          (sigmoid + threshold)
"""


def load_and_normalize(path):
    img = Image.open(path).convert("RGB")
    orig_size = img.size  # (W, H)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    return tensor, orig_size


def sliding_window_predict(model, img_a, img_b, patch_size, device, stride=None):
    """img_a/img_b: (3,H,W) normalized tensors. Returns full-res probability map."""
    stride = stride or patch_size
    _, H, W = img_a.shape

    if H <= patch_size and W <= patch_size:
        pad_h, pad_w = patch_size - H, patch_size - W
        a = F.pad(img_a.unsqueeze(0), (0, pad_w, 0, pad_h))
        b = F.pad(img_b.unsqueeze(0), (0, pad_w, 0, pad_h))
        with torch.no_grad():
            logits = model(a.to(device), b.to(device))
        prob = torch.sigmoid(logits)[0, 0, :H, :W].cpu().numpy()
        return prob

    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, x1 = min(y + patch_size, H), min(x + patch_size, W)
            y0, x0 = max(0, y1 - patch_size), max(0, x1 - patch_size)

            patch_a = img_a[:, y0:y1, x0:x1].unsqueeze(0).to(device)
            patch_b = img_b[:, y0:y1, x0:x1].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(patch_a, patch_b)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            prob_map[y0:y1, x0:x1] += prob
            count_map[y0:y1, x0:x1] += 1

    return prob_map / np.maximum(count_map, 1)


def find_matching_pairs(before_dir, after_dir):
    """Match files by filename between the two folders. Returns a sorted
    list of (name, before_path, after_path) tuples."""
    before_files = {f for f in os.listdir(before_dir) if f.lower().endswith(IMG_EXTENSIONS)}
    after_files = {f for f in os.listdir(after_dir) if f.lower().endswith(IMG_EXTENSIONS)}

    common = sorted(before_files & after_files)
    missing_after = sorted(before_files - after_files)
    missing_before = sorted(after_files - before_files)

    if missing_after:
        print(f"[warn] {len(missing_after)} file(s) in --before have no match in --after, skipping: "
              f"{missing_after[:5]}{' ...' if len(missing_after) > 5 else ''}")
    if missing_before:
        print(f"[warn] {len(missing_before)} file(s) in --after have no match in --before, skipping: "
              f"{missing_before[:5]}{' ...' if len(missing_before) > 5 else ''}")

    return [(name, os.path.join(before_dir, name), os.path.join(after_dir, name)) for name in common]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, help="Folder of 'before' images")
    parser.add_argument("--after", required=True, help="Folder of 'after' images")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default="outputs/inference_results",
                         help="Output folder where all results are written")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(backbone=cfg["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(PIPELINE_DIAGRAM)

    pairs = find_matching_pairs(args.before, args.after)
    if not pairs:
        print("No matching before/after filename pairs found. Nothing to do.")
        return
    print(f"Found {len(pairs)} matching before/after pairs. Running inference...\n")

    summary_rows = []
    for i, (name, before_path, after_path) in enumerate(pairs, 1):
        stem = os.path.splitext(name)[0]

        img_a, size_a = load_and_normalize(before_path)
        img_b, size_b = load_and_normalize(after_path)
        if size_a != size_b:
            print(f"[skip] {name}: before/after size mismatch ({size_a} vs {size_b})")
            continue

        prob_map = sliding_window_predict(model, img_a, img_b, cfg["img_size"], device)
        pred_mask = (prob_map > cfg["threshold"]).astype(np.uint8)

        stats = changed_area_stats(pred_mask)
        print(f"[{i}/{len(pairs)}] {name}: Changed Area = {stats['changed_pct']}% | "
              f"Unchanged Area = {stats['unchanged_pct']}%")

        fig_path = os.path.join(args.out, f"{stem}_result.png")
        save_comparison_figure(img_a, img_b, None, pred_mask, fig_path,
                                changed_pct=stats["changed_pct"])

        mask_path = os.path.join(args.out, f"{stem}_mask.png")
        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mask_path)

        summary_rows.append({
            "name": name,
            "changed_pct": stats["changed_pct"],
            "unchanged_pct": stats["unchanged_pct"],
            "changed_pixels": stats["changed_pixels"],
            "total_pixels": stats["total_pixels"],
        })

    # Write a single summary CSV covering the whole folder
    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "changed_pct", "unchanged_pct",
                                                "changed_pixels", "total_pixels"])
        writer.writeheader()
        writer.writerows(summary_rows)

    if summary_rows:
        mean_changed = np.mean([r["changed_pct"] for r in summary_rows])
        print(f"\nProcessed {len(summary_rows)} pairs.")
        print(f"Mean changed area across folder: {mean_changed:.2f}%")
        print(f"Mean unchanged area across folder: {100 - mean_changed:.2f}%")

    print(f"\nAll results (comparison figures, masks, summary.csv) saved to: {args.out}")


if __name__ == "__main__":
    main()
