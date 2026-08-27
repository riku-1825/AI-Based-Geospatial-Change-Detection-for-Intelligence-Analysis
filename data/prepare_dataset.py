"""
data/prepare_dataset.py

Two utilities in one script:

1. --synthetic : generate a LEVIR-CD-shaped synthetic dataset (before/after/
   label triplets) so you can build & debug the full pipeline before your
   real download finishes.

2. --patchify   : cut large source images (e.g. native 1024x1024 LEVIR-CD or
   giant WHU-CD orthophotos) into fixed-size training patches, preserving the
   train/val/test split and A/B/label folder structure.

Usage
-----
Synthetic:
    python data/prepare_dataset.py --synthetic --out dataset_synth \
        --n_train 400 --n_val 80 --n_test 80 --img_size 256

Patchify real data:
    python data/prepare_dataset.py --patchify --src dataset --dst dataset_patched \
        --patch_size 256
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #

def _random_terrain(size, seed):
    """Cheap procedural 'satellite-like' background: layered noise + smoothing."""
    rng = np.random.RandomState(seed)
    base = rng.randint(60, 180, (size, size, 3), dtype=np.uint8).astype(np.float32)
    # low-frequency component for large land-cover regions
    low = rng.randint(40, 200, (size // 16 + 1, size // 16 + 1, 3)).astype(np.float32)
    low_img = Image.fromarray(low.astype(np.uint8)).resize((size, size), Image.BICUBIC)
    low = np.array(low_img).astype(np.float32)
    terrain = 0.35 * base + 0.65 * low
    terrain = np.clip(terrain, 0, 255).astype(np.uint8)
    return Image.fromarray(terrain)


def _draw_structures(img, structures, color=(200, 200, 195)):
    """Draw rectangular 'buildings' / linear 'roads' onto an image in place."""
    draw = ImageDraw.Draw(img)
    for s in structures:
        kind, coords = s
        if kind == "building":
            draw.rectangle(coords, fill=color, outline=(90, 90, 90))
        elif kind == "road":
            draw.line(coords, fill=(110, 108, 100), width=6)
    return img


def _random_structures(size, n, rng, kind="building"):
    out = []
    for _ in range(n):
        if kind == "building":
            w, h = rng.randint(15, 45), rng.randint(15, 45)
            x0 = rng.randint(0, size - w)
            y0 = rng.randint(0, size - h)
            out.append(("building", (x0, y0, x0 + w, y0 + h)))
        else:
            x0, y0 = rng.randint(0, size), rng.randint(0, size)
            x1, y1 = rng.randint(0, size), rng.randint(0, size)
            out.append(("road", (x0, y0, x1, y1)))
    return out


def generate_pair(size, seed):
    """Generate one (before, after, mask) triplet with plausible changes."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    terrain = _random_terrain(size, seed)
    before = terrain.copy()
    after = terrain.copy()

    # Persistent structures present in both images
    persistent = _random_structures(size, np_rng.randint(3, 8), np_rng, "building")
    before = _draw_structures(before, persistent)
    after = _draw_structures(after, persistent)

    # Changed structures: appear only in "after" (new construction),
    # or only in "before" (demolition) -> both count as "change"
    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)

    n_new = np_rng.randint(1, 5)
    new_structs = _random_structures(size, n_new, np_rng, "building")
    after = _draw_structures(after, new_structs, color=(225, 60, 40))  # freshly built look
    for _, coords in new_structs:
        mdraw.rectangle(coords, fill=255)

    n_removed = np_rng.randint(0, 3)
    removed_structs = _random_structures(size, n_removed, np_rng, "building")
    before = _draw_structures(before, removed_structs, color=(170, 170, 160))
    for _, coords in removed_structs:
        mdraw.rectangle(coords, fill=255)

    # A road appears (e.g. new infrastructure)
    if np_rng.rand() > 0.5:
        road = _random_structures(size, 1, np_rng, "road")
        after = _draw_structures(after, road)
        x0, y0, x1, y1 = road[0][1]
        mdraw.line((x0, y0, x1, y1), fill=255, width=8)

    # mild sensor/illumination noise so before != after even where unchanged
    before_arr = np.array(before).astype(np.int16)
    after_arr = np.array(after).astype(np.int16)
    before_arr = np.clip(before_arr + np_rng.randint(-8, 8, before_arr.shape), 0, 255)
    after_arr = np.clip(after_arr + np_rng.randint(-8, 8, after_arr.shape), 0, 255)

    return (Image.fromarray(before_arr.astype(np.uint8)),
            Image.fromarray(after_arr.astype(np.uint8)),
            mask)


def make_synthetic_split(out_root, split, n, img_size, seed_offset):
    for sub in ["A", "B", "label"]:
        os.makedirs(os.path.join(out_root, split, sub), exist_ok=True)
    for i in range(n):
        before, after, mask = generate_pair(img_size, seed_offset + i)
        name = f"{split}_{i:05d}.png"
        before.save(os.path.join(out_root, split, "A", name))
        after.save(os.path.join(out_root, split, "B", name))
        mask.save(os.path.join(out_root, split, "label", name))
    print(f"[synthetic] wrote {n} pairs to {out_root}/{split}")


def run_synthetic(args):
    make_synthetic_split(args.out, "train", args.n_train, args.img_size, seed_offset=0)
    make_synthetic_split(args.out, "val", args.n_val, args.img_size, seed_offset=100000)
    make_synthetic_split(args.out, "test", args.n_test, args.img_size, seed_offset=200000)
    print("Done. Point configs/config.yaml -> data_root at:", args.out)


# --------------------------------------------------------------------------- #
# Patchify real data
# --------------------------------------------------------------------------- #

def patchify_image(img: Image.Image, patch_size, stride=None):
    stride = stride or patch_size
    w, h = img.size
    patches, coords = [], []
    for y in range(0, max(h - patch_size + 1, 1), stride):
        for x in range(0, max(w - patch_size + 1, 1), stride):
            box = (x, y, x + patch_size, y + patch_size)
            patch = img.crop(box)
            if patch.size != (patch_size, patch_size):
                # pad if the image doesn't divide evenly
                canvas = Image.new(img.mode, (patch_size, patch_size))
                canvas.paste(patch, (0, 0))
                patch = canvas
            patches.append(patch)
            coords.append((x, y))
    return patches, coords


def run_patchify(args):
    src, dst, ps = Path(args.src), Path(args.dst), args.patch_size
    for split in ["train", "val", "test"]:
        a_dir, b_dir, l_dir = src / split / "A", src / split / "B", src / split / "label"
        if not a_dir.exists():
            print(f"[patchify] skipping missing split: {split}")
            continue
        out_a, out_b, out_l = dst / split / "A", dst / split / "B", dst / split / "label"
        for d in (out_a, out_b, out_l):
            d.mkdir(parents=True, exist_ok=True)

        names = sorted(os.listdir(a_dir))
        count = 0
        for name in names:
            img_a = Image.open(a_dir / name).convert("RGB")
            img_b = Image.open(b_dir / name).convert("RGB")
            img_l = Image.open(l_dir / name).convert("L")

            patches_a, coords = patchify_image(img_a, ps)
            patches_b, _ = patchify_image(img_b, ps)
            patches_l, _ = patchify_image(img_l, ps)

            stem = Path(name).stem
            for i, (pa, pb, pl) in enumerate(zip(patches_a, patches_b, patches_l)):
                out_name = f"{stem}_{i:03d}.png"
                pa.save(out_a / out_name)
                pb.save(out_b / out_name)
                pl.save(out_l / out_name)
                count += 1
        print(f"[patchify] {split}: {len(names)} images -> {count} patches")
    print("Done. Point configs/config.yaml -> data_root at:", args.dst)


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--patchify", action="store_true")

    # synthetic args
    p.add_argument("--out", default="dataset_synth")
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--n_val", type=int, default=80)
    p.add_argument("--n_test", type=int, default=80)
    p.add_argument("--img_size", type=int, default=256)

    # patchify args
    p.add_argument("--src", default="dataset")
    p.add_argument("--dst", default="dataset_patched")
    p.add_argument("--patch_size", type=int, default=256)

    args = p.parse_args()

    if args.synthetic:
        run_synthetic(args)
    elif args.patchify:
        run_patchify(args)
    else:
        print("Specify --synthetic or --patchify. See --help.")


if __name__ == "__main__":
    main()
