"""
data/dataset.py

PyTorch Dataset for bi-temporal change-detection data in LEVIR-CD layout:

    data_root/{split}/A/*.png      (before image)
    data_root/{split}/B/*.png      (after image)
    data_root/{split}/label/*.png  (binary mask, 0=unchanged 255=changed)

A/, B/, label/ must contain files with matching filenames.
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBU = True
except ImportError:
    _HAS_ALBU = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size, train: bool):
    """Shared augmentation applied identically to A and B (and label) via
    albumentations' `additional_targets` so geometric transforms stay aligned."""
    if not _HAS_ALBU:
        return None
    if train:
        return A.Compose(
            [
                A.RandomCrop(img_size, img_size) if img_size else A.NoOp(),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02, p=0.5),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            additional_targets={"image_b": "image", "mask": "mask"},
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            additional_targets={"image_b": "image", "mask": "mask"},
        )


class ChangeDetectionDataset(Dataset):
    def __init__(self, data_root, split="train", img_size=256, train=None):
        self.root = Path(data_root) / split
        self.a_dir = self.root / "A"
        self.b_dir = self.root / "B"
        self.l_dir = self.root / "label"
        assert self.a_dir.exists(), f"Missing folder: {self.a_dir}"

        self.names = sorted(os.listdir(self.a_dir))
        self.train = (split == "train") if train is None else train
        self.transform = build_transforms(img_size, self.train)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_a = np.array(Image.open(self.a_dir / name).convert("RGB"))
        img_b = np.array(Image.open(self.b_dir / name).convert("RGB"))
        mask = np.array(Image.open(self.l_dir / name).convert("L"))
        mask = (mask > 127).astype(np.float32)  # binarize 0/255 -> 0/1

        if self.transform is not None:
            out = self.transform(image=img_a, image_b=img_b, mask=mask)
            img_a, img_b, mask = out["image"], out["image_b"], out["mask"]
            mask = mask.unsqueeze(0).float()
        else:
            # fallback if albumentations isn't installed
            img_a = torch.from_numpy(img_a / 255.0).permute(2, 0, 1).float()
            img_b = torch.from_numpy(img_b / 255.0).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {"A": img_a, "B": img_b, "mask": mask, "name": name}


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset_synth"
    ds = ChangeDetectionDataset(root, split="train", img_size=256)
    print(f"Loaded {len(ds)} training pairs from {root}")
    sample = ds[0]
    print("A:", sample["A"].shape, "B:", sample["B"].shape, "mask:", sample["mask"].shape,
          "mask unique:", torch.unique(sample["mask"]))
