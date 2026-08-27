"""utils/visualize.py -- plotting helpers for report figures."""
import matplotlib.pyplot as plt
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(img_tensor):
    """img_tensor: (3,H,W) normalized tensor -> (H,W,3) uint8 numpy."""
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def save_comparison_figure(img_a, img_b, gt_mask, pred_mask, save_path, changed_pct=None):
    """
    img_a, img_b: (3,H,W) normalized tensors (before/after)
    gt_mask: (H,W) numpy 0/1 array or None
    pred_mask: (H,W) numpy 0/1 array
    """
    a = denormalize(img_a)
    b = denormalize(img_b)

    n_panels = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))

    axes[0].imshow(a); axes[0].set_title("Before (T1)"); axes[0].axis("off")
    axes[1].imshow(b); axes[1].set_title("After (T2)"); axes[1].axis("off")

    idx = 2
    if gt_mask is not None:
        axes[idx].imshow(gt_mask, cmap="gray")
        axes[idx].set_title("Ground Truth Change")
        axes[idx].axis("off")
        idx += 1

    overlay = b.copy()
    overlay[pred_mask.astype(bool)] = [255, 0, 0]
    blended = (0.6 * b + 0.4 * overlay).astype(np.uint8)
    title = "Predicted Change"
    if changed_pct is not None:
        title += f"\n(Changed: {changed_pct:.1f}%)"
    axes[idx].imshow(blended)
    axes[idx].set_title(title)
    axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history, save_path):
    """history: dict with lists 'train_loss','val_loss','val_f1','val_iou'."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train loss")
    axes[0].plot(history["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(history["val_f1"], label="val F1")
    axes[1].plot(history["val_iou"], label="val IoU")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score"); axes[1].legend()
    axes[1].set_title("Validation Metrics")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
