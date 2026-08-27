"""utils/metrics.py -- standard binary change-detection metrics."""
import numpy as np
import torch


class ConfusionMeter:
    """Accumulates a binary confusion matrix (TP/FP/FN/TN) across batches."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits, targets):
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()
            targets = targets.float()

            self.tp += ((preds == 1) & (targets == 1)).sum().item()
            self.fp += ((preds == 1) & (targets == 0)).sum().item()
            self.fn += ((preds == 0) & (targets == 1)).sum().item()
            self.tn += ((preds == 0) & (targets == 0)).sum().item()

    def compute(self):
        eps = 1e-7
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        oa = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        changed_pct = 100.0 * (self.tp + self.fn) / (self.tp + self.fp + self.fn + self.tn + eps)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "overall_accuracy": oa,
            "gt_changed_pct": changed_pct,
        }

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0


def changed_area_stats(pred_mask: np.ndarray):
    """
    pred_mask: 2D binary array (0/1) of predicted change.
    Returns dict with changed_pct / unchanged_pct, matching the
    'Changed Area: 14.7% / Unchanged Area: 85.3%' style summary.
    """
    total = pred_mask.size
    changed = int(pred_mask.sum())
    changed_pct = 100.0 * changed / total
    return {
        "changed_pixels": changed,
        "total_pixels": total,
        "changed_pct": round(changed_pct, 2),
        "unchanged_pct": round(100.0 - changed_pct, 2),
    }
