"""
Utility functions: EMA, metrics, logging helpers.
"""

import copy
import os
import datetime
import torch


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


class TextLogger:
    """Simple text file logger for remote access."""

    def __init__(self, log_dir, model_name):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"trm_{model_name}_{timestamp}.txt")
        # Write header
        with open(self.path, 'w') as f:
            f.write(f"TRM Training Log — {model_name} — {timestamp}\n")
            f.write("=" * 80 + "\n")

    def log(self, msg):
        with open(self.path, 'a') as f:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            f.write(f"[{ts}] {msg}\n")
        print(msg)

    def log_epoch(self, epoch, train_metrics, val_metrics):
        line_parts = [f"Epoch {epoch:03d}"]
        for k, v in train_metrics.items():
            line_parts.append(f"train_{k}={v:.4f}")
        for k, v in val_metrics.items():
            line_parts.append(f"val_{k}={v:.4f}")
        self.log(" | ".join(line_parts))


def compute_accuracy(logits, labels):
    """Compute accuracy from logits and labels."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def compute_per_tag_accuracy(all_preds, all_labels, all_tags):
    """Compute accuracy per reasoning tag."""
    tag_correct = {}
    tag_total = {}
    for pred, label, tag in zip(all_preds, all_labels, all_tags):
        if tag not in tag_correct:
            tag_correct[tag] = 0
            tag_total[tag] = 0
        tag_total[tag] += 1
        if pred == label:
            tag_correct[tag] += 1

    tag_acc = {}
    for tag in sorted(tag_correct.keys()):
        tag_acc[tag] = tag_correct[tag] / tag_total[tag] if tag_total[tag] > 0 else 0.0
    return tag_acc


def save_checkpoint(model, ema, optimizer, epoch, val_acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, path)
