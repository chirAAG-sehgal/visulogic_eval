"""
TRM Training Script.

Trains a Tiny Recursion Model on pre-extracted Qwen3-VL hidden states
for VisuLogic MCQ classification.

Usage:
    python -m trm.train --model_name Qwen3-VL-2B-Instruct --gpu 0
    python -m trm.train --model_name Qwen3-VL-4B-Instruct --gpu 1
"""

import sys
sys.path.append(".")
import argparse
import os

# Parse GPU early
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=0)
_early, _ = _parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_early.gpu)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

import mlflow

from trm.model import TRM
from trm.dataset import HiddenStateDataset, collate_fn
from trm.utils import EMA, TextLogger, compute_accuracy, compute_per_tag_accuracy, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train TRM on hidden states")

    # Data paths
    parser.add_argument('--model_name', type=str, default='Qwen3-VL-2B-Instruct',
                        help='Model name folder under hidden_states/')
    parser.add_argument('--train_hidden_dir', type=str, default=None,
                        help='Override train hidden states dir')
    parser.add_argument('--val_hidden_dir', type=str, default=None,
                        help='Override val hidden states dir')
    parser.add_argument('--train_labels', type=str, default='data/visulogic_train/visulogic_train_qwen.jsonl')
    parser.add_argument('--val_labels', type=str, default='data/visulogic_benchmark/data.jsonl')
    parser.add_argument('--train_label_key', type=str, default='answer')
    parser.add_argument('--val_label_key', type=str, default='label')

    # Architecture
    parser.add_argument('--dim', type=int, default=2048)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--n_classes', type=int, default=4)

    # Recursion
    parser.add_argument('--n_latent_steps', type=int, default=6)
    parser.add_argument('--n_deep_passes', type=int, default=3)
    parser.add_argument('--n_sup_steps', type=int, default=8)

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_embed', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Logging
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/trm')
    parser.add_argument('--mlflow_experiment', type=str, default='TRM-VisuLogic')

    args = parser.parse_args()

    # Set default paths based on model_name
    if args.train_hidden_dir is None:
        args.train_hidden_dir = f'outputs/hidden_states/{args.model_name}'
    if args.val_hidden_dir is None:
        args.val_hidden_dir = f'outputs/hidden_states/val/{args.model_name}'

    return args


def train_one_epoch(model, loader, optimizer, scheduler, ema, device, args):
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_halt_loss = 0.0
    total_correct = 0
    total_samples = 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Train", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for step, batch in enumerate(pbar):
        x = batch['x'].to(device)
        z = batch['z'].to(device)
        x_mask = batch['x_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(x, z, labels=labels, x_mask=x_mask, n_sup_steps=args.n_sup_steps)

        loss = outputs['loss'] / args.grad_accum_steps
        loss.backward()

        if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            ema.update(model)

        B = labels.shape[0]
        total_loss += outputs['loss'].item() * B
        total_pred_loss += outputs['pred_loss'].item() * B
        total_halt_loss += outputs['halt_loss'].item() * B
        total_correct += (outputs['logits'].argmax(-1) == labels).sum().item()
        total_samples += B

        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    n = total_samples
    return {
        'loss': total_loss / n,
        'pred_loss': total_pred_loss / n,
        'halt_loss': total_halt_loss / n,
        'acc': total_correct / n,
    }


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_halt_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_tags = []

    pbar = tqdm(loader, desc="Val", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for batch in pbar:
        x = batch['x'].to(device)
        z = batch['z'].to(device)
        x_mask = batch['x_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(x, z, labels=labels, x_mask=x_mask, n_sup_steps=args.n_sup_steps)

        B = labels.shape[0]
        total_loss += outputs['loss'].item() * B
        total_pred_loss += outputs['pred_loss'].item() * B
        total_halt_loss += outputs['halt_loss'].item() * B

        preds = outputs['logits'].argmax(-1)
        total_correct += (preds == labels).sum().item()
        total_samples += B

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_tags.extend(batch['tags'])

    n = total_samples
    tag_acc = compute_per_tag_accuracy(all_preds, all_labels, all_tags)

    return {
        'loss': total_loss / n,
        'pred_loss': total_pred_loss / n,
        'halt_loss': total_halt_loss / n,
        'acc': total_correct / n,
        'tag_acc': tag_acc,
    }


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets
    print("Loading datasets...")
    train_dataset = HiddenStateDataset(
        args.train_hidden_dir, args.train_labels, label_key=args.train_label_key)
    val_dataset = HiddenStateDataset(
        args.val_hidden_dir, args.val_labels, label_key=args.val_label_key)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    # Model
    print("Building TRM...")
    model = TRM(
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        n_classes=args.n_classes,
        n_latent_steps=args.n_latent_steps,
        n_deep_passes=args.n_deep_passes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TRM parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Optimizer — separate LR for y_init embedding
    embed_params = [model.y_init]
    net_params = [p for name, p in model.named_parameters() if 'y_init' not in name]
    optimizer = torch.optim.AdamW([
        {'params': net_params, 'lr': args.lr},
        {'params': embed_params, 'lr': args.lr_embed},
    ], betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # Scheduler: linear warmup + cosine decay
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_steps])

    # EMA
    ema = EMA(model, decay=args.ema_decay)

    # Logging
    logger = TextLogger(args.log_dir, args.model_name)
    logger.log(f"Config: {vars(args)}")
    logger.log(f"TRM parameters: {n_params:,}")
    logger.log(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # MLflow
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.start_run(run_name=f"TRM-{args.model_name}")
    mlflow.log_params({k: v for k, v in vars(args).items() if not k.startswith('_')})
    mlflow.log_param("n_params", n_params)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    ckpt_dir = os.path.join(args.ckpt_dir, args.model_name)

    for epoch in range(1, args.epochs + 1):
        logger.log(f"--- Epoch {epoch}/{args.epochs} ---")

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, ema, device, args)

        # Validate with EMA weights
        val_metrics = validate(ema.shadow, val_loader, device, args)
        tag_acc = val_metrics.pop('tag_acc')

        # Log
        logger.log_epoch(epoch, train_metrics, val_metrics)
        for tag, acc in tag_acc.items():
            logger.log(f"  {tag}: {acc:.4f}")

        # MLflow logging
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)
        for tag, acc in tag_acc.items():
            safe_tag = tag.replace(" ", "_").lower()
            mlflow.log_metric(f"val_tag_{safe_tag}", acc, step=epoch)
        mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=epoch)

        # Checkpointing
        save_checkpoint(model, ema, optimizer, epoch, val_metrics['acc'],
                        os.path.join(ckpt_dir, 'latest.pt'))

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            patience_counter = 0
            save_checkpoint(model, ema, optimizer, epoch, best_val_acc,
                            os.path.join(ckpt_dir, 'best.pt'))
            logger.log(f"  New best val acc: {best_val_acc:.4f}")
            mlflow.log_metric("best_val_acc", best_val_acc, step=epoch)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.log(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    logger.log(f"Training complete. Best val acc: {best_val_acc:.4f}")
    mlflow.log_metric("final_best_val_acc", best_val_acc)
    mlflow.end_run()


if __name__ == '__main__':
    main()
