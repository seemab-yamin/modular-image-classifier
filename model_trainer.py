"""
Modular Image Classifier - Training Script
"""

import os

import torch

from data_loader import make_dataloaders
from eval import evaluate, save_confusion_matrix, save_summary_report
from model_factory import make_model
from utils import parse_args_with_defaults, set_seed


# ============================================================
# BLOCK 1: CONFIGURATION & SETUP
# ============================================================
def setup():
    """Parse arguments, set seed, and create directories."""

    args = parse_args_with_defaults()
    set_seed(args.seed)

    # Create results directories
    os.makedirs(args.artifacts_dir, exist_ok=True)
    os.makedirs(os.path.join(args.artifacts_dir, "part_1_results"), exist_ok=True)
    os.makedirs(os.path.join(args.artifacts_dir, "checkpoints"), exist_ok=True)
    return args


def print_config(args):
    """Print configuration."""
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")


# ============================================================
# BLOCK 2: DATA LOADING
# ============================================================
def load_data(args):
    """Create dataloaders and get dataset info."""

    train_loader, val_loader, info = make_dataloaders(
        dataset_name=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        resize_size=(224, 224) if args.arch == "vit" else None,
    )

    print(f"Created dataloaders with {info.num_classes} classes")
    print(f"Batch shape: {next(iter(train_loader))[0].shape}")

    return train_loader, val_loader, info


# ============================================================
# BLOCK 3: MODEL CREATION
# ============================================================
def create_model(args, num_classes):
    """Create model, optimizer, and criterion."""

    import torch.nn as nn
    from torch.optim import Adam

    model = make_model(
        arch=args.arch,
        in_channels=3,
        num_classes=num_classes,
        pretrained=args.pre_trained,
        freeze_backbone=args.freeze_backbone,
    )

    print(f"Selected model: {args.arch} with pretrained={args.pre_trained}")
    print(f"Created model:\n{str(model)}")

    # Select trainable parameters
    if args.pre_trained and args.freeze_backbone:
        update_params = [p for p in model.parameters() if p.requires_grad]
    else:
        update_params = model.parameters()

    optimizer = Adam(params=update_params, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


# ============================================================
# BLOCK 4: TRAINING
# ============================================================
def train_epoch(model, train_loader, optimizer, criterion, device, use_amp):
    """Train one epoch and return metrics."""
    import time

    if use_amp:
        from torch.cuda.amp import GradScaler, autocast

        # 1. Create scaler
        scaler = GradScaler()

    model.train()
    epoch_start = time.time()
    batch_times = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start = time.time()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # forward pass with mixed precision if enabled
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}: total={batch_time * 1000:.1f}ms")

    epoch_time = time.time() - epoch_start
    avg_batch = sum(batch_times) / len(batch_times)

    return epoch_time, avg_batch


# ============================================================
# BLOCK 5: VALIDATION
# ============================================================
def validate(model, val_loader, device):
    """Validate model and return predictions and labels."""

    import time

    model.eval()
    val_start = time.time()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_time = time.time() - val_start
    val_acc = (
        100
        * sum(1 for pred, label in zip(all_preds, all_labels) if pred == label)
        / len(all_labels)
    )

    return all_preds, all_labels, val_time, val_acc


# ============================================================
# BLOCK 6: REPORTING
# ============================================================
def print_epoch_report(epoch, epoch_time, avg_batch, val_time, val_acc, device):
    """Print epoch summary."""

    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch + 1}:")
    print(f"  Train time: {epoch_time:.2f}s ({avg_batch * 1000:.1f}ms/batch)")
    print(f"  Val time: {val_time:.2f}s")
    print(f"  Val Acc: {val_acc:.2f}%")

    if device == "cuda":
        print(
            f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / "
            f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
        )
    print(f"{'=' * 60}\n")


# ============================================================
# BLOCK 7: MAIN
# ============================================================
def main():
    import torch

    # 1. Setup
    args = setup()
    print_config(args)

    # 2. Load data
    train_loader, val_loader, info = load_data(args)

    # 3. Create model
    model, optimizer, criterion = create_model(args, info.num_classes)

    # 4. Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}\n")

    # 5. Training loop
    for epoch in range(args.epochs):
        epoch_time, avg_batch = train_epoch(
            model, train_loader, optimizer, criterion, device, args.use_amp
        )
        all_preds, all_labels, val_time, val_acc = validate(model, val_loader, device)
        print_epoch_report(epoch, epoch_time, avg_batch, val_time, val_acc, device)

    # 6. Final evaluation
    metrics = evaluate(all_labels, all_preds, info.class_names)

    # 7. Save artifacts
    cm_path = os.path.join(
        args.artifacts_dir, "part_1_results", f"cm_{args.arch}_MODEL_{args.dataset}.png"
    )
    save_confusion_matrix(
        metrics["confusion_matrix"],
        info.class_names,
        cm_path,
        title=f"Confusion Matrix - {args.arch} Model on {args.dataset}",
    )

    summary_path = os.path.join(args.artifacts_dir, "part_1_results", "summary.txt")
    save_summary_report(
        metrics["report"],
        info.class_names,
        args.arch,
        args.dataset,
        summary_path,
    )

    # 8. Save checkpoint
    checkpoint_path = os.path.join(
        args.artifacts_dir, "checkpoints", f"{args.arch}_MODEL_{args.dataset}_best.pth"
    )
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"✅ Checkpoint saved: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation Accuracy: {val_acc:.2f}%")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"All artifacts saved to: {args.artifacts_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
