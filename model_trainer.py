import os

import matplotlib.pyplot as plt
import seaborn as sns

from utils import parse_args_with_defaults, set_seed

# ============================================================
# BLOCK 1: CONFIGURATION & SETUP
# ============================================================
def setup():
    """Parse arguments and set seed."""
    args = parse_args_with_defaults()
    set_seed(args.seed)

    # Create results directories
    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs(os.path.join(args.reports_dir, "part_1_results"), exist_ok=True)
    os.makedirs(os.path.join(args.reports_dir, "metrics"), exist_ok=True)

    return args


# ============================================================
# BLOCK 2: DATA LOADING
# ============================================================
def load_data(args):
    """Create dataloaders and get dataset info."""
    from data_loader import make_dataloaders

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
    """Create model and optimizer."""
    import torch.nn as nn
    from torch.optim import Adam

    from model_factory import make_model

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
# BLOCK 4: TRAINING LOOP
# ============================================================
def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train one epoch and return metrics."""
    import time

    model.train()
    epoch_start = time.time()
    batch_times = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start = time.time()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
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
# BLOCK 5: VALIDATION LOOP
# ============================================================
def validate(model, val_loader, device):
    """Validate model and return accuracy."""
    import time

    import torch

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
# BLOCK 6: METRICS & CONFUSION MATRIX
# ============================================================
def get_confusion_metrics(all_labels, all_preds, info):
    """Compute classification metrics and confusion matrix."""
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    # 1. Per-class + global metrics
    report = classification_report(
        all_labels,
        all_preds,
        target_names=info.class_names,
        output_dict=True,
    )

    macro_f1 = report["macro avg"]["f1-score"]
    weighted_acc = report["weighted avg"]["precision"]

    # 2. Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # 3. Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
    )

    return macro_f1, weighted_acc, report, cm, precision, recall, f1, support


def save_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved: {save_path}")


# ============================================================
# BLOCK 7: REPORTING
# ============================================================
def print_epoch_report(epoch, epoch_time, avg_batch, val_time, val_acc, device):
    """Print epoch summary."""
    import torch

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


def print_final_metrics(report, cm, class_names, model_arch, dataset):
    """Print final evaluation summary."""
    print("\n" + "=" * 60)
    print("FINAL EVALUATION METRICS")
    print("=" * 60)

    print(f"\nModel: {model_arch}")
    print(f"Dataset: {dataset}")
    print(f"\nMacro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Accuracy: {report['weighted avg']['precision']:.4f}")

    print("\nPer-class metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 60)
    for i, name in enumerate(class_names):
        print(
            f"{name:<15} {report[name]['precision']:<12.4f} "
            f"{report[name]['recall']:<12.4f} {report[name]['f1-score']:<12.4f} "
            f"{int(report[name]['support']):<10}"
        )
    print("=" * 60 + "\n")


def save_summary_report(report, class_names, model_arch, dataset, save_path):
    """
    Save summary report with class-wise and global metrics.
    Each experiment appends a block with:
    - Header line: Experiment: {model_arch} | Dataset: {dataset}
    - Class-wise metrics table (pandas-style)
    - Global aggregates table
    - Empty line between experiments
    """
    import pandas as pd

    # Prepare class-wise data
    class_data = []
    for name in class_names:
        class_data.append(
            {
                "Class": name,
                "Precision": report[name]["precision"],
                "Recall": report[name]["recall"],
                "F1-Score": report[name]["f1-score"],
                "Support": int(report[name]["support"]),
            }
        )

    class_df = pd.DataFrame(class_data)

    # Prepare global aggregates
    global_data = {
        "Metric": ["Macro F1", "Weighted Precision", "Weighted Recall", "Weighted F1"],
        "Score": [
            report["macro avg"]["f1-score"],
            report["weighted avg"]["precision"],
            report["weighted avg"]["recall"],
            report["weighted avg"]["f1-score"],
        ],
    }
    global_df = pd.DataFrame(global_data)

    # Write to file (append mode)
    with open(save_path, "a") as f:
        # Header
        f.write(f"Experiment: {model_arch} | Dataset: {dataset}\n")
        f.write("-" * 60 + "\n")

        # Class-wise metrics
        f.write("\nClass-wise Metrics:\n")
        f.write(class_df.to_string(index=False))
        f.write("\n\n")

        # Global aggregates
        f.write("Global Metrics:\n")
        f.write(global_df.to_string(index=False))
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("\n")  # Empty line between experiments

    print(f"✅ Summary report appended: {save_path}")


# ============================================================
# BLOCK 8: MAIN
# ============================================================
def main():
    import torch

    # Setup
    args = setup()

    # Print config
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    # Load data
    train_loader, val_loader, info = load_data(args)

    # Create model
    model, optimizer, criterion = create_model(args, info.num_classes)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}\n")

    # Training loop
    for epoch in range(args.epochs):
        # Train
        epoch_time, avg_batch = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        all_preds, all_labels, val_time, val_acc = validate(model, val_loader, device)

        # Compute metrics
        macro_f1, weighted_acc, report, cm, precision, recall, f1, support = (
            get_confusion_metrics(all_labels, all_preds, info)
        )

        # Report
        print_epoch_report(epoch, epoch_time, avg_batch, val_time, val_acc, device)

    # ============================================================
    # FINAL EVALUATION & SAVE ARTIFACTS
    # ============================================================

    # 1. Print final metrics
    print_final_metrics(report, cm, info.class_names, args.arch, args.dataset)

    # 2. Save confusion matrix
    cm_filename = os.path.join(
        args.reports_dir, "part_1_results", f"cm_{args.arch}_{args.dataset}.png"
    )
    save_confusion_matrix(
        cm,
        info.class_names,
        cm_filename,
        title=f"Confusion Matrix - {args.arch} on {args.dataset}",
    )

    # 3. Save summary report (appends to file)
    summary_filename = os.path.join(args.reports_dir, "part_1_results", "summary.txt")
    save_summary_report(
        report,
        info.class_names,
        args.arch,
        args.dataset,
        summary_filename,
    )

    # 4. Save model checkpoint
    checkpoint_dir = os.path.join(args.reports_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{args.arch}_{args.dataset}_best.pth"
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
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"All artifacts saved to: {args.reports_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
