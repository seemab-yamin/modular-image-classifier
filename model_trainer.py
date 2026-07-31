import argparse
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from data_loader import make_dataloaders
from model_factory import make_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, help="Model architecture")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--batch-size", type=int, help="Batch size for dataloaders")
    parser.add_argument(
        "--learning-rate", type=float, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for dataloaders"
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained weights"
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the backbone of the model",
    )
    parser.add_argument(
        "--root-dir", type=str, default="./data", help="Root directory for the dataset"
    )
    args = parser.parse_args()

    train_loader, val_loader, info = make_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root_dir=args.root_dir,
    )
    print(f"Created dataloaders with {info.num_classes} classes")
    print(f"Batch shape: {next(iter(train_loader))[0].shape}")

    model = make_model(
        arch=args.arch,
        in_channels=3,
        num_classes=info.num_classes,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    print(f"Selected model:\n{args.arch} with pretrained={args.pretrained}")
    model_summary = str(model)
    print(f"Created model:\n{model_summary}")

    if args.pretrained and args.freeze_backbone:
        update_params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        update_params = model.parameters()

    optimizer = Adam(params=update_params, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # training loop block
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for epoch in range(10):
        # ============================================
        # TRAINING PHASE - Timed
        # ============================================
        model.train()

        epoch_start = time.time()
        batch_times = []
        forward_times = []
        backward_times = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()

            # 1. Data transfer to GPU
            transfer_start = time.time()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            transfer_time = time.time() - transfer_start

            # 2. Forward pass
            forward_start = time.time()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            forward_time = time.time() - forward_start

            # 3. Backward pass
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            forward_times.append(forward_time)
            backward_times.append(backward_time)

            # Print every 10 batches
            if batch_idx % 10 == 0:
                print(
                    f"  Batch {batch_idx}: total={batch_time * 1000:.1f}ms, "
                    f"transfer={transfer_time * 1000:.1f}ms, "
                    f"forward={forward_time * 1000:.1f}ms, "
                    f"backward={backward_time * 1000:.1f}ms"
                )

        epoch_time = time.time() - epoch_start
        avg_batch = sum(batch_times) / len(batch_times)
        avg_forward = sum(forward_times) / len(forward_times)
        avg_backward = sum(backward_times) / len(backward_times)

        # ============================================
        # VALIDATION PHASE - Timed
        # ============================================
        model.eval()
        correct, total = 0, 0
        val_start = time.time()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_time = time.time() - val_start
        val_acc = 100 * correct / total

        # ============================================
        # REPORT
        # ============================================
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}:")
        print(f"  Train time: {epoch_time:.2f}s ({avg_batch * 1000:.1f}ms/batch)")
        print(f"    - Forward: {avg_forward * 1000:.1f}ms/batch")
        print(f"    - Backward: {avg_backward * 1000:.1f}ms/batch")
        print(
            f"    - Transfer: {(avg_batch - avg_forward - avg_backward) * 1000:.1f}ms/batch"
        )
        print(f"  Val time: {val_time:.2f}s")
        print(f"  Val Acc: {val_acc:.2f}%")

        # GPU memory usage
        if device == "cuda":
            print(
                f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
            )
        print(f"{'=' * 60}\n")
