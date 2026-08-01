from utils import parse_args_with_defaults, set_seed

if __name__ == "__main__":
    # Parse args with YAML defaults
    args = parse_args_with_defaults()

    # Access arguments
    print(f"Architecture: {args.arch}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    set_seed(args.seed)

    import time

    import torch
    import torch.nn as nn
    from torch.optim import Adam

    from data_loader import make_dataloaders
    from model_factory import make_model

    train_loader, val_loader, info = make_dataloaders(
        dataset_name=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        model_arch=args.arch,
    )
    print(f"Created dataloaders with {info.num_classes} classes")
    print(f"Batch shape: {next(iter(train_loader))[0].shape}")

    model = make_model(
        arch=args.arch,
        in_channels=3,
        num_classes=info.num_classes,
        pretrained=args.pre_trained,
        freeze_backbone=args.freeze_backbone,
    )
    print(f"Selected model:\n{args.arch} with pretrained={args.pre_trained}")
    model_summary = str(model)
    print(f"Created model:\n{model_summary}")

    if args.pre_trained and args.freeze_backbone:
        update_params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        update_params = model.parameters()

    optimizer = Adam(params=update_params, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # training loop block
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for epoch in range(args.epochs):
        # ============================================
        # TRAINING PHASE - Timed
        # ============================================
        model.train()

        epoch_start = time.time()
        batch_times = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()

            # 1. Data transfer to GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 2. Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 3. Backward pass
            loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Print every 10 batches
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: total={batch_time * 1000:.1f}ms, ")

        epoch_time = time.time() - epoch_start
        avg_batch = sum(batch_times) / len(batch_times)

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
        print(f"  Val time: {val_time:.2f}s")
        print(f"  Val Acc: {val_acc:.2f}%")

        # GPU memory usage
        if device == "cuda":
            print(
                f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
            )
        print(f"{'=' * 60}\n")
