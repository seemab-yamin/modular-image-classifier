import argparse

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR100_CLASSES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
]
STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


class DatasetInfo:
    """Container for dataset metadata."""

    def __init__(self, num_classes, input_shape, mean, std, class_names):
        self.num_classes = num_classes
        self.input_shape = input_shape  # (C, H, W)
        self.mean = mean  # for normalization
        self.std = std  # for normalization
        self.class_names = class_names


def make_dataloaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    root_dir: str = "./data",
    augment: bool = True,
):
    """
    dataset factory to load datasets
    """
    dataset_name = dataset_name.lower()

    # Set dataset-specific parameters
    if dataset_name in ["cifar10", "cifar100"]:
        input_shape = (3, 32, 32)
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif dataset_name == "stl10":
        input_shape = (3, 96, 96)
        mean, std = (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # build transforms
    train_transforms = [transforms.Resize(input_shape[1:]), transforms.ToTensor()]
    if augment:
        train_transforms.insert(1, transforms.RandomHorizontalFlip())
    train_transform = transforms.Compose(
        [
            *train_transforms,
            transforms.Normalize(mean, std),
        ]
    )

    val_transforms = [transforms.Resize(input_shape[1:]), transforms.ToTensor()]
    val_transform = transforms.Compose(
        [
            *val_transforms,
            transforms.Normalize(mean, std),
        ]
    )

    # Dataset mapping
    dataset_map = {
        "cifar10": (datasets.CIFAR10, 10, CIFAR10_CLASSES),
        "cifar100": (datasets.CIFAR100, 100, CIFAR100_CLASSES),
        "stl10": (datasets.STL10, 10, STL10_CLASSES),
    }

    dataset_class, num_classes, class_names = dataset_map[dataset_name]
    # Build datasets
    train_dataset = dataset_class(
        root=root_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = dataset_class(
        root=root_dir, train=False, download=True, transform=val_transform
    )

    # Handle STL10 split naming
    if dataset_name == "stl10":
        val_dataset = dataset_class(
            root=root_dir, split="test", download=True, transform=val_transform
        )

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    info = DatasetInfo(num_classes, input_shape, mean, std, class_names)

    return train_loader, val_loader, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation")
    parser.add_argument(
        "--root-dir", type=str, default="./data", help="Root directory for the dataset"
    )
    args = parser.parse_args()

    train_loader, val_loader, info = make_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        root_dir=args.root_dir,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Classes: {info.num_classes}")
    print(f"Input shape: {info.input_shape}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
