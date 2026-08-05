import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import seed_worker

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
    "cra",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
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

# Dataset-specific configurations
DATASET_CONFIGS = {
    "cifar10": {
        "class": datasets.CIFAR10,
        "num_classes": 10,
        "class_names": CIFAR10_CLASSES,
        "input_shape": (3, 32, 32),
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "train_kwargs": {"train": True},
        "test_kwargs": {"train": False},
    },
    "cifar100": {
        "class": datasets.CIFAR100,
        "num_classes": 100,
        "class_names": CIFAR100_CLASSES,
        "input_shape": (3, 32, 32),
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "train_kwargs": {"train": True},
        "test_kwargs": {"train": False},
    },
    "stl10": {
        "class": datasets.STL10,
        "num_classes": 10,
        "class_names": STL10_CLASSES,
        "input_shape": (3, 96, 96),
        "mean": (0.4467, 0.4398, 0.4066),
        "std": (0.2603, 0.2566, 0.2713),
        "train_kwargs": {"split": "train"},
        "test_kwargs": {"split": "test"},
    },
}


class DatasetInfo:
    """Container for dataset metadata."""

    def __init__(self, num_classes, input_shape, mean, std, class_names):
        self.num_classes = num_classes
        self.input_shape = input_shape  # (C, H, W)
        self.mean = mean  # for normalization
        self.std = std  # for normalization
        self.class_names = class_names


def make_dataloaders(
    dataset_name: str = "cifar10",
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
    data_dir: str = "./data",
    augment: bool = True,
    resize_size: tuple = (),
):
    """
    dataset factory to load datasets
    """
    dataset_name = dataset_name.lower()

    # Get dataset config
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_config = DATASET_CONFIGS[dataset_name]
    input_shape = dataset_config["input_shape"]
    mean, std = dataset_config["mean"], dataset_config["std"]

    # build transforms
    train_transforms = [transforms.ToTensor()]
    if resize_size:
        train_transforms = [transforms.Resize(resize_size)] + train_transforms
        input_shape = (
            input_shape[0],
            resize_size[0],
            resize_size[1],
        )  # Update input shape if resized
    if augment:
        train_transforms = [transforms.RandomHorizontalFlip()] + train_transforms

    train_transform = transforms.Compose(
        [
            *train_transforms,
            transforms.Normalize(mean, std),
        ]
    )

    val_transforms = [transforms.ToTensor()]
    if resize_size:
        val_transforms = [transforms.Resize(resize_size)] + val_transforms
    val_transform = transforms.Compose(
        [
            *val_transforms,
            transforms.Normalize(mean, std),
        ]
    )

    dataset_class = dataset_config["class"]

    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    if dataset_name == "stl10":
        train_dataset_kwargs["split"] = "train"
        test_dataset_kwargs["split"] = "test"
    else:
        train_dataset_kwargs["train"] = True
        test_dataset_kwargs["train"] = False

    # Build datasets
    train_dataset = dataset_class(
        root=data_dir,
        download=True,
        transform=train_transform,
        **dataset_config["train_kwargs"],
    )
    val_dataset = dataset_class(
        root=data_dir,
        download=True,
        transform=val_transform,
        **dataset_config["test_kwargs"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": (num_workers > 0)
        and (device == "cuda"),  # default, will be set to True if device is CUDA
        "drop_last": True,  # drop last incomplete batch
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),  # for reproducibility
    }

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )
    if device == "cuda":
        print(
            f"Using CUDA with {num_workers} workers and pin_memory={dataloader_kwargs.get('pin_memory', False)}"
        )

    # Create info object
    info = DatasetInfo(
        num_classes=dataset_config["num_classes"],
        input_shape=input_shape,
        mean=mean,
        std=std,
        class_names=dataset_config["class_names"],
    )
    return train_loader, val_loader, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation")
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Data directory for the dataset"
    )
    args = parser.parse_args()

    train_loader, val_loader, info = make_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        data_dir=args.data_dir,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Classes: {info.num_classes}")
    print(f"Input shape: {info.input_shape}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
