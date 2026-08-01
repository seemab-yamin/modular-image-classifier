# config_loader.py
import argparse

import yaml


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args_with_defaults(config_path="config.yaml"):
    """
    Parse CLI arguments with defaults from YAML config.
    CLI args override YAML defaults.
    """
    # Load defaults from YAML
    defaults = load_config(config_path)

    # Set up argument parser with defaults from YAML
    parser = argparse.ArgumentParser(description="Image Classification Training")

    parser.add_argument(
        "--arch",
        type=str,
        default=defaults.get("arch", "custom"),
        help="Model architecture",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults.get("epochs", 10),
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=defaults.get("dataset", "cifar10"),
        help="Name of the dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.get("batch_size", 32),
        help="Batch size for dataloaders",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.get("learning_rate", 0.001),
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults.get("num_workers", 4),
        help="Number of workers for dataloaders",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=defaults.get("pretrained", False),
        help="Use pretrained weights",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=defaults.get("freeze_backbone", False),
        help="Freeze the backbone of the model",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=defaults.get("root_dir", "./data"),
        help="Root directory for the dataset",
    )

    return parser.parse_args()
