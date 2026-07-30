import torch
import torch.nn as nn


class CustomTinyCNN(nn.Module):
    """A simple CNN model for image classification."""

    def __init__(self, num_classes: int = 10, max_pool_kernel_size: int = 2):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=max_pool_kernel_size)

        # Define your model architecture here
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            self.pool,
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            self.pool,
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self.pool,
        )

        # adaptive pooling layer to reduce the spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = self.block3[
            0
        ].out_channels  # Get the number of output channels from the last conv layer

        # classifier head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(self.out_channels, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # validate the input shape
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input shape (N, 3, H, W), but got {x.shape}")

        # Define the forward pass
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # flatten the output of the convolutional layers
        x = self.adaptive_pool(x)

        x = self.head(x)
        return x

    def __repr__(self):
        return f"CustomTinyCNN(num_classes={self.head[-1].out_features})"
