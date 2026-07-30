import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, dim, num_of_classes):
        super().__init__()
        self.num_of_classes = num_of_classes

        # Define your model architecture here
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # fully connected classifier layer
        self.fc = nn.Linear(16 * 32 * 32, self.num_of_classes)

    def forward(self, x):
        # Define the forward pass
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
