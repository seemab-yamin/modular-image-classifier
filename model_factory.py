import torch
import torchvision.models as models
from torch import nn

from cnn_architectures import CustomTinyCNN


class FasterRCNNBackboneClassifier(nn.Module):
    def __init__(self, pretrained=True, num_classes=10, freeze_backbone=True):
        super().__init__()

        # backbone
        detection_model = models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None
        )
        self.backbone = detection_model.backbone

        self.out_channels = self.backbone.out_channels

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(
            in_features=self.out_channels, out_features=num_classes
        )

        if pretrained and freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)

        feat = features["0"]
        feat = self.pool(feat)

        feat = feat.flatten(1)

        logits = self.classifier(feat)

        return logits


def make_model(arch, in_channels, num_classes, pretrained, freeze_backbone):
    # Placeholder for model creation logic
    if arch == "custom":
        model = CustomTinyCNN(in_channels, num_classes)
    elif arch == "convnext":
        model = models.convnext_base(weights="DEFAULT" if pretrained else None)

        # for fine tuning we freeze parameters
        # so learning from Image Net doesn't lose while training
        if pretrained and freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif arch == "vit":
        model = models.vit_b_16(weights="DEFAULT" if pretrained else None)

        # for fine tuning we freeze parameters
        # so learning from Image Net doesn't lose while training
        if pretrained and freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif arch == "frcnn":
        model = FasterRCNNBackboneClassifier(
            pretrained, num_classes, freeze_backbone=freeze_backbone
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


if __name__ == "__main__":
    arch = "custom"
    num_classes = 10
    pretrained = False
    model = make_model(arch=arch, num_classes=num_classes, pretrained=pretrained)

    print(f"Selected model:\n{arch} with pretrained={pretrained}")
    model_summary = str(model)
    print(f"Created model:\n{model_summary}")
