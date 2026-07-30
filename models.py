from torch import nn
import torch
import torchvision.models as models
from custom_cnn import CustomModel

def make_model(arch, num_classes, pretrained):
    # Placeholder for model creation logic
    if arch == "custom":
        model = CustomModel(num_classes)
    elif arch == "convnext":
        model = models.convnext_base(pretrained=pretrained)

        # for fine tuning we freeze parameters
        # so learning from Image Net doesn't lose while traiing
        for param in model.parameters():
            param.requires_grad = False

        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif arch == "vit":
        model = models.vit_b_16(pretrained=pretrained)

        # for fine tuning we freeze parameters
        # so learning from Image Net doesn't lose while traiing
        for param in model.parameters():
            param.requires_grad = False

        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif arch == "frcnn":
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

        # for fine tuning we freeze parameters
        # so learning from Image Net doesn't lose while traiing
        for param in model.parameters():
            param.requires_grad = False

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
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
