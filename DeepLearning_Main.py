import os
import timm
import torch
import torch.nn as nn
from torchvision import models


def build_densenet_model(device, retrain, UF_flag, weights_path):
    print("\nDownloading and Building DenseNet-169...")
    model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )

    if retrain:
        print(f"Loading model: {os.path.basename(weights_path)}")
        model.load_state_dict(
            torch.load(
                weights_path,
                map_location=device,
                weights_only=True
            )
        )

    for param in model.parameters():
        param.requires_grad = False

#----------------Layers to be unfrozen----------------#
    for param in model.classifier.parameters():
        param.requires_grad = True

    if UF_flag:
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True

        for param in model.features.norm5.parameters():
            param.requires_grad = True

    return model.to(device)


def build_resnet_model(device, retrain, UF_flag, weights_path):
    print("\nDownloading and Building ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )

    if retrain:
        print(f"Loading model: {os.path.basename(weights_path)}")
        model.load_state_dict(
            torch.load(
                weights_path,
                map_location=device,
                weights_only=True
            )
        )

    for param in model.parameters():
        param.requires_grad = False

    # ----------------Layers to be unfrozen----------------#
    for param in model.fc.parameters():
        param.requires_grad = True

    if UF_flag:
        for param in model.layer4.parameters():
            param.requires_grad = True

    return model.to(device)


def build_inception_model(device, retrain, UF_flag, weights_path):
    print("\nDownloading and Building Inception-ResNet v2...")
    model = timm.create_model('inception_resnet_v2', pretrained=True)

    in_features = model.classif.in_features

    model.classif = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )

    if retrain:
        print(f"Loading model: {os.path.basename(weights_path)}")
        model.load_state_dict(
            torch.load(
                weights_path,
                map_location=device,
                weights_only=True
            )
        )

    for param in model.parameters():
        param.requires_grad = False

#----------------Layers to be unfrozen----------------#
    for param in model.classif.parameters():
        param.requires_grad = True

    if UF_flag:
        children = list(model.children())
        total_blocks = len(children)
        unfreeze_cutoff = int(total_blocks * 0.85)
        print(f"Total High-Level Blocks: {total_blocks}")
        print(f"Unfreezing blocks from index {unfreeze_cutoff} to the end...")

        for i, child in enumerate(children):
            if i >= unfreeze_cutoff:
                for param in child.parameters():
                    param.requires_grad = True

    return model.to(device)


def build_model(model_name, device, retrain, UF_flag, weights_path):
    if model_name == "densenet":
        return build_densenet_model(device, retrain, UF_flag, weights_path)
    elif model_name == "resnet":
        return build_resnet_model(device, retrain, UF_flag, weights_path)
    elif model_name == "inception":
        return build_inception_model(device, retrain, UF_flag, weights_path)
    else:
        raise ValueError(f"Invalid model name: {model_name}")