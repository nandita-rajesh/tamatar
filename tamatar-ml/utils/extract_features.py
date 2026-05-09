import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json

def main():
    # ========================
    # CONFIG
    # ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    CHECKPOINT_DIR = Path("checkpoints")
    OUTPUT_DIR = Path("features")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========================
    # TRANSFORMS
    # ========================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # ========================
    # LOAD DATA 
    # ========================
    dataset = ImageFolder("dataset", transform=transform)

    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    classes = dataset.classes
    num_classes = len(classes)

    with open(OUTPUT_DIR / "class_mapping.json", "w") as f:
        json.dump(dataset.class_to_idx, f, indent=4)

    # ========================
    # LOAD MODELS
    # ========================
    def load_resnet():
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(CHECKPOINT_DIR / "resnet18_tomato.pth", map_location=device))
        return model.to(device).eval()

    def load_mobilenet():
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        model.load_state_dict(torch.load(CHECKPOINT_DIR / "mobilenet_tomato.pth", map_location=device))
        return model.to(device).eval()

    def load_shufflenet():
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(CHECKPOINT_DIR / "shufflenet_v2_tomato.pth", map_location=device))
        return model.to(device).eval()

    resnet = load_resnet()
    mobilenet = load_mobilenet()
    shufflenet = load_shufflenet()

    # ========================
    # FEATURE STORAGE
    # ========================
    features_resnet = []
    features_mobilenet = []
    features_shufflenet = []
    labels_all = []

    # ========================
    # EXTRACTION LOOP
    # ========================
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            # Extract features (FC output)
            f_res = resnet(images)
            f_mob = mobilenet(images)
            f_shuf = shufflenet(images)

            features_resnet.append(f_res.cpu().numpy())
            features_mobilenet.append(f_mob.cpu().numpy())
            features_shufflenet.append(f_shuf.cpu().numpy())
            labels_all.append(labels.numpy())

    # ========================
    # CONCATENATE FEATURES
    # ========================
    features_resnet = np.vstack(features_resnet)
    features_mobilenet = np.vstack(features_mobilenet)
    features_shufflenet = np.vstack(features_shufflenet)
    labels_all = np.hstack(labels_all)

    # Combine
    features_combined = np.hstack([
        features_resnet,
        features_mobilenet,
        features_shufflenet
    ])

    # ========================
    # SAVE OUTPUT
    # ========================
    np.save(OUTPUT_DIR / "resnet_features.npy", features_resnet)
    np.save(OUTPUT_DIR / "mobilenet_features.npy", features_mobilenet)
    np.save(OUTPUT_DIR / "shufflenet_features.npy", features_shufflenet)
    np.save(OUTPUT_DIR / "combined_features.npy", features_combined)
    np.save(OUTPUT_DIR / "labels.npy", labels_all)

    print("\nFeature extraction completed")
    print(f"ResNet features shape: {features_resnet.shape}")
    print(f"MobileNet features shape: {features_mobilenet.shape}")
    print(f"ShuffleNet features shape: {features_shufflenet.shape}")
    print(f"Combined features shape: {features_combined.shape}")

if __name__ == "__main__":
    main()