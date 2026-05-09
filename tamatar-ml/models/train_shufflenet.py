import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from utils.dataset_loader import load_dataset
from config import EPOCHS, LEARNING_RATE
from pathlib import Path

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms (IMPORTANT: same for train + test)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomAffine(
        degrees=0,
        shear=10,
        scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# Load dataset
train_loader, _, _ = load_dataset(
    data_path="dataset",
    batch_size=10,
    transform=transform,
    train_ratio=1.0,
    val_ratio=0.0,
    test_ratio=0.0
)

# Get class names
classes = train_loader.dataset.dataset.classes

# Model
model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)

# Freeze layers
for param in model.parameters():
    param.requires_grad = True

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.fc.parameters(), 
    lr=LEARNING_RATE,
    momentum=0.9)

# ========================
# TRAINING
# ========================
for epoch in range(EPOCHS):
    print("\n" + "="*30)
    print(f"Starting epoch {epoch+1}/{EPOCHS}")
    print("-"*30)

    model.train()
    running_loss = 0.0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    train_loss = running_loss / total

    print("\n" + "="*30)
    print(f"Epoch {epoch+1} / {EPOCHS}")
    print("-"*30)
    print(f"Loss : {train_loss:.4f}")
    print("="*30)

# Save model
output_dir = Path(__file__).resolve().parents[1] / "checkpoints"
output_dir.mkdir(parents=True, exist_ok=True)
save_path = output_dir / "shufflenet_v2_tomato.pth"

torch.save(model.state_dict(), str(save_path))
print(f"\nModel saved to {save_path}")
