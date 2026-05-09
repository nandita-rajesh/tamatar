import torch
import torchvision
from torchvision import transforms
from utils.dataset_loader import load_dataset
from config import EPOCHS, LEARNING_RATE
import os

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

loader, _, _ = load_dataset(
    data_path="dataset",
    batch_size=10,
    transform=transform,
    train_ratio=1.0,
    val_ratio=0.0,
    test_ratio=0.0
)

classes = loader.dataset.dataset.classes

model = torchvision.models.mobilenet_v2(
    weights=torchvision.models.MobileNet_V2_Weights.DEFAULT
)

for param in model.parameters():
    param.requires_grad = True

model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.classifier[1].parameters(),
    lr=LEARNING_RATE,
    momentum=0.9
)

for epoch in range(EPOCHS):
    print("\n" + "=" * 30)
    print(f"Starting epoch {epoch+1}/{EPOCHS}")
    print("-" * 30)

    model.train()
    running_loss = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/total:.4f}")

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/mobilenet_tomato.pth")