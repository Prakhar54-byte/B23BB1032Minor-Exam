import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from datasets import load_dataset
from huggingface_hub import HfApi
import wandb
import os
import copy

# ==========================
# 1. Load Data from Hugging Face
# ==========================
print("Loading dataset from Hugging Face...")
ds = load_dataset("Chiranjeev007/CIFAR-10_Subset")
print(ds)

# ==========================
# 2. Define Transforms
# ==========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================
# 3. Custom Dataset Class
# ==========================
class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================
# 4. Create DataLoaders
# ==========================
BATCH_SIZE = 32

train_dataset = HFImageDataset(ds["train"], transform=train_transform)
val_dataset = HFImageDataset(ds["validation"], transform=val_transform)
test_dataset = HFImageDataset(ds["test"], transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ==========================
# 5. Load Pretrained ResNet-18
# ==========================
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

print(f"Model loaded on: {DEVICE}")

# ==========================
# 6. Loss, Optimizer, Config
# ==========================
EPOCHS = 10
LR = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================
# 7. Initialize WandB
# ==========================
wandb.init(
    project="minor_b23bb1032",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "model": "resnet18-pretrained",
        "dataset": "Chiranjeev007/CIFAR-10_Subset",
        "optimizer": "Adam",
    }
)

# ==========================
# 8. Training Loop with WandB Logging
# ==========================
best_val_acc = 0.0
best_model_state = None

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    })

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # --- Save Best Model ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, "best_model.pth")
        print(f"  -> Best model saved! (Val Acc: {val_acc:.2f}%)")

print(f"\nTraining Complete! Best Val Accuracy: {best_val_acc:.2f}%")

# ==========================
# 9. Test Evaluation
# ==========================
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100.0 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")
wandb.log({"test_accuracy": test_acc})

# ==========================
# 10. Push Best Model to Hugging Face Hub
# ==========================
HF_REPO_ID = "Prakhar54-byte/HF_minor_b23bb1032"  

api = HfApi()

# Create repo (if it doesn't exist)
api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)

# Upload the best model file
api.upload_file(
    path_or_fileobj="best_model.pth",
    path_in_repo="best_model.pth",
    repo_id=HF_REPO_ID,
)

print(f"Model pushed to https://huggingface.co/{HF_REPO_ID}")

wandb.finish()