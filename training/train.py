"""
Chest X-Ray Pneumonia Detection — ResNet-18 Fine-Tuning
Dataset: NIH Chest X-Ray (subset: Normal vs Pneumonia)
Output : model.pth + model.onnx
"""

import os
import time
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# ─── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")   # data/train  data/val
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "training")
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR         = 1e-4
NUM_CLASSES = 2            # 0 = Normal, 1 = Pneumonia
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ─── Transforms ────────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─── Dataset ───────────────────────────────────────────────────────────────────
def load_datasets():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir   = os.path.join(DATA_DIR, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_transforms)

    # Weighted sampler to handle class imbalance
    targets   = torch.tensor(train_ds.targets)
    class_counts = torch.bincount(targets).float()
    weights   = 1.0 / class_counts
    sample_wt = weights[targets]
    sampler   = WeightedRandomSampler(sample_wt, len(sample_wt))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"[INFO] Classes: {train_ds.classes}")
    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader, train_ds.classes

# ─── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze all layers except layer4 + fc
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    return model.to(DEVICE)

# ─── Training Loop ─────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_auc   = 0.0
    best_wts   = copy.deepcopy(model.state_dict())
    history    = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        # ── Train phase
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # ── Val phase
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_auc   = roc_auc_score(all_labels, all_probs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
              f"AUC={val_auc:.4f} | {elapsed:.1f}s")

        if val_auc > best_auc:
            best_auc = val_auc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, os.path.join(OUTPUT_DIR, "model.pth"))
            print(f"  ✔ Best model saved (AUC={best_auc:.4f})")

        scheduler.step()

    model.load_state_dict(best_wts)
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[DONE] Best Val AUC: {best_auc:.4f}")
    return model

# ─── Final Evaluation ──────────────────────────────────────────────────────────
def evaluate(model, val_loader, classes):
    model.eval()
    preds, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            out  = model(imgs)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred  = out.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            labels_all.extend(labels.numpy())
            probs_all.extend(probs)
    print("\n=== Classification Report ===")
    print(classification_report(labels_all, preds, target_names=classes))
    auc = roc_auc_score(labels_all, probs_all)
    print(f"ROC-AUC: {auc:.4f}")

# ─── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, classes = load_datasets()
    model = build_model()
    model = train(model, train_loader, val_loader)
    evaluate(model, val_loader, classes)
    print("\n[INFO] Training complete. Run export_onnx.py next.")
