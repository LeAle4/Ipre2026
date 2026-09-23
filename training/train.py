import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR   = "data"
SEED       = 42
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-4

TRAIN_AREA = "Area chugchug"
VAL_AREA   = "Area lluta"
TEST_AREA  = "Area unita"
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        # samples: list of (path, label, area)
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, area = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, area


def split_samples(samples, train_area, val_area, test_area):
    train = [s for s in samples if s[2] == train_area]
    val   = [s for s in samples if s[2] == val_area]
    test  = [s for s in samples if s[2] == test_area]
    return train, val, test


train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    return model.to(DEVICE)


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for imgs, labels, _ in tqdm(loader, leave=False):
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            probs = model(imgs)
            loss  = criterion(probs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds       = (probs >= 0.5).float()
            correct    += (preds == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


# ── Sample Collection ─────────────────────────────────────────────────────────

def collect_samples(data_dir):
    samples = []
    for area_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(area_dir):
            continue
        area = os.path.basename(area_dir)
        for folder, label in [("positives", 1), ("negatives", 0)]:
            folder_path = os.path.join(area_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
                for path in glob.glob(os.path.join(folder_path, ext)):
                    samples.append((path, label, area))
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Using device: {DEVICE}")
    samples = collect_samples(DATA_DIR)
    if not samples:
        raise FileNotFoundError(f"No images found under '{DATA_DIR}'. Check DATA_DIR.")

    train_s, val_s, test_s = split_samples(samples, TRAIN_AREA, VAL_AREA, TEST_AREA)
    print(f"Samples — train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}")

    train_loader = DataLoader(ImageDataset(train_s, train_tf), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(ImageDataset(val_s,   eval_tf),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(ImageDataset(test_s,  eval_tf),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = build_model()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion)
        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"train loss {tr_loss:.4f}  acc {tr_acc:.4f}  |  "
              f"val loss {vl_loss:.4f}  acc {vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"             v saved best model (val acc {best_val_acc:.4f})")

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.eval()

    all_preds, all_labels, all_areas = [], [], []
    with torch.no_grad():
        for imgs, labels, areas in tqdm(test_loader, desc="Testing"):
            imgs  = imgs.to(DEVICE)
            probs = model(imgs).squeeze(-1)
            preds = (probs >= 0.5).long().cpu().tolist()
            all_preds  .extend(preds)
            all_labels .extend(labels.tolist())
            all_areas  .extend(areas)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    print(f"\nTest results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {TEST_AREA}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{TEST_AREA}.png", dpi=150)
    plt.close()
    print(f"Saved confusion_matrix_{TEST_AREA}.png")

    # ── Per-area accuracy ─────────────────────────────────────────────────────
    all_areas = np.array(all_areas)
    areas     = sorted(set(all_areas))
    area_accs, area_counts = [], []

    print("\nPer-area accuracy:")
    for area in areas:
        mask  = all_areas == area
        n     = mask.sum()
        a_acc = (all_preds[mask] == all_labels[mask]).mean()
        area_accs.append(a_acc)
        area_counts.append(n)
        print(f"  {area}: {a_acc*100:.1f}% ({n} samples)")

    fig, ax = plt.subplots(figsize=(max(6, len(areas) * 1.1), 5))
    bars = ax.barh(areas, [a * 100 for a in area_accs], color="#4C72B0", edgecolor="white")
    for bar, a_acc, n in zip(bars, area_accs, area_counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{a_acc*100:.1f}%  (n={n})", va="center", fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Test Accuracy per Area")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("area_accuracy.png", dpi=150)
    plt.close()
    print("Saved area_accuracy.png")


if __name__ == "__main__":
    main()

