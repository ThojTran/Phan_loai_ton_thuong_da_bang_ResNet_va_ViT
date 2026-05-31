"""
train_resnet.py — PH04: Huấn luyện ResNet18 trên HAM10000.

Chạy từ thư mục gốc:  python src/train_resnet.py
Output:
    checkpoints/resnet18_best.pth
    logs/resnet18_log.csv
"""

import csv, time, sys
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).parent))
from dataset import SkinDataset, NUM_CLASSES
from transforms import get_resnet_transforms

# ── Cấu hình ──────────────────────────────────────────────────────
SPLIT_DIR    = Path("data/splits")
CKPT_DIR     = Path("checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR      = Path("logs");        LOG_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE   = 64
NUM_EPOCHS   = 10
LR           = 1e-4        # LR nhỏ vì dùng pretrained weights
WEIGHT_DECAY = 1e-4        # L2 regularization nhẹ
NUM_WORKERS  = 0           # Windows: đặt 0 tránh lỗi multiprocessing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Class weights — xử lý imbalanced data ─────────────────────────
def compute_class_weights(dataset):
    # Lớp ít ảnh (vasc=142, df=115) sẽ có weight cao hơn
    counts  = dataset.df["label"].value_counts().sort_index()
    weights = 1.0 / counts.values.astype(float)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)

# ── Build model ────────────────────────────────────────────────────
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False             # đóng băng backbone
    model.fc = nn.Sequential(
        nn.Dropout(0.3),                        # dropout trước fc
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    return model

# ── Train / Validate 1 epoch ───────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval(); total_loss = 0.0; correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# ── Main ───────────────────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}")

    train_ds = SkinDataset(SPLIT_DIR / "ham10000_train.csv",
                           transform=get_resnet_transforms("train"))
    val_ds   = SkinDataset(SPLIT_DIR / "ham10000_val.csv",
                           transform=get_resnet_transforms("val"))
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Train : {len(train_ds):,}  |  Val : {len(val_ds):,}")

    model     = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_ds).to(DEVICE))
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    log_path = LOG_DIR / "resnet18_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss","val_acc","lr"])

    best_val_acc = 0.0; unfreeze_done = False

    for epoch in range(1, NUM_EPOCHS + 1):

        # Epoch 6: unfreeze backbone, backbone dùng LR × 0.1
        if epoch == 6 and not unfreeze_done:
            print("\n[Unfreeze] Mở backbone để fine-tune toàn bộ...")
            for p in model.parameters(): p.requires_grad = True
            optimizer = torch.optim.Adam([
                {"params": model.fc.parameters(), "lr": LR},
                {"params": [p for n,p in model.named_parameters() if "fc" not in n],
                 "lr": LR * 0.1},                # backbone LR nhỏ hơn 10 lần
            ], weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
            unfreeze_done = True

        t0 = time.time()
        train_loss        = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  ({time.time()-t0:.1f}s)")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{optimizer.param_groups[0]['lr']:.2e}"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_acc": val_acc, "val_loss": val_loss},
                       CKPT_DIR / "resnet18_best.pth")
            print(f"  ✅ Saved best  val_acc={val_acc:.4f}")

    print(f"\nDone. Best val_acc={best_val_acc:.4f}")

if __name__ == "__main__":
    main()