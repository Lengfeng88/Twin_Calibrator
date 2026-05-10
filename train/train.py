import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.dataset import LightDataset
from train.model   import LightRegressor, denormalize

# ── 超参 ────────────────────────────────────────
EPOCHS      = 30
BATCH_SIZE  = 256
LR          = 1e-3
N_SAMPLES   = 8000
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH   = "models/light_regressor.pt"

print(f"训练设备: {DEVICE}")

# ── 数据集 ──────────────────────────────────────
print("生成合成数据集...")
ds = LightDataset(n_samples=N_SAMPLES, img_size=64)
n_val = int(len(ds) * 0.15)
train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

# ── 模型 / 优化器 ───────────────────────────────
model     = LightRegressor().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                               weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()

# ── 训练循环 ────────────────────────────────────
best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for imgs, labels in tqdm(train_loader,
                              desc=f"Epoch {epoch:02d}/{EPOCHS}",
                              leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(imgs)
    scheduler.step()
    train_loss /= len(train_ds)

    # ── 验证 ──────────────────────────────────
    model.eval()
    lum_errs, cct_errs = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            pred = model(imgs)
            lum_p, cct_p = denormalize(pred)
            lum_t = labels[:, 0] * 100.0
            cct_t = labels[:, 1] * 5300.0 + 2700.0
            lum_errs.append((lum_p - lum_t).abs().mean().item())
            cct_errs.append((cct_p - cct_t).abs().mean().item())

    mae_lum = sum(lum_errs) / len(lum_errs)
    mae_cct = sum(cct_errs) / len(cct_errs)

    print(f"Epoch {epoch:02d} | train_loss={train_loss:.5f} "
          f"| MAE lum={mae_lum:.2f}%  cct={mae_cct:.0f}K")

    if mae_lum < best_val:
        best_val = mae_lum
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  → 保存最优模型 (mae_lum={mae_lum:.2f}%)")

print(f"\n训练完成。最优模型: {SAVE_PATH}")