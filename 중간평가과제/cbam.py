import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os

os.makedirs('results', exist_ok=True)

# ===== 데이터셋 =====
class WaferDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X / 2.0).unsqueeze(1)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===== CBAM 모듈 =====
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        max_ = self.fc(self.max_pool(x).view(b, c))
        att = self.sigmoid(avg + max_).view(b, c, 1, 1)
        return x * att


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        att = self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * att


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ===== CNN + CBAM 모델 =====
class LightCNN_CBAM(nn.Module):
    def __init__(self, num_classes=8):
        super(LightCNN_CBAM, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cbam1 = CBAM(32)

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cbam2 = CBAM(64)

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.cbam3 = CBAM(128)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.cbam1(self.block1(x))
        x = self.cbam2(self.block2(x))
        x = self.cbam3(self.block3(x))
        x = self.classifier(x)
        return x


# ===== 학습/평가 함수 =====
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            preds = torch.sigmoid(output) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    accuracy = (all_preds == all_labels).all(axis=1).mean()
    macro_f1 = f1_score(all_labels, all_preds,
                        average='macro', zero_division=0)
    return total_loss / len(loader), accuracy, macro_f1


# ===== 데이터 로딩 =====
print("데이터 로딩 중...")
data = np.load('D:/indai/data/Wafer_Map_Datasets.npz')
X = data['arr_0']
y = data['arr_1']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {len(X_train)}장")
print(f"Val:   {len(X_val)}장")
print(f"Test:  {len(X_test)}장")

train_loader = DataLoader(WaferDataset(X_train, y_train),
                          batch_size=64, shuffle=True)
val_loader   = DataLoader(WaferDataset(X_val, y_val),
                          batch_size=64, shuffle=False)
test_loader  = DataLoader(WaferDataset(X_test, y_test),
                          batch_size=64, shuffle=False)

# ===== 학습 설정 =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

model = LightCNN_CBAM(num_classes=8).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30)

total_params = sum(p.numel() for p in model.parameters())
print(f"CNN+CBAM 파라미터 수: {total_params:,}개 ({total_params/1e6:.2f}M)")

# ===== 학습 실행 =====
print("\n학습 시작! (CNN + CBAM)")
EPOCHS = 30
train_losses, val_losses, val_accs, val_f1s = [], [], [], []
best_acc = 0

for epoch in range(EPOCHS):
    train_loss = train_epoch(
        model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, val_f1 = eval_epoch(
        model, val_loader, criterion, device)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),
                   'results/best_cbam_cuda.pth')

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Macro F1: {val_f1:.4f}")

print(f"\n최고 Val Accuracy: {best_acc:.4f}")

# ===== 테스트 결과 =====
model.load_state_dict(torch.load(
    'results/best_cbam_cuda.pth', map_location=device))
test_loss, test_acc, test_f1 = eval_epoch(
    model, test_loader, criterion, device)

print(f"\n=== 최종 비교 결과 ===")
print(f"{'모델':<30} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 52)
print(f"{'DC-Net 원논문 (Wang 2020)':<30} {'93.20%':>10} {'94.00%':>10}")
print(f"{'DC-Net 재현 (우리)':<30} {'98.13%':>10} {'98.54%':>10}")
print(f"{'CNN + CBAM (우리)':<30} "
      f"{test_acc*100:>9.2f}% "
      f"{test_f1*100:>9.2f}%")

diff_dcnet = test_acc * 100 - 98.13
diff_orig  = test_acc * 100 - 93.20
print(f"\nCNN+CBAM vs DC-Net 재현: {diff_dcnet:+.2f}%p")
print(f"CNN+CBAM vs 원논문:      {diff_orig:+.2f}%p")

# ===== 학습 곡선 =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_title('Loss Curve (CNN + CBAM)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(val_accs, label='Val Accuracy', color='green')
ax2.plot(val_f1s, label='Macro F1', color='blue')
ax2.axhline(y=0.9813, color='red', linestyle='--', label='DC-Net 98.13%')
ax2.axhline(y=0.9320, color='gray', linestyle='--', label='DC-Net 93.20%')
ax2.set_title('Accuracy Curve (CNN + CBAM)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Score')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/cbam_cuda_curve.png', dpi=150)
plt.show()
print("학습 곡선 저장 완료!")