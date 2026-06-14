import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from torchvision.ops import deform_conv2d

# ===== 데이터셋 =====
class WaferDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X / 2.0).unsqueeze(1)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===== Deformable Conv 레이어 =====
class DeformableConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformableConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

        # 오프셋 예측
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=padding
        )
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels,
                        kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            padding=self.padding
        )


# ===== DC-Net (Wang 2020 논문 정확한 구조) =====
class DCNet(nn.Module):
    """
    논문 Fig.5 기반 정확한 구조:
    Module1: Conv(32, 5x5) + BN + ReLU + MaxPool
    Module2: Conv(64, 3x3) + DC(64) + BN + ReLU + MaxPool
    Module3: Conv(128, 3x3) + DC(128) + BN + ReLU + MaxPool
    Module4: Conv(128, 3x3) + DC(128) + BN + ReLU + AdaptivePool
    Module5: FC(128) -> FC(8) + Sigmoid
    """
    def __init__(self, num_classes=8):
        super(DCNet, self).__init__()

        # Module 1: 일반 Conv (5x5) - 논문에서 첫 레이어는 일반 Conv
        self.module1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 52 → 26
        )

        # Module 2: 일반 Conv + Deformable Conv
        self.module2_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.module2_dc = DeformableConvLayer(64, 64)
        self.module2_bn = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 26 → 13
        )

        # Module 3: 일반 Conv + Deformable Conv
        self.module3_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.module3_dc = DeformableConvLayer(128, 128)
        self.module3_bn = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 13 → 6
        )

        # Module 4: 일반 Conv + Deformable Conv
        self.module4_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.module4_dc = DeformableConvLayer(128, 128)
        self.module4_bn = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3)  # → 3x3
        )

        # Module 5: FC 레이어
        self.module5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),  # FC1: 128 neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)    # FC2: 8 neurons
        )

    def forward(self, x):
        # Module 1
        x = self.module1(x)

        # Module 2
        x = self.module2_conv(x)
        x = self.module2_dc(x)
        x = self.module2_bn(x)

        # Module 3
        x = self.module3_conv(x)
        x = self.module3_dc(x)
        x = self.module3_bn(x)

        # Module 4
        x = self.module4_conv(x)
        x = self.module4_dc(x)
        x = self.module4_bn(x)

        # Module 5
        x = self.module5(x)
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
            all_labels.append(y_batch.cpu().numpy())  # ← .cpu() 추가
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    accuracy = (all_preds == all_labels).all(axis=1).mean()
    macro_f1 = f1_score(all_labels, all_preds,
                        average='macro', zero_division=0)
    return total_loss / len(loader), accuracy, macro_f1


# ===== 데이터 로딩 =====
print("데이터 로딩 중...")
data = np.load('data/Wafer_Map_Datasets.npz')
X = data['arr_0']
y = data['arr_1']

# 논문과 동일: Train 80% / Val 20%
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 테스트는 Val에서 절반 사용
X_val, X_test, y_val, y_test = train_test_split(
    X_val, y_val, test_size=0.5, random_state=42)

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

model = DCNet(num_classes=8).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5)

total_params = sum(p.numel() for p in model.parameters())
print(f"DC-Net 파라미터 수: {total_params:,}개 ({total_params/1e6:.2f}M)")

# ===== 학습 실행 =====
print("\n학습 시작! (DC-Net 논문 정확한 재현)")
EPOCHS = 30
train_losses, val_losses, val_accs = [], [], []
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

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'results/best_dcnet_accurate.pth')

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Macro F1: {val_f1:.4f}")

print(f"\n최고 Val Accuracy: {best_acc:.4f}")

# ===== 테스트 결과 =====
model.load_state_dict(torch.load(
    'results/best_dcnet_accurate.pth', map_location=device))
test_loss, test_acc, test_f1 = eval_epoch(
    model, test_loader, criterion, device)

print(f"\n=== 최종 비교 결과 ===")
print(f"{'모델':<30} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 52)
print(f"{'DC-Net 원논문 (Wang 2020)':<30} {'93.20%':>10} {'94.00%':>10}")
print(f"{'DC-Net 재현 (우리)':<30} "
      f"{test_acc*100:>9.2f}% "
      f"{test_f1*100:>9.2f}%")
print(f"{'단순 CNN (우리)':<30} {'97.21%':>10} {'99.41%':>10}")
print(f"{'CNN + CBAM (우리)':<30} {'97.87%':>10} {'99.23%':>10}")

diff_orig = test_acc * 100 - 93.20
diff_cbam = test_acc * 100 - 97.87
print(f"\nDC-Net 재현 vs 원논문:   {diff_orig:+.2f}%p")
print(f"DC-Net 재현 vs CNN+CBAM: {diff_cbam:+.2f}%p")

# ===== 학습 곡선 =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_title('Loss Curve (DC-Net 재현)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(val_accs, label='Val Accuracy', color='green')
ax2.axhline(y=0.9320, color='red', linestyle='--',
            label='원논문 93.20%')
ax2.axhline(y=0.9787, color='blue', linestyle='--',
            label='CNN+CBAM 97.87%')
ax2.set_title('Accuracy Curve (DC-Net 재현)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/dcnet_accurate_curve.png', dpi=150)
plt.show()
print("학습 곡선 저장 완료!")