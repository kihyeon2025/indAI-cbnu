import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import os
os.makedirs('results', exist_ok=True)
from torchvision.ops import deform_conv2d

# ===== 38개 패턴 이름 정의 =====
pattern_names = [
    # 단일 (C1-C9)
    'C1_Normal', 'C2_Center', 'C3_Donut', 'C4_EdgeLoc',
    'C5_EdgeRing', 'C6_Loc', 'C7_NearFull', 'C8_Scratch', 'C9_Random',
    # 2개 복합 (C10-C22)
    'C10_C+EL', 'C11_C+ER', 'C12_C+L', 'C13_C+S',
    'C14_D+EL', 'C15_D+ER', 'C16_D+L', 'C17_D+S',
    'C18_EL+L', 'C19_EL+S', 'C20_ER+L', 'C21_ER+S', 'C22_L+S',
    # 3개 복합 (C23-C34)
    'C23_C+EL+L', 'C24_C+EL+S', 'C25_C+ER+L', 'C26_C+ER+S',
    'C27_C+L+S', 'C28_D+EL+L', 'C29_D+EL+S', 'C30_D+ER+L',
    'C31_D+ER+S', 'C32_D+L+S', 'C33_EL+L+S', 'C34_ER+L+S',
    # 4개 복합 (C35-C38)
    'C35_C+L+EL+S', 'C36_C+L+ER+S', 'C37_D+L+EL+S', 'C38_D+L+ER+S'
]

defect_names = ['Center', 'Donut', 'EdgeLoc', 'EdgeRing',
                'Loc', 'NearFull', 'Scratch', 'Random']

# ===== 레이블 → 패턴 ID 매핑 =====
def label_to_pattern_id(label):
    """8차원 레이블을 패턴 ID(0~37)로 변환"""
    label = tuple(int(x) for x in label)

    # 정상
    if sum(label) == 0:
        return 0  # C1 Normal

    active = [i for i, x in enumerate(label) if x == 1]

    # 단일 불량 (C2-C9)
    if len(active) == 1:
        return active[0] + 1  # C2~C9

    # 2개 복합 (C10-C22)
    two_combos = [
        (0,2), (0,3), (0,4), (0,6),  # C10-C13: C+EL, C+ER, C+L, C+S
        (1,2), (1,3), (1,4), (1,6),  # C14-C17: D+EL, D+ER, D+L, D+S
        (2,4), (2,6), (3,4), (3,6), (4,6)  # C18-C22
    ]
    if len(active) == 2:
        combo = tuple(active)
        if combo in two_combos:
            return two_combos.index(combo) + 9

    # 3개 복합 (C23-C34)
    three_combos = [
        (0,2,4), (0,2,6), (0,3,4), (0,3,6),  # C23-C26
        (0,4,6), (1,2,4), (1,2,6), (1,3,4),  # C27-C30
        (1,3,6), (1,4,6), (2,4,6), (3,4,6)   # C31-C34
    ]
    if len(active) == 3:
        combo = tuple(active)
        if combo in three_combos:
            return three_combos.index(combo) + 22

    # 4개 복합 (C35-C38)
    four_combos = [
        (0,2,4,6), (0,3,4,6), (1,2,4,6), (1,3,4,6)
    ]
    if len(active) == 4:
        combo = tuple(active)
        if combo in four_combos:
            return four_combos.index(combo) + 34

    return -1  # 미분류


# ===== 데이터셋 =====
class WaferDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X / 2.0).unsqueeze(1)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===== DC-Net + CBAM 모델 =====
class DeformableConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2*kernel_size*kernel_size,
                                     kernel_size=kernel_size, padding=padding)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
                                                 kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.padding = padding
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        return deform_conv2d(x, offset, self.weight, self.bias, padding=self.padding)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        max_ = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + max_).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
    def forward(self, x):
        return self.spatial_att(self.channel_att(x))

class DCNet_CBAM(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2))
        self.cbam1 = CBAM(32)
        self.module2_conv = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.module2_dc = DeformableConvLayer(64, 64)
        self.module2_bn = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.cbam2 = CBAM(64)
        self.module3_conv = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.module3_dc = DeformableConvLayer(128, 128)
        self.module3_bn = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.cbam3 = CBAM(128)
        self.module4_conv = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.module4_dc = DeformableConvLayer(128, 128)
        self.module4_bn = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(3))
        self.cbam4 = CBAM(128)
        self.module5 = nn.Sequential(
            nn.Flatten(), nn.Linear(128*3*3, 128),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.cbam1(self.module1(x))
        x = self.cbam2(self.module2_bn(self.module2_dc(self.module2_conv(x))))
        x = self.cbam3(self.module3_bn(self.module3_dc(self.module3_conv(x))))
        x = self.cbam4(self.module4_bn(self.module4_dc(self.module4_conv(x))))
        return self.module5(x)


# ===== 데이터 로딩 =====
print("데이터 로딩 중...")
data = np.load('D:/indai/data/Wafer_Map_Datasets.npz')
X = data['arr_0']
y = data['arr_1']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

test_loader = DataLoader(WaferDataset(X_test, y_test),
                         batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")

# ===== 모델 로딩 =====
print("DC-Net+CBAM 모델 로딩 중...")
model = DCNet_CBAM(num_classes=8).to(device)
model.load_state_dict(torch.load(
    'results/best_dcnet_cbam.pth', map_location=device))
model.eval()

# ===== 예측 =====
print("예측 중...")
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        preds = torch.sigmoid(output) > 0.5
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# ===== 패턴별 정확도 계산 =====
print("\n패턴별 정확도 계산 중...")

# 각 샘플의 패턴 ID 계산
pattern_ids = [label_to_pattern_id(label) for label in all_labels]

# 패턴별 정확도 계산
pattern_correct = {i: [] for i in range(38)}

for i, (pred, label, pid) in enumerate(zip(all_preds, all_labels, pattern_ids)):
    if pid >= 0:
        correct = np.array_equal(pred, label)
        pattern_correct[pid].append(correct)

# 결과 출력
print(f"\n{'패턴':<20} {'샘플수':>6} {'정확도':>8} {'원논문':>8} {'차이':>8}")
print("-" * 55)

# 원논문 정확도 (논문 Table III 기준)
dc_net_acc = {
    0: 99.70, 1: 97.80, 2: 96.50, 3: 94.40,
    4: 99.80, 5: 93.80, 6: 95.80, 7: 93.40, 8: 100.0,
    9: 99.20, 10: 97.90, 11: 98.50, 12: 96.70,
    13: 99.30, 14: 96.10, 15: 98.30, 16: 93.90, 17: 92.30,
    18: 94.60, 19: 90.70, 20: 90.30, 21: 88.90,
    22: 89.40, 23: 91.40, 24: 92.50, 25: 90.50,
    26: 90.30, 27: 88.30, 28: 90.50, 29: 91.50,
    30: 88.30, 31: 86.20, 32: 89.00, 33: 88.20,
    34: 87.00, 35: 90.60, 36: 86.40, 37: 88.20
}

results = []
for pid in range(38):
    samples = pattern_correct[pid]
    if len(samples) > 0:
        acc = np.mean(samples) * 100
        orig = dc_net_acc.get(pid, 0)
        diff = acc - orig
        arrow = "↑" if diff > 0 else "↓"
        results.append((pid, acc, orig, diff))
        print(f"{pattern_names[pid]:<20} {len(samples):>6} "
              f"{acc:>7.1f}% {orig:>7.1f}% "
              f"{arrow}{abs(diff):>5.1f}%")

# ===== 그룹별 평균 =====
print("\n=== 그룹별 평균 정확도 ===")
groups = [
    ("단일 불량 (C1-C9)", range(0, 9)),
    ("2개 복합 (C10-C22)", range(9, 22)),
    ("3개 복합 (C23-C34)", range(22, 34)),
    ("4개 복합 (C35-C38)", range(34, 38))
]

for group_name, pid_range in groups:
    our_accs = [r[1] for r in results if r[0] in pid_range]
    orig_accs = [r[2] for r in results if r[0] in pid_range]
    if our_accs:
        print(f"{group_name}:")
        print(f"  우리 DC-Net+CBAM: {np.mean(our_accs):.1f}%")
        print(f"  원논문 DC-Net:    {np.mean(orig_accs):.1f}%")
        print(f"  향상:             +{np.mean(our_accs)-np.mean(orig_accs):.1f}%p")

# ===== 그래프: 패턴별 정확도 비교 =====
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Pattern-wise Accuracy: DC-Net vs DC-Net+CBAM', fontsize=14)

group_ranges = [
    ("Single (C1-C9)", range(0, 9)),
    ("2-Mixed (C10-C22)", range(9, 22)),
    ("3-Mixed (C23-C34)", range(22, 34)),
    ("4-Mixed (C35-C38)", range(34, 38))
]

for ax, (title, pid_range) in zip(axes.flatten(), group_ranges):
    pids = [r[0] for r in results if r[0] in pid_range]
    our = [r[1] for r in results if r[0] in pid_range]
    orig = [r[2] for r in results if r[0] in pid_range]
    labels = [pattern_names[p].split('_')[1] for p in pids]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, orig, width, label='DC-Net (Wang 2020)',
           color='steelblue', alpha=0.8)
    ax.bar(x + width/2, our, width, label='DC-Net+CBAM (Ours)',
           color='coral', alpha=0.8)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(60, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/pattern_accuracy_comparison.png', dpi=150)
plt.show()
print("\n패턴별 정확도 그래프 저장 완료!")
print("results/pattern_accuracy_comparison.png")