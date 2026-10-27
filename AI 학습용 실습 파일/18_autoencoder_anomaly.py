"""18. 오토인코더 기반 이상 탐지 - 가상 다변량 센서 데이터"""
import numpy as np
import torch
from torch import nn

torch.manual_seed(42)
rng = np.random.default_rng(42)

normal = rng.normal(0, 1, (1000, 8)).astype(np.float32)
normal[:, 4:] = normal[:, :4] * 0.8 + rng.normal(0, 0.1, (1000, 4))   # 변수 간 상관관계
anomal = rng.normal(0, 2.5, (50, 8)).astype(np.float32)               # 상관관계 깨진 데이터

X_train = torch.tensor(normal[:800])
X_val = torch.tensor(normal[800:])
X_anom = torch.tensor(anomal)

model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2), nn.ReLU(),
                      nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 8))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(200):
    opt.zero_grad()
    loss = ((model(X_train) - X_train) ** 2).mean()
    loss.backward()
    opt.step()


def recon_error(x):
    with torch.no_grad():
        return ((model(x) - x) ** 2).mean(dim=1)


threshold = recon_error(X_train).quantile(0.95).item()
print(f"임계값(정상 95% 분위수): {threshold:.4f}")
print(f"정상 검증 데이터 오탐률: {(recon_error(X_val) > threshold).float().mean():.3f}")
print(f"이상 데이터 탐지율    : {(recon_error(X_anom) > threshold).float().mean():.3f}")
