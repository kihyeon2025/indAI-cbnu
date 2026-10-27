"""15. PyTorch 다층 퍼셉트론 (MLP) 분류 - 와인 데이터셋"""
import torch
from torch import nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)
X_tr, X_te = (torch.tensor(sc.transform(a), dtype=torch.float32) for a in (X_tr, X_te))
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)

model = nn.Sequential(nn.Linear(13, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 3))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, 101):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(X_tr), y_tr)
    loss.backward()
    opt.step()
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(X_te).argmax(1) == y_te).float().mean().item()
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}  test_acc={acc:.4f}")
