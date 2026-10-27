"""17. PyTorch LSTM 시계열 예측 - 사인파 + 노이즈"""
import numpy as np
import torch
from torch import nn

torch.manual_seed(42)
rng = np.random.default_rng(42)
t = np.arange(1000)
series = np.sin(2 * np.pi * t / 40) + rng.normal(0, 0.05, len(t))

WIN = 30
X = np.array([series[i:i + WIN] for i in range(len(series) - WIN)])
y = series[WIN:]
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
split = int(len(X) * 0.8)


class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


model = LSTMNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

for epoch in range(1, 51):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(X[:split]), y[:split])
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test = loss_fn(model(X[split:]), y[split:]).item()
        print(f"epoch {epoch:2d}  train_mse={loss.item():.5f}  test_mse={test:.5f}")
