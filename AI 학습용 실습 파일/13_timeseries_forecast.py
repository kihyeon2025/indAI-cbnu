"""13. 시계열 예측 - 지연(lag) 변수 + 랜덤 포레스트"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)
t = np.arange(600)
series = 10 + 0.01 * t + 3 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.3, len(t))

LAGS = 10
X = np.array([series[i - LAGS:i] for i in range(LAGS, len(series))])
y = series[LAGS:]

split = int(len(X) * 0.8)          # 시계열은 순서를 유지해서 분할
model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X[:split], y[:split])
pred = model.predict(X[split:])

print(f"MAE: {mean_absolute_error(y[split:], pred):.4f}")
