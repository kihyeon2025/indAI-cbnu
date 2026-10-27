"""14. 설비 예지보전 (Predictive Maintenance) - 가상 센서 데이터 고장 분류"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "temperature": rng.normal(60, 8, n),
    "vibration": rng.normal(1.0, 0.3, n),
    "pressure": rng.normal(100, 10, n),
    "rpm": rng.normal(1500, 100, n),
})
risk = 0.04 * (df.temperature - 60) + 1.5 * (df.vibration - 1.0) + 0.01 * (df.pressure - 100)
df["failure"] = (risk + rng.normal(0, 0.3, n) > 0.9).astype(int)
print("고장 비율:", round(df.failure.mean(), 3))

X, y = df.drop(columns="failure"), df["failure"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
model.fit(X_tr, y_tr)
print(classification_report(y_te, model.predict(X_te)))
