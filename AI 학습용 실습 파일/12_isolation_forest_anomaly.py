"""12. Isolation Forest 이상 탐지 - 가상 센서 데이터 (산업 AI)"""
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(42)
normal = rng.normal(loc=[50, 1.0], scale=[2.0, 0.05], size=(500, 2))   # [온도, 진동]
faulty = rng.normal(loc=[65, 1.6], scale=[3.0, 0.15], size=(25, 2))
X = np.vstack([normal, faulty])
y_true = np.array([0] * len(normal) + [1] * len(faulty))

model = IsolationForest(contamination=0.05, random_state=42).fit(X)
y_pred = (model.predict(X) == -1).astype(int)

tp = int(((y_pred == 1) & (y_true == 1)).sum())
fp = int(((y_pred == 1) & (y_true == 0)).sum())
fn = int(((y_pred == 0) & (y_true == 1)).sum())
print(f"탐지된 이상: {int(y_pred.sum())}  TP={tp}  FP={fp}  FN={fn}")
print(f"Precision={tp / max(tp + fp, 1):.3f}  Recall={tp / max(tp + fn, 1):.3f}")
