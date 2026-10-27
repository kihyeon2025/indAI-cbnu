"""11. DBSCAN 밀도 기반 군집화 - 비선형 형태 데이터"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X, _ = make_moons(n_samples=400, noise=0.07, random_state=42)
X = StandardScaler().fit_transform(X)

labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(X)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"군집 수: {n_clusters}")
print(f"노이즈 포인트 수: {int(np.sum(labels == -1))}")
