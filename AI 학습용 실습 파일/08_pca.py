"""08. 주성분 분석 (PCA) - 차원 축소 및 분산 설명력"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X, _ = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)

pca = PCA().fit(X)
cum = np.cumsum(pca.explained_variance_ratio_)
n95 = int(np.argmax(cum >= 0.95)) + 1
print(f"원본 차원: {X.shape[1]}")
print(f"분산 95% 유지에 필요한 주성분 수: {n95}")

X2 = PCA(n_components=2).fit_transform(X)
print("2차원 변환 결과 shape:", X2.shape)
