"""07. K-평균 군집화 (K-Means) - 엘보우 & 실루엣 점수"""
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    print(f"k={k}  inertia={km.inertia_:9.1f}  silhouette={sil:.4f}")
