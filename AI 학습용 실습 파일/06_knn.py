"""06. K-최근접 이웃 (KNN) - k 값에 따른 정확도 비교"""
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)

for k in (1, 3, 5, 7, 9, 11):
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    score = cross_val_score(model, X, y, cv=5).mean()
    print(f"k={k:2d}  5-fold CV accuracy = {score:.4f}")
