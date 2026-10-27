"""04. 랜덤 포레스트 (Random Forest) + 변수 중요도 """
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

wine = load_wine()
X_tr, X_te, y_tr, y_te = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
print(f"Accuracy: {accuracy_score(y_te, model.predict(X_te)):.4f}")

ranked = sorted(zip(wine.feature_names, model.feature_importances_), key=lambda t: -t[1])
print("상위 5개 중요 변수:")
for name, score in ranked[:5]:
    print(f"  {name:<28s} {score:.4f}")
