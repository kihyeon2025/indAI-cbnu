"""03. 의사결정나무 (Decision Tree)"""
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris()
X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
print(f"Accuracy: {accuracy_score(y_te, model.predict(X_te)):.4f}")
print(export_text(model, feature_names=list(iris.feature_names)))
