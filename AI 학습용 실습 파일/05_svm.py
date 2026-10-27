"""05. 서포트 벡터 머신 (SVM) - 손글씨 숫자 데이터셋"""
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
model.fit(X_tr, y_tr)
print(f"Accuracy: {accuracy_score(y_te, model.predict(X_te)):.4f}")
