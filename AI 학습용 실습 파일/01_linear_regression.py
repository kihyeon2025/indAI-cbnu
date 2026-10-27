"""01. 선형 회귀 (Linear Regression) """
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_tr, y_tr)
pred = model.predict(X_te)

print(f"MSE: {mean_squared_error(y_te, pred):.2f}")
print(f"R2 : {r2_score(y_te, pred):.4f}")
