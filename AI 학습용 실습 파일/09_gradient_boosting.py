"""09. 그래디언트 부스팅 (Gradient Boosting) 회귀 """
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_tr, y_tr)
pred = model.predict(X_te)

print(f"MAE: {mean_absolute_error(y_te, pred):.2f}")
print(f"R2 : {r2_score(y_te, pred):.4f}")
