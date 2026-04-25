import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

# Step 1: Create dataset
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + 5 + np.random.randn(100) * 2

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Polynomial features (high degree for overfitting)
poly = PolynomialFeatures(degree=10)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Step 4: Linear Regression (Overfitting)
lr = LinearRegression()
lr.fit(X_train_poly, y_train)

y_train_pred = lr.predict(X_train_poly)
y_test_pred = lr.predict(X_test_poly)

print("----- Linear Regression (Overfitting) -----")
print("Train R2 Score:", r2_score(y_train, y_train_pred))
print("Test R2 Score:", r2_score(y_test, y_test_pred))


# Step 5: Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly, y_train)

y_train_pred_ridge = ridge.predict(X_train_poly)
y_test_pred_ridge = ridge.predict(X_test_poly)

print("\n----- Ridge Regression -----")
print("Train R2 Score:", r2_score(y_train, y_train_pred_ridge))
print("Test R2 Score:", r2_score(y_test, y_test_pred_ridge))


# Step 6: Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_poly, y_train)

y_train_pred_lasso = lasso.predict(X_train_poly)
y_test_pred_lasso = lasso.predict(X_test_poly)

print("\n----- Lasso Regression -----")
print("Train R2 Score:", r2_score(y_train, y_train_pred_lasso))
print("Test R2 Score:", r2_score(y_test, y_test_pred_lasso))

